#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::Float64Array;
    use datafusion::arrow::compute::concat_batches;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::error::Result;
    use datafusion::physical_plan::{ExecutionPlan, collect};
    use datafusion_distributed_iceberg::IcebergExt;
    use datafusion_distributed_iceberg_test_tools::{
        FixtureCatalog, FixtureTable, IcebergTestHarness, TAXI_DAYS, TAXI_TRIPS_PER_DAY,
        taxi_schema, taxi_trips,
    };
    use iceberg::TableIdent;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use parquet::file::properties::{EnabledStatistics, WriterProperties};

    /// One unpartitioned file holding every day, in row groups of two days each. The join's
    /// build side keeps one day, so its dynamic filter on `pickup_date` reaches the probe scan.
    /// With pruning on and the default `datafusion.execution.parquet.pushdown_filters=false`,
    /// the reader skips the three row groups without that day but doesn't filter rows inside
    /// the one that has it: the probe scan outputs that row group's two days, and the join
    /// drops the other day's rows. No file is skipped and the result is unchanged.
    #[tokio::test]
    async fn prunes_row_groups_but_leaves_rows_to_the_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_two_days_per_row_group_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_two_days_per_row_group_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- output_rows=25000
        files_skipped=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- output_rows=25000
        files_skipped=0 output_rows=50000
        ");
        let rows = pretty_format_batches(&on_rows)?.to_string();
        assert_eq!(pretty_format_batches(&off_rows)?.to_string(), rows);
        insta::assert_snapshot!(rows, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Like [prunes_row_groups_but_leaves_rows_to_the_join], with
    /// `datafusion.execution.parquet.pushdown_filters=true`: the reader also evaluates the
    /// dynamic filter row by row, so the probe scan outputs only the joined day.
    #[tokio::test]
    async fn pushdown_filters_also_filters_rows_in_the_scan() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_two_days_per_row_group_taxi_table().await?)
            .with_config_option("datafusion.execution.parquet.pushdown_filters", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let plan = harness.physical_plan(JOIN_QUERY).await?;
        let rows = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&plan), @"
        files_skipped=- output_rows=25000
        files_skipped=0 output_rows=25000
        ");
        insta::assert_snapshot!(pretty_format_batches(&rows)?, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Parquet statistics leave NaN out of a float column's min and max, so a row group holding
    /// `[1.0, NaN]` has max 1.0. The table has two files, read one at a time in order: the
    /// first holds `1.5`, so once the TopK has read it its dynamic filter is `value > 1.5`.
    /// NaN sorts above every number, so `value > 1.5` keeps the NaN row. The reader must not
    /// prune the second file's row group by its max: the NaN row is still the result.
    #[tokio::test]
    async fn float_bound_keeps_nan_rows() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_values_table(&[&[1.5], &[1.0, f64::NAN]]).await?)
            .with_config_option("datafusion.execution.target_partitions", 1)
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let plan = harness
            .physical_plan("SELECT value FROM t ORDER BY value DESC LIMIT 1")
            .await?;
        let rows = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;

        insta::assert_snapshot!(pretty_format_batches(&rows)?, @"
        +-------+
        | value |
        +-------+
        | NaN   |
        +-------+
        ");
        Ok(())
    }

    /// Distributed over the table with two days per row group, with and without
    /// `pushdown_filters`. Worker scans may open the file before the filter has a value, so
    /// what gets pruned varies; only the result is checked.
    #[tokio::test]
    async fn distributed_join_over_multi_day_row_groups_returns_same_rows() -> Result<()> {
        let (_, pruning_only) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_two_days_per_row_group_taxi_table().await?)
            .with_workers(2)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(JOIN_QUERY)
            .await?;
        let (_, row_filtered) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_two_days_per_row_group_taxi_table().await?)
            .with_workers(2)
            .with_config_option("datafusion.execution.parquet.pushdown_filters", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(JOIN_QUERY)
            .await?;

        assert_eq!(pruning_only, row_filtered);
        insta::assert_snapshot!(pruning_only, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// The build side is one pickup date, so the join pushes a dynamic filter on
    /// `pickup_date` into the probe-side scan of the whole table.
    const JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM (SELECT DISTINCT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10') days
        JOIN taxi ON taxi.pickup_date = days.pickup_date";

    /// Each Iceberg scan's `files_skipped` and output rows, in plan order.
    fn scan_metrics(plan: &Arc<dyn ExecutionPlan>) -> String {
        let mut lines = Vec::new();
        plan.apply(|node| {
            if node.downcast_ref::<DataSourceExec>().is_some() {
                let metrics = node.metrics().unwrap_or_default();
                let skipped = metrics
                    .sum_by_name("files_skipped")
                    .map_or("-".to_string(), |m| m.as_usize().to_string());
                let rows = metrics.output_rows().unwrap_or_default();
                lines.push(format!("files_skipped={skipped} output_rows={rows}"));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("visiting the plan doesn't fail");
        lines.join("\n")
    }

    /// Table `default.taxi`, unpartitioned, with every day's trips in one file, in date order,
    /// written in row groups of two days each (four row groups) with column chunk statistics.
    async fn create_two_days_per_row_group_taxi_table() -> Result<FixtureTable> {
        let ident = TableIdent::from_strs(["default", "taxi"]).expect("valid identifier");
        let mut table = FixtureCatalog::try_new()
            .await?
            .table(ident, taxi_schema())
            .create()
            .await?;
        let days = (0..TAXI_DAYS).map(taxi_trips).collect::<Result<Vec<_>>>()?;
        let trips = concat_batches(&days[0].schema(), &days)?;
        table
            .append()
            .writer_properties(
                WriterProperties::builder()
                    .set_max_row_group_row_count(Some(2 * TAXI_TRIPS_PER_DAY))
                    .set_statistics_enabled(EnabledStatistics::Chunk)
                    .build(),
            )
            .file(trips)
            .commit()
            .await?;
        Ok(table)
    }

    /// Table `default.t` with one required `double` column, `value`, and one data file (of one
    /// row group) per entry of `files`, appended in order.
    async fn create_values_table(files: &[&[f64]]) -> Result<FixtureTable> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "value", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()
            .expect("valid schema");
        let ident = TableIdent::from_strs(["default", "t"]).expect("valid identifier");
        let mut table = FixtureCatalog::try_new()
            .await?
            .table(ident, schema)
            .create()
            .await?;
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let mut append = table.append();
        for values in files {
            append = append.file(RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![Arc::new(Float64Array::from(values.to_vec()))],
            )?);
        }
        append.commit().await?;
        Ok(table)
    }
}
