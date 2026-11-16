mod tests {
    use datafusion::arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::common::Result;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::physical_plan::collect;
    use datafusion_distributed::test_utils::metrics::node_metrics;
    use datafusion_distributed::{
        DistributedExt, DistributedLeafExec, DistributedMetricsFormat,
        rewrite_distributed_plan_with_metrics,
    };
    use datafusion_distributed_iceberg::IcebergExt;
    use datafusion_distributed_iceberg_test_tools::{
        FixtureCatalog, FixtureTable, IcebergTestHarness,
    };
    use iceberg::TableIdent;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use std::error::Error;
    use std::sync::Arc;

    // `ids` has one file each for ids 1-10, 11-20 and 21-30. The join keeps ids 12 and 15, so
    // only the second file's bounds can hold a match.
    const TWO_IDS: &str = "SELECT COUNT(*) AS n FROM ids \
        JOIN (VALUES (CAST(12 AS BIGINT)), (CAST(15 AS BIGINT))) AS k(id) ON ids.id = k.id";

    #[tokio::test]
    async fn workers_skip_files_whose_bounds_the_join_filter_rules_out()
    -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(ids().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            .with_config_option("iceberg.column_stats_enabled", true)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1)
            })?
            .build()
            .await?;

        let plan = harness.physical_plan(TWO_IDS).await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +---+
        | n |
        +---+
        | 2 |
        +---+
        ");
        assert_eq!(
            node_metrics::<DistributedLeafExec>(&plan, "files_opened", 0),
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn files_are_read_when_the_scan_has_no_column_statistics() -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(ids().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            .with_config_option("iceberg.column_stats_enabled", false)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1)
            })?
            .build()
            .await?;

        let plan = harness.physical_plan(TWO_IDS).await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +---+
        | n |
        +---+
        | 2 |
        +---+
        ");
        assert_eq!(
            node_metrics::<DistributedLeafExec>(&plan, "files_opened", 0),
            3
        );
        Ok(())
    }

    #[tokio::test]
    async fn files_with_nan_are_read_even_when_their_bounds_rule_them_out()
    -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(readings().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            .with_config_option("iceberg.column_stats_enabled", true)
            // Read one file at a time, so top-N's filter tightens before the next file starts.
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 1;
                Ok(state)
            })?
            .build()
            .await?;

        let plan = harness
            .physical_plan("SELECT v FROM readings ORDER BY v DESC LIMIT 1")
            .await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        // DataFusion orders NaN above every number. After the first file, the filter is
        // `v > 5`. The second file's bounds are [1, 1], since Iceberg leaves NaN out of them,
        // but the file holds a NaN and must be read.
        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +-----+
        | v   |
        +-----+
        | NaN |
        +-----+
        ");
        assert_eq!(node_metrics::<DataSourceExec>(&plan, "files_opened", 0), 2);
        Ok(())
    }

    // Table `ids`, with one file each for ids 1-10, 11-20 and 21-30.
    async fn ids() -> Result<FixtureTable, Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "ids"])?, schema)
            .create()
            .await?;
        let mut append = table.append();
        for first in [1, 11, 21] {
            let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(first..first + 10));
            append = append.file(RecordBatch::try_from_iter([("id", ids)])?);
        }
        append.commit().await?;
        Ok(table)
    }

    // Table `readings`, with one file holding 5, then one holding 1 and NaN.
    async fn readings() -> Result<FixtureTable, Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "v", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "readings"])?, schema)
            .create()
            .await?;
        let mut append = table.append();
        for values in [vec![5.0, 4.0, 3.0, 2.0], vec![1.0, f64::NAN]] {
            let values: ArrayRef = Arc::new(Float64Array::from(values));
            append = append.file(RecordBatch::try_from_iter([("v", values)])?);
        }
        append.commit().await?;
        Ok(table)
    }
}
