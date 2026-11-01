#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::error::Result;
    use datafusion::physical_plan::{ExecutionPlan, collect};
    use datafusion_distributed_iceberg::IcebergExt;
    use datafusion_distributed_iceberg_test_tools::{
        FixtureCatalog, FixtureTable, IcebergTestHarness, TAXI_DAYS, taxi_schema, taxi_trips,
    };
    use iceberg::TableIdent;

    /// The table isn't partitioned, so no file can be skipped by a partition value, but each
    /// file holds one `pickup_date` and its manifest bounds say which. The semi-join's build
    /// side keeps one day, so the hash join (CollectLeft) pushes a dynamic filter on that day
    /// into the probe scan. With the setting on, the probe scan skips the other six files by
    /// their column bounds. The rows are the same.
    #[tokio::test]
    async fn lazy_scan_skips_unpartitioned_files_by_bounds_in_semi_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(SEMI_JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(SEMI_JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=25000
        files_skipped=- files_skipped_by_bounds=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=25000
        files_skipped=6 files_skipped_by_bounds=6 output_rows=25000
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

    /// Like [lazy_scan_skips_unpartitioned_files_by_bounds_in_semi_join], for files planned
    /// eagerly: the column statistics come with the planned file tasks.
    #[tokio::test]
    async fn eager_scan_skips_unpartitioned_files_by_bounds_in_semi_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(SEMI_JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(SEMI_JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=25000
        files_skipped=- files_skipped_by_bounds=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=25000
        files_skipped=6 files_skipped_by_bounds=6 output_rows=25000
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

    /// With the single-partition thresholds at 0 the join is partitioned, so the probe scan's
    /// dynamic filter routes each row to its build partition's bounds with
    /// `CASE hash(pickup_date) % 4 WHEN i THEN ...`. The build side keeps two days, and the
    /// compacted filter rules out the other five files by their column bounds.
    #[tokio::test]
    async fn lazy_scan_skips_unpartitioned_files_by_bounds_in_partitioned_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("datafusion.execution.target_partitions", 4)
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold",
                0,
            )
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold_rows",
                0,
            )
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("datafusion.execution.target_partitions", 4)
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold",
                0,
            )
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold_rows",
                0,
            )
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(PARTITIONED_JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(PARTITIONED_JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=50000
        files_skipped=- files_skipped_by_bounds=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=50000
        files_skipped=5 files_skipped_by_bounds=5 output_rows=50000
        ");
        let rows = pretty_format_batches(&on_rows)?.to_string();
        assert_eq!(pretty_format_batches(&off_rows)?.to_string(), rows);
        insta::assert_snapshot!(rows, @"
        +-------+
        | trips |
        +-------+
        | 50000 |
        +-------+
        ");
        Ok(())
    }

    /// Like [lazy_scan_skips_unpartitioned_files_by_bounds_in_partitioned_join], for files
    /// planned eagerly.
    #[tokio::test]
    async fn eager_scan_skips_unpartitioned_files_by_bounds_in_partitioned_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("datafusion.execution.target_partitions", 4)
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold",
                0,
            )
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold_rows",
                0,
            )
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("datafusion.execution.target_partitions", 4)
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold",
                0,
            )
            .with_config_option(
                "datafusion.optimizer.hash_join_single_partition_threshold_rows",
                0,
            )
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(PARTITIONED_JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(PARTITIONED_JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=50000
        files_skipped=- files_skipped_by_bounds=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- files_skipped_by_bounds=- output_rows=50000
        files_skipped=5 files_skipped_by_bounds=5 output_rows=50000
        ");
        let rows = pretty_format_batches(&on_rows)?.to_string();
        assert_eq!(pretty_format_batches(&off_rows)?.to_string(), rows);
        insta::assert_snapshot!(rows, @"
        +-------+
        | trips |
        +-------+
        | 50000 |
        +-------+
        ");
        Ok(())
    }

    /// Distributed, the file tasks and their column statistics travel to the workers. Worker
    /// probe scans may start before the coordinator forwards the dynamic filter's value, so
    /// how many files they skip varies: only the rows are checked.
    #[tokio::test]
    async fn distributed_unpartitioned_semi_join_with_bound_skipping_returns_same_rows()
    -> Result<()> {
        let (_, rows) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_unpartitioned_taxi_table().await?)
            .with_workers(2)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(SEMI_JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(rows, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    const SEMI_JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM taxi
        WHERE pickup_date IN (SELECT DISTINCT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10')";

    /// Without DISTINCT the build side isn't small, so the join can be partitioned.
    const PARTITIONED_JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM taxi
        WHERE pickup_date IN (
            SELECT pickup_date FROM taxi
            WHERE pickup_date IN (DATE '2024-01-08', DATE '2024-01-14')
        )";

    /// For each scan in `plan`: all the files it skipped, those only their column bounds ruled
    /// out (`-` when it doesn't skip files), and its output rows.
    fn scan_metrics(plan: &Arc<dyn ExecutionPlan>) -> String {
        let mut lines = Vec::new();
        plan.apply(|node| {
            if node.downcast_ref::<DataSourceExec>().is_some() {
                let metrics = node.metrics().unwrap_or_default();
                let count = |name| {
                    metrics
                        .sum_by_name(name)
                        .map_or("-".to_string(), |m| m.as_usize().to_string())
                };
                let rows = metrics.output_rows().unwrap_or_default();
                lines.push(format!(
                    "files_skipped={} files_skipped_by_bounds={} output_rows={rows}",
                    count("files_skipped"),
                    count("files_skipped_by_bounds"),
                ));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("visiting the plan doesn't fail");
        lines.join("\n")
    }

    /// Table `default.taxi` with one file per day, like the taxi fixture, but without a
    /// partition spec.
    async fn create_unpartitioned_taxi_table() -> Result<FixtureTable> {
        let ident = TableIdent::from_strs(["default", "taxi"]).expect("valid identifier");
        let mut table = FixtureCatalog::try_new()
            .await?
            .table(ident, taxi_schema())
            .create()
            .await?;
        let mut append = table.append();
        for day in 0..TAXI_DAYS {
            append = append.file(taxi_trips(day)?);
        }
        append.commit().await?;
        Ok(table)
    }
}
