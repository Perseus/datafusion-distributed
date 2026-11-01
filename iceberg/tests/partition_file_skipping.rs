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
        FixtureCatalog, IcebergTestHarness, create_taxi_table, taxi_schema, taxi_trips,
    };
    use iceberg::TableIdent;
    use iceberg::spec::Transform;

    /// The taxi table has one file per `pickup_date`, partitioned by identity on it. The
    /// semi-join's build side keeps one day, so the hash join (CollectLeft) pushes a dynamic
    /// filter on that day into the probe scan. With the setting on, the probe scan drops the
    /// other six files by their partition value before reading them. The rows are the same.
    #[tokio::test]
    async fn lazy_scan_skips_files_by_partition_value_in_semi_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(SEMI_JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(SEMI_JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- output_rows=25000
        files_skipped=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- output_rows=25000
        files_skipped=6 output_rows=25000
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

    /// Like [lazy_scan_skips_files_by_partition_value_in_semi_join], for files planned eagerly.
    #[tokio::test]
    async fn eager_scan_skips_files_by_partition_value_in_semi_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(SEMI_JOIN_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(SEMI_JOIN_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"
        files_skipped=- output_rows=25000
        files_skipped=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- output_rows=25000
        files_skipped=6 output_rows=25000
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
    /// `CASE hash(pickup_date) % 4 WHEN i THEN ...`. The build side keeps two days. The routing
    /// `CASE` is compacted to those two days, so the other five files are skipped.
    #[tokio::test]
    async fn lazy_scan_skips_files_by_partition_value_in_partitioned_join() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
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
            .with_table(create_taxi_table().await?)
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
        files_skipped=- output_rows=50000
        files_skipped=- output_rows=175000
        ");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"
        files_skipped=- output_rows=50000
        files_skipped=5 output_rows=50000
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

    /// TopK pushes a dynamic filter on its current threshold into the scan and tightens it as
    /// it sees smaller dates. The files are appended out of date order: 2024-01-10, then
    /// 2024-01-08, then the rest. After the first file the threshold is 2024-01-10, which
    /// keeps 2024-01-08's file; after that file it is 2024-01-08, and the five later files
    /// are skipped. One partition and one file read at a time make the order deterministic:
    /// the reader pulls the next task only after the previous file's last batch.
    #[tokio::test]
    async fn topk_skips_files_once_its_threshold_tightens() -> Result<()> {
        let off = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table_out_of_date_order().await?)
            .with_config_option("datafusion.execution.target_partitions", 1)
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table_out_of_date_order().await?)
            .with_config_option("datafusion.execution.target_partitions", 1)
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let off_plan = off.physical_plan(TOPK_QUERY).await?;
        let off_rows = collect(Arc::clone(&off_plan), off.state().task_ctx()).await?;
        let on_plan = on.physical_plan(TOPK_QUERY).await?;
        let on_rows = collect(Arc::clone(&on_plan), on.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&off_plan), @"files_skipped=- output_rows=175000");
        insta::assert_snapshot!(scan_metrics(&on_plan), @"files_skipped=5 output_rows=50000");
        let rows = pretty_format_batches(&on_rows)?.to_string();
        assert_eq!(pretty_format_batches(&off_rows)?.to_string(), rows);
        insta::assert_snapshot!(rows, @"
        +-------------+---------------------+
        | pickup_date | pickup_at           |
        +-------------+---------------------+
        | 2024-01-08  | 2024-01-08T00:00:02 |
        | 2024-01-08  | 2024-01-08T00:00:05 |
        | 2024-01-08  | 2024-01-08T00:00:08 |
        +-------------+---------------------+
        ");
        Ok(())
    }

    /// Executing the same plan again reuses the hash join's build side, so its dynamic filter
    /// reaches the same value and the shared, latched value skips the same six files again
    /// (the metric adds up across both executions). The rows are the same both times.
    #[tokio::test]
    async fn same_plan_executed_twice_skips_the_same_files() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let plan = harness.physical_plan(SEMI_JOIN_QUERY).await?;
        let first = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let second = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;

        insta::assert_snapshot!(scan_metrics(&plan), @"
        files_skipped=- output_rows=25000
        files_skipped=12 output_rows=50000
        ");
        let rows = pretty_format_batches(&second)?.to_string();
        assert_eq!(pretty_format_batches(&first)?.to_string(), rows);
        insta::assert_snapshot!(rows, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// A recursive query re-executes its recursive term once per iteration after
    /// `reset_plan_states`, which in DataFusion 55 drops the hash join's dynamic filter before
    /// the first iteration too, so the probe scan's filter never receives a value. Here the
    /// Iceberg scan is the join's probe side in every iteration, and each iteration needs a
    /// different day: the rows match the setting off. Lazy scans can't be re-executed, so the
    /// files are planned eagerly.
    #[tokio::test]
    async fn recursive_query_reexecuting_the_probe_scan_returns_same_rows() -> Result<()> {
        let (_, off_rows) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .query(RECURSIVE_QUERY)
            .await?;
        let (_, on_rows) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(RECURSIVE_QUERY)
            .await?;

        assert_eq!(off_rows, on_rows);
        insta::assert_snapshot!(on_rows, @"
        +------------+
        | d          |
        +------------+
        | 2024-01-08 |
        | 2024-01-09 |
        | 2024-01-10 |
        | 2024-01-11 |
        | 2024-01-12 |
        +------------+
        ");
        Ok(())
    }

    /// Distributed, worker probe scans may start before the coordinator forwards the dynamic
    /// filter's value, so how many files they skip varies: only the rows are checked.
    #[tokio::test]
    async fn distributed_semi_join_with_file_skipping_returns_same_rows() -> Result<()> {
        let (_, rows) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
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

    /// Walks one day at a time from 2024-01-08, probing the taxi table for the next day. The
    /// `+ INTERVAL '0 days'` drops the Iceberg field id, which the recursive term's output
    /// must not carry.
    const RECURSIVE_QUERY: &str = r"WITH RECURSIVE days(d) AS (
            SELECT DATE '2024-01-08'
            UNION ALL
            SELECT DISTINCT t.pickup_date + INTERVAL '0 days'
            FROM days JOIN taxi t ON t.pickup_date = days.d + INTERVAL '1 day'
            WHERE days.d < DATE '2024-01-12'
        )
        SELECT d FROM days ORDER BY d";

    /// Pickup times are distinct within a day, so the order is total.
    const TOPK_QUERY: &str = r"SELECT pickup_date, pickup_at
        FROM taxi
        ORDER BY pickup_date, pickup_at
        LIMIT 3";

    /// For each scan in `plan`, its `files_skipped` (`-` when it doesn't skip files) and its
    /// output rows.
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

    /// The taxi table with one file per day, appended 2024-01-10 first, then 2024-01-08,
    /// then the other days in date order.
    async fn create_taxi_table_out_of_date_order()
    -> Result<datafusion_distributed_iceberg_test_tools::FixtureTable> {
        let fixtures = FixtureCatalog::try_new().await?;
        let ident = TableIdent::from_strs(["default", "taxi"]).expect("valid identifier");
        let mut table = fixtures
            .table(ident, taxi_schema())
            .partition_by("pickup_date", Transform::Identity)
            .create()
            .await?;
        let mut append = table.append();
        for day in [2, 0, 1, 3, 4, 5, 6] {
            append = append.file(taxi_trips(day)?);
        }
        append.commit().await?;
        Ok(table)
    }
}
