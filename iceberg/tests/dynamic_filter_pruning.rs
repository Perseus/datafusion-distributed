#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::error::Result;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::physical_plan::joins::HashJoinExec;
    use datafusion::physical_plan::{collect, displayable};
    use datafusion_distributed::DistributedExt;
    use datafusion_distributed_iceberg::{IcebergExt, IcebergIntegrationOptions};
    use datafusion_distributed_iceberg_test_tools::{
        CountingStorageFactory, IcebergTestHarness, create_taxi_table,
    };

    /// Single node, so the join's build side is complete before the probe scan starts reading
    /// and the dynamic filter's value is known when every probe file opens. With pruning on,
    /// the probe scan reads only the row groups of the joined day: one of the seven files'
    /// worth of rows instead of all of them. The join's result is the same either way.
    #[tokio::test]
    async fn lazy_scan_prunes_with_dynamic_filter() -> Result<()> {
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

        let (off_scans, off_batches) = run_scans(&off).await?;
        let (on_scans, on_batches) = run_scans(&on).await?;

        insta::assert_snapshot!(off_scans, @"
        output_rows=25000: DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], predicate=pickup_date = 2024-01-10, physical_predicate=pickup_date@0 = 2024-01-10
        output_rows=175000: DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], physical_predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        ");
        insta::assert_snapshot!(on_scans, @"
        output_rows=25000: DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], predicate=pickup_date = 2024-01-10, physical_predicate=pickup_date@0 = 2024-01-10
        output_rows=25000: DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], physical_predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        ");
        assert_eq!(off_batches, on_batches);
        insta::assert_snapshot!(on_batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Like [lazy_scan_prunes_with_dynamic_filter], for a scan whose files are planned eagerly.
    #[tokio::test]
    async fn eager_scan_prunes_with_dynamic_filter() -> Result<()> {
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

        let (off_scans, off_batches) = run_scans(&off).await?;
        let (on_scans, on_batches) = run_scans(&on).await?;

        insta::assert_snapshot!(off_scans, @"
        output_rows=25000: DataSourceExec: format=iceberg, planned_files=1, predicate=pickup_date@0 = 2024-01-10
        output_rows=175000: DataSourceExec: format=iceberg, planned_files=7, predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        ");
        insta::assert_snapshot!(on_scans, @"
        output_rows=25000: DataSourceExec: format=iceberg, planned_files=1, predicate=pickup_date@0 = 2024-01-10
        output_rows=25000: DataSourceExec: format=iceberg, planned_files=7, predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        ");
        assert_eq!(off_batches, on_batches);
        insta::assert_snapshot!(on_batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Distributed, the worker probe scans may start before the coordinator forwards the
    /// dynamic filter's value, so only the result is checked: it matches the unpruned result.
    #[tokio::test]
    async fn distributed_lazy_scan_with_dynamic_filter_pruning_returns_same_rows() -> Result<()> {
        let (_, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1_000_000)
            })?
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Like [distributed_lazy_scan_with_dynamic_filter_pruning_returns_same_rows], for eagerly
    /// planned files.
    #[tokio::test]
    async fn distributed_eager_scan_with_dynamic_filter_pruning_returns_same_rows() -> Result<()> {
        let (_, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_workers(2)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Single node, over the taxi table, partitioned by identity on `pickup_date` with one file
    /// per day. The semi-join's build side is the one day the static filter keeps, so its
    /// dynamic filter rules out the other six days' files by their partition value. With pruning
    /// on, the probe scan skips those files without opening them: the build scan reads its one
    /// file, and the probe scan reads one file instead of seven. The result is the same.
    #[tokio::test]
    async fn lazy_scan_skips_files_by_partition_value() -> Result<()> {
        let off_storage = CountingStorageFactory::new();
        let off = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&off_storage))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on_storage = CountingStorageFactory::new();
        let on = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&on_storage))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let (_, off_batches) = off.query(SEMI_JOIN_QUERY).await?;
        let (_, on_batches) = on.query(SEMI_JOIN_QUERY).await?;

        insta::assert_snapshot!(
            data_file_reads(&off_storage, &on_storage),
            @"
        data files read with pruning off: 8
        data files read with pruning on: 2
        "
        );
        assert_eq!(off_batches, on_batches);
        insta::assert_snapshot!(on_batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Like [lazy_scan_skips_files_by_partition_value], for a scan whose files are planned
    /// eagerly.
    #[tokio::test]
    async fn eager_scan_skips_files_by_partition_value() -> Result<()> {
        let off_storage = CountingStorageFactory::new();
        let off = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&off_storage))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", false)
            .build()
            .await?;
        let on_storage = CountingStorageFactory::new();
        let on = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&on_storage))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .build()
            .await?;

        let (_, off_batches) = off.query(SEMI_JOIN_QUERY).await?;
        let (_, on_batches) = on.query(SEMI_JOIN_QUERY).await?;

        insta::assert_snapshot!(
            data_file_reads(&off_storage, &on_storage),
            @"
        data files read with pruning off: 8
        data files read with pruning on: 2
        "
        );
        assert_eq!(off_batches, on_batches);
        insta::assert_snapshot!(on_batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Distributed, the worker probe scans may start before the dynamic filter has a value, so
    /// which files get skipped varies; only the result is checked.
    #[tokio::test]
    async fn distributed_semi_join_with_file_skipping_returns_same_rows() -> Result<()> {
        let (_, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_workers(2)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(SEMI_JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// Keeps the trips on the one day the build side selects.
    const SEMI_JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM taxi
        WHERE pickup_date IN (SELECT DISTINCT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10')";

    /// A partitioned hash join builds one filter per build partition and routes each probe row
    /// to its partition's filter with a `CASE` on the row's hash. With pruning on, that filter
    /// still prunes row groups and skips whole files. The join's result is the same either way.
    #[tokio::test]
    async fn lazy_scan_prunes_with_partitioned_join_filter() -> Result<()> {
        let off_storage = CountingStorageFactory::new();
        let off = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&off_storage))
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
        let on_storage = CountingStorageFactory::new();
        let on = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&on_storage))
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

        let off_run = run_partitioned_join(&off).await?;
        let on_run = run_partitioned_join(&on).await?;

        insta::assert_snapshot!(off_run.join_mode, @"Partitioned");
        insta::assert_snapshot!(
            format!("pruning off: {}\npruning on: {}", off_run.probe_rows, on_run.probe_rows),
            @"
        pruning off: 175000
        pruning on: 75000
        "
        );
        insta::assert_snapshot!(data_file_reads(&off_storage, &on_storage), @"
        data files read with pruning off: 9
        data files read with pruning on: 5
        ");
        assert_eq!(off_run.result, on_run.result);
        insta::assert_snapshot!(on_run.result, @"
        +-------+
        | trips |
        +-------+
        | 50000 |
        +-------+
        ");
        Ok(())
    }

    /// Like [lazy_scan_prunes_with_partitioned_join_filter], for a scan whose files are planned
    /// eagerly.
    #[tokio::test]
    async fn eager_scan_prunes_with_partitioned_join_filter() -> Result<()> {
        let off_storage = CountingStorageFactory::new();
        let off = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&off_storage))
            .with_table(create_taxi_table().await?)
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
        let on_storage = CountingStorageFactory::new();
        let on = IcebergTestHarness::builder()
            .with_session_extension(iceberg_integration(&on_storage))
            .with_table(create_taxi_table().await?)
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

        let off_run = run_partitioned_join(&off).await?;
        let on_run = run_partitioned_join(&on).await?;

        insta::assert_snapshot!(off_run.join_mode, @"Partitioned");
        insta::assert_snapshot!(
            format!("pruning off: {}\npruning on: {}", off_run.probe_rows, on_run.probe_rows),
            @"
        pruning off: 175000
        pruning on: 75000
        "
        );
        insta::assert_snapshot!(data_file_reads(&off_storage, &on_storage), @"
        data files read with pruning off: 9
        data files read with pruning on: 5
        ");
        assert_eq!(off_run.result, on_run.result);
        insta::assert_snapshot!(on_run.result, @"
        +-------+
        | trips |
        +-------+
        | 50000 |
        +-------+
        ");
        Ok(())
    }

    /// Distributed, the coordinator ORs together the routing filters of every producer task.
    /// Worker probe scans may start before that filter has a value, so only the result is
    /// checked.
    #[tokio::test]
    async fn distributed_partitioned_join_with_pruning_returns_same_rows() -> Result<()> {
        let (_, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_workers(2)
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
            .query(PARTITIONED_JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 50000 |
        +-------+
        ");
        Ok(())
    }

    /// Keeps the trips on the two days the build side selects. Without DISTINCT the build side
    /// isn't small, so with the single-partition thresholds at 0 the join is partitioned.
    const PARTITIONED_JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM taxi
        WHERE pickup_date IN (
            SELECT pickup_date FROM taxi
            WHERE pickup_date IN (DATE '2024-01-10', DATE '2024-01-12')
        )";

    struct PartitionedJoinRun {
        /// The hash join's `mode=` in the plan.
        join_mode: String,
        /// Rows output by the join's probe-side scan.
        probe_rows: usize,
        result: String,
    }

    /// Runs [PARTITIONED_JOIN_QUERY] and reports the join mode, the probe scan's output rows
    /// and the result.
    async fn run_partitioned_join(harness: &IcebergTestHarness) -> Result<PartitionedJoinRun> {
        let plan = harness.physical_plan(PARTITIONED_JOIN_QUERY).await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let mut join_mode = String::new();
        let mut probe_rows = 0;
        plan.apply(|node| {
            let Some(join) = node.downcast_ref::<HashJoinExec>() else {
                return Ok(TreeNodeRecursion::Continue);
            };
            join_mode = format!("{:?}", join.mode);
            join.right().apply(|probe| {
                if probe.downcast_ref::<DataSourceExec>().is_none() {
                    return Ok(TreeNodeRecursion::Continue);
                }
                probe_rows = probe
                    .metrics()
                    .and_then(|m| m.output_rows())
                    .unwrap_or_default();
                Ok(TreeNodeRecursion::Stop)
            })?;
            Ok(TreeNodeRecursion::Stop)
        })?;
        Ok(PartitionedJoinRun {
            join_mode,
            probe_rows,
            result: datafusion::arrow::util::pretty::pretty_format_batches(&batches)?.to_string(),
        })
    }

    /// Registers the Iceberg integration reading files through `storage`.
    fn iceberg_integration(
        storage: &CountingStorageFactory,
    ) -> impl Fn(SessionStateBuilder) -> SessionStateBuilder + Send + Sync + 'static {
        let storage = storage.clone();
        move |state| {
            state.with_iceberg_integration(IcebergIntegrationOptions {
                storage_factory: Arc::new(storage.clone()),
                ..Default::default()
            })
        }
    }

    fn data_file_reads(off: &CountingStorageFactory, on: &CountingStorageFactory) -> String {
        format!(
            "data files read with pruning off: {}\ndata files read with pruning on: {}",
            off.data_file_reads(),
            on.data_file_reads()
        )
    }

    /// The build side is one pickup date, so the join pushes a dynamic filter on
    /// `pickup_date` into the probe-side scan of the whole table, one file per day.
    const JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM (SELECT DISTINCT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10') days
        JOIN taxi ON taxi.pickup_date = days.pickup_date";

    /// Runs [JOIN_QUERY] and returns, for each Iceberg scan, its display after execution
    /// (showing the dynamic filter's final value) and the rows it read, plus the result.
    async fn run_scans(harness: &IcebergTestHarness) -> Result<(String, String)> {
        let plan = harness.physical_plan(JOIN_QUERY).await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let mut scans = Vec::new();
        plan.apply(|node| {
            if node.downcast_ref::<DataSourceExec>().is_some() {
                let rows = node
                    .metrics()
                    .and_then(|m| m.output_rows())
                    .unwrap_or_default();
                let display = displayable(node.as_ref()).one_line().to_string();
                scans.push(format!("output_rows={rows}: {}", display.trim_end()));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok((
            scans.join("\n"),
            datafusion::arrow::util::pretty::pretty_format_batches(&batches)?.to_string(),
        ))
    }
}
