mod tests {
    use datafusion::common::Result;
    use datafusion_distributed::DistributedExt;
    use datafusion_distributed_iceberg::IcebergExt;
    use datafusion_distributed_iceberg_test_tools::IcebergTestHarness;
    use datafusion_distributed_iceberg_test_tools::create_taxi_table;

    #[tokio::test]
    async fn lazy_planning_workers_report_files_opened_and_bytes_scanned_per_task() -> Result<()> {
        // The taxi table has one file per day, so the filter selects three files. The
        // coordinator lists them and feeds them to the two tasks.
        let (plan, _) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_workers(2)
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 2;
                state.with_distributed_file_scan_config_bytes_per_partition(1)
            })?
            .query_with_metrics(
                "SELECT pickup_date, COUNT(*) FROM taxi \
                 WHERE pickup_date <= DATE '2024-01-10' GROUP BY pickup_date",
            )
            .await?;

        insta::with_settings!({
            filters => [(r"elapsed_compute=\{[^}]*\}, ", "")],
        }, {
            insta::assert_snapshot!(scan_lines(&plan), @"DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], predicate=pickup_date <= 2024-01-10, filters=pickup_date@0 <= 2024-01-10, metrics=[output_rows={0:50.00 K, 1:25.00 K}, output_bytes={0:256.0 KB, 1:128.0 KB}, output_batches={0:8, 1:4}, batches_split={0:0, 1:0}, bytes_scanned={0:1.05 M, 1:524.4 K}, files_opened={0:2, 1:1}]");
        });
        Ok(())
    }

    #[tokio::test]
    async fn eager_planning_workers_report_files_opened_and_bytes_scanned_per_task() -> Result<()> {
        // The taxi table has one file per day, so the filter selects three files. The
        // coordinator splits them between the two tasks when planning, and each task
        // gets its own scan.
        let (plan, _) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_workers(2)
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 2;
                state.with_distributed_file_scan_config_bytes_per_partition(1)
            })?
            .query_with_metrics(
                "SELECT pickup_date, COUNT(*) FROM taxi \
                 WHERE pickup_date <= DATE '2024-01-10' GROUP BY pickup_date",
            )
            .await?;

        insta::with_settings!({
            filters => [(r"elapsed_compute=\{[^}]*\}, ", "")],
        }, {
            insta::assert_snapshot!(scan_lines(&plan), @"
            t0: DataSourceExec: format=iceberg, planned_files=1, filters=pickup_date@0 <= 2024-01-10, metrics=[output_rows={0:25.00 K}, output_bytes={0:128.0 KB}, output_batches={0:4}, batches_split={0:0}, bytes_scanned={0:524.4 K}, files_opened={0:1}]
            t1: DataSourceExec: format=iceberg, planned_files=2, filters=pickup_date@0 <= 2024-01-10, metrics=[output_rows={1:50.00 K}, output_bytes={1:256.0 KB}, output_batches={1:8}, batches_split={1:0}, bytes_scanned={1:1.05 M}, files_opened={1:2}]
            ");
        });
        Ok(())
    }

    #[tokio::test]
    async fn lazy_planning_limit_counts_only_the_files_the_reader_opened() -> Result<()> {
        // All seven files are in one partition, and the reader reads one file at a time,
        // so the first file satisfies the limit and the other six are never opened.
        let (plan, _) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 1;
                Ok(state)
            })?
            .query_with_metrics("SELECT * FROM taxi LIMIT 1")
            .await?;

        insta::with_settings!({
            filters => [(r"elapsed_compute=[^,]*, ", "")],
        }, {
            insta::assert_snapshot!(scan_lines(&plan), @"DataSourceExec: format=iceberg, table=default.taxi, projection=[vendor_id, pickup_at, dropoff_at, passenger_count, trip_distance, pickup_location_id, dropoff_location_id, payment_type, fare_amount, tip_amount, tolls_amount, total_amount, pickup_date], fetch=1, metrics=[output_rows=1, output_bytes=704.0 KB, output_batches=1, batches_split=0, bytes_scanned=1.33 M, files_opened=1]");
        });
        Ok(())
    }

    #[tokio::test]
    async fn eager_planning_limit_counts_only_the_files_the_reader_opened() -> Result<()> {
        // All seven files are in one partition, and the reader reads one file at a time,
        // so the first file satisfies the limit and the other six are never opened.
        let (plan, _) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 1;
                Ok(state)
            })?
            .query_with_metrics("SELECT * FROM taxi LIMIT 1")
            .await?;

        insta::with_settings!({
            filters => [(r"elapsed_compute=[^,]*, ", "")],
        }, {
            insta::assert_snapshot!(scan_lines(&plan), @"DataSourceExec: format=iceberg, planned_files=7, fetch=1, metrics=[output_rows=1, output_bytes=704.0 KB, output_batches=1, batches_split=0, bytes_scanned=1.33 M, files_opened=1]");
        });
        Ok(())
    }

    /// The plan's Iceberg scans, one per line, without the plan's indentation.
    fn scan_lines(plan: &str) -> String {
        plan.lines()
            .filter(|line| line.contains("DataSourceExec: format=iceberg"))
            .map(|line| line.trim_start_matches([' ', '│']))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
