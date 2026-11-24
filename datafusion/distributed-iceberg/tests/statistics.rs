mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use datafusion_distributed_iceberg_test_tools::{IcebergTestHarness, without_snapshots};

    use datafusion_distributed_iceberg::IcebergExt;

    use datafusion_distributed_iceberg_test_tools::{
        FixtureCatalog, FixtureTable, create_taxi_table,
    };

    use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray, record_batch};
    // `record_batch!` refers to `arrow_schema` by name.
    use datafusion::arrow::datatypes as arrow_schema;
    use datafusion::common::stats::Precision;
    use datafusion::common::{ColumnStatistics, Statistics};
    use datafusion::error::Result;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::parquet::file::properties::{EnabledStatistics, WriterProperties};
    use datafusion::parquet::schema::types::ColumnPath;
    use datafusion::physical_plan::joins::HashJoinExec;
    use datafusion::physical_plan::statistics::{StatisticsArgs, StatisticsContext};
    use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion::scalar::ScalarValue;
    use datafusion_distributed::test_utils::in_memory_channel_resolver::{
        InMemoryChannelResolver, InMemoryWorkerResolver,
    };
    use datafusion_distributed::{
        DistributedConfig, DistributedExt, DistributedMetricsFormat, SessionStateBuilderExt,
        WorkerQueryContext, rewrite_distributed_plan_with_metrics,
    };
    use iceberg::puffin::APACHE_DATASKETCHES_THETA_V1;
    use iceberg::spec::{
        BlobMetadata, DataFile, MAIN_BRANCH, NestedField, Operation, PrimitiveType, Schema,
        Snapshot, StatisticsFile, Summary, TableMetadata, Transform, Type,
    };
    use iceberg::{MetadataLocation, TableIdent};
    use std::str::FromStr;

    // Values from the generated taxi snapshot summary.
    const TAXI_ROWS: usize = 175_000;
    const TAXI_BYTES: usize = 5_655_782;
    // The rows across both files of the metrics table.
    const METRICS_ROWS: usize = 9;

    #[tokio::test]
    async fn missing_snapshot_summary_statistics_are_absent() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_metadata(metadata_without_summary_statistics)
            .build()
            .await?;
        let stats = query_statistics(&harness, "SELECT * FROM taxi").await?;

        assert_eq!(stats.num_rows, Precision::Absent);
        assert_eq!(stats.total_byte_size, Precision::Absent);
        Ok(())
    }

    #[tokio::test]
    async fn planned_file_statistics_use_manifest_totals_when_snapshot_summary_is_missing()
    -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_metadata(metadata_without_summary_statistics)
            .build()
            .await?;
        let stats = query_statistics(&harness, "SELECT * FROM taxi").await?;
        assert_eq!(stats.num_rows, Precision::Exact(TAXI_ROWS));
        assert_eq!(stats.total_byte_size, Precision::Exact(TAXI_BYTES));

        let filtered = query_statistics(
            &harness,
            "SELECT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10'",
        )
        .await?;
        assert!(matches!(filtered.num_rows, Precision::Inexact(_)));
        assert!(matches!(filtered.total_byte_size, Precision::Inexact(_)));
        Ok(())
    }

    #[tokio::test]
    async fn eager_filtered_scan_skips_unused_column_statistics() -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_table_option("iceberg.snapshot_id", "1")
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;
        // The manifest's partition summary excludes this query. Delete the manifest
        // so a redundant whole-snapshot column-statistics read fails the test.
        let table = harness.table().table();
        let snapshot = table
            .metadata()
            .current_snapshot()
            .expect("taxi has a snapshot");
        for manifest in table.manifest_list_reader(snapshot).load().await?.entries() {
            table.file_io().delete(&manifest.manifest_path).await?;
        }
        let stats = query_statistics(
            &harness,
            "SELECT * FROM taxi WHERE pickup_date = DATE '1900-01-01'",
        )
        .await?;
        assert_eq!(stats.num_rows, Precision::Exact(0));
        Ok(())
    }

    #[tokio::test]
    async fn full_scan_without_column_stats() -> Result<(), Box<dyn Error>> {
        let (table, files) = create_metrics_table().await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_table_option("iceberg.snapshot_id", "2")
            .build()
            .await?;
        let stats = query_statistics(&harness, "SELECT * FROM metrics").await?;

        assert_eq!(stats.num_rows, Precision::Exact(METRICS_ROWS));
        assert_eq!(
            stats.total_byte_size,
            Precision::Exact(
                files
                    .iter()
                    .map(|file| file.file_size_in_bytes() as usize)
                    .sum()
            )
        );
        assert_eq!(
            stats.column_statistics,
            vec![ColumnStatistics::new_unknown(); 3]
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_scan_with_column_stats() -> Result<(), Box<dyn Error>> {
        let (table, files) = create_metrics_table().await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_table_option("iceberg.snapshot_id", "2")
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;
        let stats = query_statistics(&harness, "SELECT * FROM metrics").await?;

        assert_eq!(stats.num_rows, Precision::Exact(METRICS_ROWS));
        assert_eq!(
            stats.total_byte_size,
            Precision::Exact(
                files
                    .iter()
                    .map(|file| file.file_size_in_bytes() as usize)
                    .sum()
            )
        );
        assert_eq!(
            stats.column_statistics,
            expected_column_statistics(&files).to_vec()
        );
        Ok(())
    }

    #[tokio::test]
    async fn projection_without_column_stats() -> Result<(), Box<dyn Error>> {
        let (table, _) = create_metrics_table().await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_table_option("iceberg.snapshot_id", "2")
            .build()
            .await?;
        let stats = query_statistics(
            &harness,
            "SELECT passenger_count, vendor_id, trip_distance FROM metrics",
        )
        .await?;

        assert_eq!(stats.num_rows, Precision::Exact(METRICS_ROWS));
        assert_eq!(
            stats.column_statistics,
            vec![ColumnStatistics::new_unknown(); 3]
        );
        Ok(())
    }

    #[tokio::test]
    async fn projection_with_column_stats() -> Result<(), Box<dyn Error>> {
        let (table, files) = create_metrics_table().await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_table_option("iceberg.snapshot_id", "2")
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;
        let stats = query_statistics(
            &harness,
            "SELECT passenger_count, vendor_id, trip_distance FROM metrics",
        )
        .await?;

        assert_eq!(stats.num_rows, Precision::Exact(METRICS_ROWS));
        let [vendor_id, passenger_count, trip_distance] = expected_column_statistics(&files);
        assert_eq!(
            stats.column_statistics,
            vec![passenger_count, vendor_id, trip_distance]
        );
        Ok(())
    }

    #[tokio::test]
    async fn reports_statistics_for_the_selected_snapshot() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_table_option("iceberg.snapshot_id", "42")
            .with_metadata(historical_taxi_metadata)
            .build()
            .await?;
        let stats = query_statistics(&harness, "SELECT * FROM taxi").await?;

        // This historical summary omits the optional delete-file count.
        assert_eq!(stats.num_rows, Precision::Inexact(42));
        assert_eq!(stats.total_byte_size, Precision::Exact(4_242));
        Ok(())
    }

    #[tokio::test]
    async fn statistics_propagate_through_filter() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .build()
            .await?;
        let stats = query_statistics(
            &harness,
            "SELECT vendor_id FROM taxi WHERE pickup_date = DATE '2024-01-10'",
        )
        .await?;

        assert!(matches!(stats.num_rows, Precision::Inexact(_)));
        assert_eq!(stats.column_statistics.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn statistics_propagate_through_projection_and_sort() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .build()
            .await?;
        let stats = query_statistics(
            &harness,
            "SELECT vendor_id, trip_distance FROM taxi ORDER BY pickup_at",
        )
        .await?;

        assert_eq!(stats.num_rows, Precision::Exact(TAXI_ROWS));
        assert_eq!(stats.column_statistics.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn explain_shows_statistics_on_the_iceberg_source() -> Result<()> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .build()
            .await?;
        let plan = harness.physical_plan("SELECT vendor_id FROM taxi").await?;
        let display = displayable(plan.as_ref())
            .set_show_statistics(true)
            .indent(true)
            .to_string();
        insta::assert_snapshot!(display, @"
        CooperativeExec, statistics=[Rows=Exact(175000), Bytes=Exact(5655782), [(Col[0]:)]]
          DataSourceExec: format=iceberg, table=default.taxi, projection=[vendor_id], statistics=[Rows=Exact(175000), Bytes=Exact(5655782), [(Col[0]:)]]
        ");
        Ok(())
    }

    #[tokio::test]
    async fn count_star_skips_scan_without_column_stats() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .query("SELECT count(*) FROM taxi")
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[175000 as count(*)]
          PlaceholderRowExec
        ");
        insta::assert_snapshot!(batches, @"
        +----------+
        | count(*) |
        +----------+
        | 175000   |
        +----------+
        ");
        Ok(())
    }

    #[tokio::test]
    async fn count_star_skips_scan_with_column_stats() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.column_stats_enabled", true)
            .query("SELECT count(*) FROM taxi")
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[175000 as count(*)]
          PlaceholderRowExec
        ");
        insta::assert_snapshot!(batches, @"
        +----------+
        | count(*) |
        +----------+
        | 175000   |
        +----------+
        ");
        Ok(())
    }

    #[tokio::test]
    async fn count_star_applies_position_deletes() -> Result<(), Box<dyn Error>> {
        let (mut table, files) = create_metrics_table().await?;
        // Nine data rows, with two positions removed from the first file.
        table
            .append()
            .position_deletes(&files[0], [0, 2])
            .commit()
            .await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_config_option("iceberg.eager_file_planning", false)
            .build()
            .await?;
        let (_, batches) = harness.query("SELECT count(*) FROM metrics").await?;

        insta::assert_snapshot!(batches, @"
        +----------+
        | count(*) |
        +----------+
        | 7        |
        +----------+
        ");
        Ok(())
    }

    // Observe the query's output statistics, including projection and propagation.
    async fn query_statistics(harness: &IcebergTestHarness, sql: &str) -> Result<Arc<Statistics>> {
        let plan = harness.physical_plan(sql).await?;
        StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new())
    }

    fn metadata_without_summary_statistics(metadata: &TableMetadata) -> TableMetadata {
        let current = metadata.current_snapshot().expect("taxi has a snapshot");
        let snapshot = Snapshot::builder()
            .with_snapshot_id(current.snapshot_id())
            .with_parent_snapshot_id(current.parent_snapshot_id())
            .with_sequence_number(current.sequence_number())
            .with_timestamp_ms(current.timestamp_ms())
            .with_manifest_list(current.manifest_list())
            .schema_id_opt(current.schema_id())
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: Default::default(),
            })
            .build();
        without_snapshots(metadata)
            .set_branch_snapshot(snapshot, MAIN_BRANCH)
            .expect("taxi snapshot can be added")
            .build()
            .expect("taxi metadata is valid")
            .metadata
    }

    fn historical_taxi_metadata(metadata: &TableMetadata) -> TableMetadata {
        let current = metadata.current_snapshot().expect("taxi has a snapshot");
        // The summary totals are synthetic; the manifest still describes the full taxi data.
        // This fixture is for statistics planning, not executing a 42-row scan.
        let historical = Snapshot::builder()
            .with_snapshot_id(42)
            .with_sequence_number(current.sequence_number() - 1)
            .with_timestamp_ms(current.timestamp_ms() - 1)
            .with_manifest_list(current.manifest_list())
            .schema_id_opt(current.schema_id())
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: [
                    ("total-records".to_string(), "42".to_string()),
                    ("total-files-size".to_string(), "4242".to_string()),
                ]
                .into_iter()
                .collect(),
            })
            .build();
        without_snapshots(metadata)
            .add_snapshot(historical)
            .expect("historical snapshot can be added")
            .set_branch_snapshot(current.as_ref().clone(), MAIN_BRANCH)
            .expect("taxi snapshot can be added")
            .build()
            .expect("taxi metadata is valid")
            .metadata
    }

    // A table of two files on the current snapshot, committed separately. The second is
    // written without Parquet statistics for passenger_count, so its manifest entry has a
    // size for that column but no null count or bounds.
    async fn create_metrics_table() -> Result<(FixtureTable, Vec<DataFile>), Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "vendor_id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "passenger_count", Type::Primitive(PrimitiveType::Long))
                    .into(),
                NestedField::optional(3, "trip_distance", Type::Primitive(PrimitiveType::Double))
                    .into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "metrics"])?, schema)
            .create()
            .await?;
        let mut files = table
            .append()
            .file(record_batch!(
                ("vendor_id", Int32, [Some(2), Some(9), None, None]),
                ("passenger_count", Int64, [Some(10), Some(20), None, None]),
                ("trip_distance", Float64, [1.5, 2.5, 0.5, 3.0])
            )?)
            .commit()
            .await?;
        let without_passenger_count_statistics = WriterProperties::builder()
            .set_column_statistics_enabled(
                ColumnPath::from("passenger_count"),
                EnabledStatistics::None,
            )
            .build();
        files.extend(
            table
                .append()
                .writer_properties(without_passenger_count_statistics)
                .file(record_batch!(
                    ("vendor_id", Int32, [Some(1), Some(5), None, None, None]),
                    ("passenger_count", Int64, [30, 40, 35, 33, 31]),
                    ("trip_distance", Float64, [4.0, 0.25, 1.0, 2.0, 3.0])
                )?)
                .commit()
                .await?,
        );
        Ok((table, files))
    }

    #[tokio::test]
    async fn pruned_scan_omits_snapshot_ndv_and_non_singleton_equality_bounds()
    -> Result<(), Box<dyn Error>> {
        let (table, skewed_file_bytes) = create_skewed_tenant_events_table(20_000).await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_metadata(|meta| with_puffin_ndv(meta, 1, 5_000))
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;

        let stats = query_statistics(
            &harness,
            "SELECT tenant_id, item_key FROM events WHERE tenant_id = 1",
        )
        .await?;

        // File pruning keeps only the file containing tenant_id = 1 (20,000 rows for tenant 1 + 2
        // rows for tenant 2_000_000). Snapshot NDV (5,000) and non-singleton bounds [1, 2_000_000]
        // must not deflate this down to ~4 rows (1/NDV) or 1 row (1/(max-min+1)).
        assert_eq!(stats.num_rows, Precision::Inexact(20_002));
        assert_eq!(
            stats.column_statistics[0].distinct_count,
            Precision::Inexact(1)
        );
        assert!(
            stats.column_statistics[1]
                .byte_size
                .get_value()
                .copied()
                .unwrap_or(0)
                > 0,
            "projected column byte_size from the selected file should still be reported"
        );
        assert_eq!(stats.total_byte_size, Precision::Inexact(skewed_file_bytes));
        Ok(())
    }

    #[tokio::test]
    async fn pruned_scan_keeps_singleton_equality_bounds() -> Result<(), Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "tenant_id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "item_key", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "single_tenant"])?, schema)
            .partition_by("tenant_id", Transform::Identity)
            .create()
            .await?;
        table
            .append()
            .file(record_batch!(
                ("tenant_id", Int64, [1, 1, 1, 1]),
                ("item_key", Utf8, ["k1", "k2", "k3", "k4"])
            )?)
            .file(record_batch!(
                ("tenant_id", Int64, [99, 99, 99]),
                ("item_key", Utf8, ["a", "b", "c"])
            )?)
            .commit()
            .await?;

        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_metadata(|meta| with_puffin_ndv(meta, 1, 5_000))
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;

        let stats = query_statistics(
            &harness,
            "SELECT tenant_id, item_key FROM single_tenant WHERE tenant_id = 1",
        )
        .await?;

        assert_eq!(stats.num_rows, Precision::Inexact(4));
        assert_eq!(
            stats.column_statistics[0].min_value,
            Precision::Exact(ScalarValue::Int64(Some(1)))
        );
        assert_eq!(
            stats.column_statistics[0].max_value,
            Precision::Exact(ScalarValue::Int64(Some(1)))
        );
        Ok(())
    }

    #[tokio::test]
    async fn pruned_scan_keeps_range_filter_bounds() -> Result<(), Box<dyn Error>> {
        let (table, _) = create_skewed_tenant_events_table(1_000).await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;

        let stats = query_statistics(
            &harness,
            "SELECT tenant_id FROM events WHERE tenant_id >= 3000100 AND tenant_id <= 3000200",
        )
        .await?;

        // Range predicates must retain min/max bounds so FilterExec can still tighten the range.
        assert_eq!(
            stats.column_statistics[0].min_value,
            Precision::Inexact(ScalarValue::Int64(Some(3_000_100)))
        );
        assert_eq!(
            stats.column_statistics[0].max_value,
            Precision::Inexact(ScalarValue::Int64(Some(3_000_200)))
        );
        Ok(())
    }

    #[tokio::test]
    async fn lazy_equality_scan_without_ndv_omits_non_singleton_sparse_bounds()
    -> Result<(), Box<dyn Error>> {
        let (table, _) = create_skewed_tenant_events_table(10_000).await?;
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(table)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.column_stats_enabled", true)
            .build()
            .await?;

        let stats = query_statistics(
            &harness,
            "SELECT tenant_id, item_key FROM events WHERE tenant_id = 1",
        )
        .await?;

        // Without Puffin NDV, FilterExec must not divide by (2_000_000 - 1 + 1) and collapse to 1 row.
        assert!(
            stats.num_rows.get_value().copied().unwrap_or(0) >= 10_000,
            "expected >= 10_000 estimated rows, got {:?}",
            stats.num_rows
        );
        Ok(())
    }

    #[tokio::test]
    async fn skewed_tenant_join_keeps_large_table_on_probe_side() -> Result<(), Box<dyn Error>> {
        let (events, _) = create_skewed_tenant_events_table(80_000).await?;
        let items = create_items_table(500).await?;

        // Write events metadata with a snapshot-wide Puffin NDV sketch on tenant_id (5,000 tenants).
        let events_meta = with_puffin_ndv(events.table().metadata(), 1, 5_000);
        let events_loc = MetadataLocation::from_str(events.metadata_location()?)
            .map_err(|e| format!("{e}"))?
            .with_next_version();
        events_meta
            .write_to(events.table().file_io(), &events_loc)
            .await?;
        let items_loc = items.metadata_location()?.to_string();

        let ext = Arc::new(|state: SessionStateBuilder| {
            state.with_iceberg_integration(Default::default())
        });
        let ext_worker = Arc::clone(&ext);
        let resolver =
            InMemoryChannelResolver::from_session_builder(move |ctx: WorkerQueryContext| {
                let state = ext_worker(ctx.builder).build();
                async move { Ok(state) }
            });
        let mut state_builder = ext(SessionStateBuilder::new()
            .with_default_features()
            .with_config(SessionConfig::new().with_target_partitions(4))
            .with_distributed_option_extension(DistributedConfig::default()));
        let opts = state_builder.config().get_or_insert_default().options_mut();
        opts.set("iceberg.eager_file_planning", "true")?;
        opts.set("iceberg.column_stats_enabled", "true")?;
        opts.set("datafusion.optimizer.join_reordering", "true")?;
        let state = state_builder
            .with_distributed_planner()
            .with_distributed_worker_resolver(InMemoryWorkerResolver::new(2))
            .with_distributed_channel_resolver(resolver)
            .build();
        let ctx = SessionContext::new_with_state(state);

        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE events STORED AS ICEBERG LOCATION '{events_loc}'"
        ))
        .await?
        .collect()
        .await?;
        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE items STORED AS ICEBERG LOCATION '{items_loc}'"
        ))
        .await?
        .collect()
        .await?;

        let sql = "SELECT count(*) AS cnt \
                   FROM items i \
                   JOIN events e ON i.tenant_id = e.tenant_id AND i.item_key = e.item_key \
                   WHERE i.tenant_id = 1 AND e.tenant_id = 1";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        let cnt = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count(*) is Int64")
            .value(0);
        assert_eq!(cnt, 500);

        let metrics_plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;
        let mut build_rows = Vec::new();
        collect_hash_join_build_rows(metrics_plan.as_ref(), &mut build_rows);
        assert!(
            !build_rows.is_empty(),
            "expected at least one HashJoinExec in plan"
        );
        // The small `items` table (500 rows) must be the build side, never the 80,000-row
        // skewed `events` table.
        for rows in &build_rows {
            assert!(
                *rows <= 500,
                "HashJoinExec built on the large skewed events side ({rows} build rows) instead of items (500 rows)"
            );
        }
        Ok(())
    }

    /// Reproducible benchmark for the skewed multi-tenant join scenario.
    ///
    /// Run with:
    /// `cargo test -p datafusion-distributed-iceberg --release --test statistics skewed_tenant_join_benchmark -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "manual benchmark for before/after comparison"]
    async fn skewed_tenant_join_benchmark() -> Result<(), Box<dyn Error>> {
        let (events, _) = create_skewed_tenant_events_table(300_000).await?;
        let items = create_items_table(2_000).await?;

        let events_meta = with_puffin_ndv(events.table().metadata(), 1, 5_000);
        let events_loc = MetadataLocation::from_str(events.metadata_location()?)
            .map_err(|e| format!("{e}"))?
            .with_next_version();
        events_meta
            .write_to(events.table().file_io(), &events_loc)
            .await?;
        let items_loc = items.metadata_location()?.to_string();

        let ext = Arc::new(|state: SessionStateBuilder| {
            state.with_iceberg_integration(Default::default())
        });
        let ext_worker = Arc::clone(&ext);
        let resolver =
            InMemoryChannelResolver::from_session_builder(move |ctx: WorkerQueryContext| {
                let state = ext_worker(ctx.builder).build();
                async move { Ok(state) }
            });
        let mut state_builder = ext(SessionStateBuilder::new()
            .with_default_features()
            .with_config(SessionConfig::new().with_target_partitions(4))
            .with_distributed_option_extension(DistributedConfig::default()));
        let opts = state_builder.config().get_or_insert_default().options_mut();
        opts.set("iceberg.eager_file_planning", "true")?;
        opts.set("iceberg.column_stats_enabled", "true")?;
        opts.set("datafusion.optimizer.join_reordering", "true")?;
        let state = state_builder
            .with_distributed_planner()
            .with_distributed_worker_resolver(InMemoryWorkerResolver::new(2))
            .with_distributed_channel_resolver(resolver)
            .build();
        let ctx = SessionContext::new_with_state(state);

        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE events STORED AS ICEBERG LOCATION '{events_loc}'"
        ))
        .await?
        .collect()
        .await?;
        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE items STORED AS ICEBERG LOCATION '{items_loc}'"
        ))
        .await?
        .collect()
        .await?;

        let events_filter_plan = ctx
            .sql("SELECT tenant_id, item_key, payload FROM events WHERE tenant_id = 1")
            .await?
            .create_physical_plan()
            .await?;
        let events_stats = StatisticsContext::new()
            .compute(events_filter_plan.as_ref(), &StatisticsArgs::new())?;

        let sql = "SELECT count(*) AS cnt \
                   FROM items i \
                   JOIN events e ON i.tenant_id = e.tenant_id AND i.item_key = e.item_key \
                   WHERE i.tenant_id = 1 AND e.tenant_id = 1";

        // Warmup
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let _ = collect(plan, ctx.task_ctx()).await?;

        let mut durations_ms = Vec::with_capacity(5);
        let mut last_metrics_plan = None;
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
            durations_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
            last_metrics_plan = Some(
                rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                    .await?,
            );
        }
        durations_ms.sort_by(|a, b| a.total_cmp(b));

        let mut join_summaries = Vec::new();
        if let Some(metrics_plan) = last_metrics_plan {
            collect_hash_join_summaries(metrics_plan.as_ref(), &mut join_summaries);
        }
        println!(
            "events_filter_stats: rows={:?}, bytes={:?}",
            events_stats.num_rows, events_stats.total_byte_size
        );
        for summary in &join_summaries {
            println!("hash_join: {summary}");
        }
        println!(
            "latency_ms: min={:.2}, median={:.2}, max={:.2}",
            durations_ms[0],
            durations_ms[durations_ms.len() / 2],
            durations_ms[durations_ms.len() - 1]
        );
        Ok(())
    }

    fn collect_hash_join_summaries(plan: &dyn ExecutionPlan, out: &mut Vec<String>) {
        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            let metrics = join.metrics();
            let build_rows = metrics
                .as_ref()
                .and_then(|m| m.sum_by_name("build_input_rows"))
                .map(|v| v.as_usize())
                .unwrap_or(0);
            let build_mem = metrics
                .as_ref()
                .and_then(|m| m.sum_by_name("build_mem_used"))
                .map(|v| v.as_usize())
                .unwrap_or(0);
            let probe_rows = metrics
                .as_ref()
                .and_then(|m| m.sum_by_name("input_rows"))
                .map(|v| v.as_usize())
                .unwrap_or(0);
            out.push(format!(
                "mode={:?}, build_input_rows={}, probe_input_rows={}, build_mem_used={} bytes",
                join.partition_mode(),
                build_rows,
                probe_rows,
                build_mem
            ));
        }
        for child in plan.children() {
            collect_hash_join_summaries(child.as_ref(), out);
        }
    }

    fn collect_hash_join_build_rows(plan: &dyn ExecutionPlan, out: &mut Vec<usize>) {
        if let Some(join) = plan.downcast_ref::<HashJoinExec>()
            && let Some(metrics) = join.metrics()
            && let Some(count) = metrics.sum_by_name("build_input_rows")
        {
            out.push(count.as_usize());
        }
        for child in plan.children() {
            collect_hash_join_build_rows(child.as_ref(), out);
        }
    }

    fn with_puffin_ndv(metadata: &TableMetadata, field_id: i32, ndv: usize) -> TableMetadata {
        let snapshot = metadata
            .current_snapshot()
            .expect("table has a current snapshot");
        let stats_file = StatisticsFile {
            snapshot_id: snapshot.snapshot_id(),
            statistics_path: format!("{}/metadata/stats.puffin", metadata.location()),
            file_size_in_bytes: 128,
            file_footer_size_in_bytes: 32,
            key_metadata: None,
            blob_metadata: vec![BlobMetadata {
                r#type: APACHE_DATASKETCHES_THETA_V1.to_string(),
                snapshot_id: snapshot.snapshot_id(),
                sequence_number: snapshot.sequence_number(),
                fields: vec![field_id],
                properties: [("ndv".to_string(), ndv.to_string())].into_iter().collect(),
            }],
        };
        metadata
            .clone()
            .into_builder(None)
            .set_statistics(stats_file)
            .build()
            .expect("statistics metadata is valid")
            .metadata
    }

    // Creates a multi-tenant table with two data files:
    // - File 1: dominated by tenant_id = 1 (`heavy_tenant_rows` rows) alongside two rows from
    //   tenant_id = 2_000_000 so the file has wide non-singleton bounds [1, 2_000_000].
    // - File 2: other tenants in [3_000_000, 3_000_999], which eager file planning prunes for
    //   `tenant_id = 1`.
    async fn create_skewed_tenant_events_table(
        heavy_tenant_rows: usize,
    ) -> Result<(FixtureTable, usize), Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "tenant_id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "item_key", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "payload", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "events"])?, schema)
            .create()
            .await?;

        let arrow_schema = table.schema()?;

        let mut tenant_ids = Vec::with_capacity(heavy_tenant_rows + 2);
        let mut keys = Vec::with_capacity(heavy_tenant_rows + 2);
        let mut payloads = Vec::with_capacity(heavy_tenant_rows + 2);
        for i in 0..heavy_tenant_rows {
            tenant_ids.push(1_i64);
            keys.push(format!("namespace/category/item-key-{i:08}"));
            payloads.push(format!(
                "payload-record-{i:016x}-extra-padding-for-realistic-string-column-width"
            ));
        }
        for i in 0..2 {
            tenant_ids.push(2_000_000_i64);
            keys.push(format!("namespace/category/other-tenant-item-{i}"));
            payloads.push("small-payload".to_string());
        }
        let skewed_batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(tenant_ids)),
                Arc::new(StringArray::from(keys)),
                Arc::new(StringArray::from(payloads)),
            ],
        )?;
        let skewed_files = table.append().file(skewed_batch).commit().await?;
        let skewed_bytes: usize = skewed_files
            .iter()
            .map(|f| f.file_size_in_bytes() as usize)
            .sum();

        // Add a second file in a disjoint tenant_id range so eager file planning prunes it.
        let mut other_tenants = Vec::with_capacity(1_000);
        let mut other_keys = Vec::with_capacity(1_000);
        let mut other_payloads = Vec::with_capacity(1_000);
        for tenant in 3_000_000_i64..3_001_000_i64 {
            other_tenants.push(tenant);
            other_keys.push(format!("namespace/category/tenant-{tenant:012}-item"));
            other_payloads.push("other-payload".to_string());
        }
        let other_batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(other_tenants)),
                Arc::new(StringArray::from(other_keys)),
                Arc::new(StringArray::from(other_payloads)),
            ],
        )?;
        table.append().file(other_batch).commit().await?;
        Ok((table, skewed_bytes))
    }

    async fn create_items_table(rows: usize) -> Result<FixtureTable, Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "tenant_id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "item_key", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "items"])?, schema)
            .create()
            .await?;
        let arrow_schema = table.schema()?;
        let tenant_ids: Vec<i64> = vec![1; rows];
        let keys: Vec<String> = (0..rows)
            .map(|i| format!("namespace/category/item-key-{i:08}"))
            .collect();
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(tenant_ids)),
                Arc::new(StringArray::from(keys)),
            ],
        )?;
        table.append().file(batch).commit().await?;
        Ok(table)
    }

    // The column statistics of the metrics table, in schema order, with column stats enabled.
    fn expected_column_statistics(files: &[DataFile]) -> [ColumnStatistics; 3] {
        let size = |field_id| {
            let size = files
                .iter()
                .map(|file| file.column_sizes()[&field_id])
                .sum::<u64>();
            Precision::Inexact(size as usize)
        };
        [
            ColumnStatistics {
                null_count: Precision::Exact(5),
                min_value: Precision::Inexact(ScalarValue::Int32(Some(1))),
                max_value: Precision::Inexact(ScalarValue::Int32(Some(9))),
                byte_size: size(1),
                ..ColumnStatistics::new_unknown()
            },
            // The second file has no null count or bounds for passenger_count, so the totals
            // must stay unknown.
            ColumnStatistics {
                byte_size: size(2),
                ..ColumnStatistics::new_unknown()
            },
            ColumnStatistics {
                null_count: Precision::Exact(0),
                min_value: Precision::Inexact(ScalarValue::Float64(Some(0.25))),
                max_value: Precision::Inexact(ScalarValue::Float64(Some(4.0))),
                byte_size: size(3),
                ..ColumnStatistics::new_unknown()
            },
        ]
    }
}
