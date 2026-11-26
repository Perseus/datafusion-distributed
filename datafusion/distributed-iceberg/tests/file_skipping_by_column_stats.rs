mod tests {
    use datafusion::arrow::array::{
        ArrayRef, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
        TimestampMicrosecondArray,
    };
    use datafusion::arrow::datatypes::SchemaRef;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::physical_plan::metrics::MetricValue;
    use datafusion::physical_plan::{ExecutionPlan, collect};
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

    // `events` has 4 files with ids 0-9, 10-19, 20-29 and 30-39. The join's keys fall in the
    // second and third files, so the scan can skip the other two by their id bounds.
    const IDS_IN_TWO_FILES: &str = "SELECT COUNT(*) AS events FROM events \
        JOIN (VALUES (CAST(15 AS BIGINT)), (CAST(25 AS BIGINT))) AS keys(id) \
        ON events.id = keys.id";

    #[tokio::test]
    async fn lazy_planning_workers_skip_files_outside_the_join_keys() -> Result<(), Box<dyn Error>>
    {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(events().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1)
            })?
            .build()
            .await?;

        let plan = harness.physical_plan(IDS_IN_TWO_FILES).await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +--------+
        | events |
        +--------+
        | 2      |
        +--------+
        ");
        // The first `DataSourceExec` is the join's `VALUES` table. The second is the events scan.
        assert_eq!(node_metrics::<DataSourceExec>(&plan, "files_opened", 1), 2);
        Ok(())
    }

    #[tokio::test]
    async fn eager_planning_workers_skip_files_outside_the_join_keys() -> Result<(), Box<dyn Error>>
    {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(events().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1)
            })?
            .build()
            .await?;

        let plan = harness.physical_plan(IDS_IN_TWO_FILES).await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +--------+
        | events |
        +--------+
        | 2      |
        +--------+
        ");
        assert_eq!(
            node_metrics::<DistributedLeafExec>(&plan, "files_opened", 0),
            2
        );
        Ok(())
    }

    #[tokio::test]
    async fn top_n_skips_files_once_it_fills_up() -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(events().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
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
            .physical_plan("SELECT id FROM events ORDER BY id LIMIT 3")
            .await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +----+
        | id |
        +----+
        | 0  |
        | 1  |
        | 2  |
        +----+
        ");
        // The first file fills the top 3. The filter then becomes `id < 2`, and the other files'
        // lower bounds are all above that.
        assert_eq!(node_metrics::<DataSourceExec>(&plan, "files_opened", 0), 1);
        Ok(())
    }

    #[tokio::test]
    async fn min_skips_files_once_it_has_a_value() -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(events().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            // Read one file at a time, so min()'s filter tightens before the next file starts.
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            // With one partition, DataFusion runs the aggregate in a single step, which has no
            // dynamic filter. Two partitions give it a partial step with one.
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 2;
                Ok(state)
            })?
            .build()
            .await?;

        let plan = harness
            .physical_plan("SELECT MIN(id) AS first FROM events")
            .await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +-------+
        | first |
        +-------+
        | 0     |
        +-------+
        ");
        // The files go to the two partitions in turn. Partition 0 reads ids 0-9 and then 20-29,
        // and partition 1 reads 10-19 and then 30-39. Each partition's first file brings the
        // filter to at most `id < 10`, which rules out its second file.
        let files = scan_metric(&plan, "files_pruned_by_statistics").expect("a dynamic filter");
        insta::assert_snapshot!(files, @"4 total → 2 matched");
        assert_eq!(node_metrics::<DataSourceExec>(&plan, "files_opened", 0), 2);
        Ok(())
    }

    #[tokio::test]
    async fn top_n_reads_files_with_nan() -> Result<(), Box<dyn Error>> {
        let harness = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(scores().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
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
            .physical_plan("SELECT score FROM scores ORDER BY score DESC LIMIT 1")
            .await?;
        let batches = collect(Arc::clone(&plan), harness.state().task_ctx()).await?;
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        // DataFusion sorts NaN above every number.
        insta::assert_snapshot!(pretty_format_batches(&batches)?, @"
        +-------+
        | score |
        +-------+
        | NaN   |
        +-------+
        ");
        // The first file sets the filter to `score > 3.0`. The second file's upper bound is 0.2,
        // so the scan skips it. The third file's upper bound is 0.5, but Iceberg bounds leave
        // out NaN and that file has one, so the scan reads it.
        let files = scan_metric(&plan, "files_pruned_by_statistics").expect("a dynamic filter");
        insta::assert_snapshot!(files, @"3 total → 2 matched");
        Ok(())
    }

    #[tokio::test]
    async fn validate_predicate_translation() -> Result<(), Box<dyn Error>> {
        let skipping = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(readings().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.dynamic_filtering_enabled", true)
            // Read one file at a time, so top-N's filter tightens before the next file starts.
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            // A join with more than 2 keys filters only on their bounds, so the cases cover
            // both kinds of join filter.
            .with_config_option(
                "datafusion.optimizer.hash_join_inlist_pushdown_max_distinct_values",
                2,
            )
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 1;
                Ok(state)
            })?
            .build()
            .await?;
        let reading_every_file = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(readings().await?)
            .with_config_option("iceberg.eager_file_planning", false)
            .with_config_option("iceberg.dynamic_filtering_enabled", false)
            .with_config_option("iceberg.data_file_concurrency_limit", 1)
            .with_config_option(
                "datafusion.optimizer.hash_join_inlist_pushdown_max_distinct_values",
                2,
            )
            .configure_session(|mut state| {
                let session = state.config().get_or_insert_default();
                session.options_mut().execution.target_partitions = 1;
                Ok(state)
            })?
            .build()
            .await?;

        // Keys at file bounds, keys in no file, nulls, zeros and NaN, for each type.
        let schema = skipping.table().schema()?;
        let cases = [
            join(&schema, "id", &["3"]),
            join(&schema, "id", &["0"]),
            join(&schema, "id", &["2", "3", "4"]),
            join(&schema, "id", &["NULL"]),
            join(&schema, "id", &["1", "NULL"]),
            join(&schema, "name", &["'avocado'"]),
            join(&schema, "name", &["'b'"]),
            join(&schema, "name", &["'éclair'"]),
            join(&schema, "day", &["'2024-02-01'"]),
            join(&schema, "day", &["'2024-01-15'"]),
            join(&schema, "n", &["12"]),
            join(&schema, "n", &["-1", "11", "13"]),
            join(&schema, "at", &["'2024-01-02T00:00:00'"]),
            join(&schema, "at", &["'2024-01-01T23:59:59.999999'"]),
            join(&schema, "x", &["1.5"]),
            join(&schema, "x", &["0.0"]),
            join(&schema, "x", &["-0.0"]),
            join(&schema, "x", &["'NaN'"]),
            join(&schema, "x", &["-1.0", "0.0", "2.5"]),
            top_n("id", 1),
            top_n("id DESC", 1),
            top_n("id NULLS FIRST", 2),
            top_n("id DESC NULLS LAST", 2),
            top_n("name", 1),
            top_n("name DESC", 1),
            top_n("day", 2),
            top_n("n", 1),
            top_n("at DESC", 2),
            top_n("x", 2),
            top_n("x DESC", 2),
        ];

        // Each case must return the same rows whether the scan skips files or reads them all.
        // The snapshot shows how many files each case kept.
        let mut skipped = Vec::new();
        for (case, sql) in cases {
            let plan = skipping.physical_plan(&sql).await?;
            let rows = collect(Arc::clone(&plan), skipping.state().task_ctx()).await?;
            let plan =
                rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                    .await?;
            let expected = reading_every_file.query(&sql).await?.1;
            assert_eq!(
                pretty_format_batches(&rows)?.to_string(),
                expected,
                "{case}"
            );
            let files = scan_metric(&plan, "files_pruned_by_statistics")
                .unwrap_or_else(|| panic!("{case}: the scan has no dynamic filter"));
            let rows: usize = rows.iter().map(RecordBatch::num_rows).sum();
            skipped.push(format!("{case}: rows: {rows}, files: {files}"));
        }
        insta::assert_snapshot!(skipped.join("\n"), @"
        join on id with 3: rows: 1, files: 6 total → 1 matched
        join on id with 0: rows: 0, files: 6 total → 0 matched
        join on id with 2, 3, 4: rows: 3, files: 6 total → 3 matched
        join on id with NULL: rows: 0, files: 0 total → 0 matched
        join on id with 1, NULL: rows: 1, files: 6 total → 1 matched
        join on name with 'avocado': rows: 1, files: 6 total → 1 matched
        join on name with 'b': rows: 0, files: 6 total → 0 matched
        join on name with 'éclair': rows: 1, files: 6 total → 1 matched
        join on day with '2024-02-01': rows: 1, files: 6 total → 1 matched
        join on day with '2024-01-15': rows: 0, files: 6 total → 0 matched
        join on n with 12: rows: 1, files: 6 total → 1 matched
        join on n with -1, 11, 13: rows: 3, files: 6 total → 3 matched
        join on at with '2024-01-02T00:00:00': rows: 1, files: 6 total → 1 matched
        join on at with '2024-01-01T23:59:59.999999': rows: 1, files: 6 total → 1 matched
        join on x with 1.5: rows: 1, files: 6 total → 2 matched
        join on x with 0.0: rows: 2, files: 6 total → 3 matched
        join on x with -0.0: rows: 2, files: 6 total → 3 matched
        join on x with 'NaN': rows: 1, files: 6 total → 1 matched
        join on x with -1.0, 0.0, 2.5: rows: 3, files: 6 total → 5 matched
        ORDER BY id LIMIT 1: rows: 1, files: 6 total → 1 matched
        ORDER BY id DESC LIMIT 1: rows: 1, files: 6 total → 6 matched
        ORDER BY id NULLS FIRST LIMIT 2: rows: 2, files: 6 total → 3 matched
        ORDER BY id DESC NULLS LAST LIMIT 2: rows: 2, files: 6 total → 5 matched
        ORDER BY name LIMIT 1: rows: 1, files: 6 total → 1 matched
        ORDER BY name DESC LIMIT 1: rows: 1, files: 6 total → 5 matched
        ORDER BY day LIMIT 2: rows: 2, files: 6 total → 3 matched
        ORDER BY n LIMIT 1: rows: 1, files: 6 total → 2 matched
        ORDER BY at DESC LIMIT 2: rows: 2, files: 6 total → 5 matched
        ORDER BY x LIMIT 2: rows: 2, files: 6 total → 4 matched
        ORDER BY x DESC LIMIT 2: rows: 2, files: 6 total → 5 matched
        ");
        Ok(())
    }

    // Table `events` with one `id` column, in 4 files with ids 0-9, 10-19, 20-29 and 30-39.
    async fn events() -> Result<FixtureTable, Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "events"])?, schema)
            .create()
            .await?;
        let mut append = table.append();
        for first in [0, 10, 20, 30] {
            let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(first..first + 10));
            append = append.file(RecordBatch::try_from_iter([("id", ids)])?);
        }
        append.commit().await?;
        Ok(table)
    }

    // Table `scores` with one `score` column, in 3 files read in this order: 1.0 to 3.0, then
    // 0.1 and 0.2, then 0.5 and NaN.
    async fn scores() -> Result<FixtureTable, Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "score", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "scores"])?, schema)
            .create()
            .await?;
        let mut append = table.append();
        for scores in [vec![1.0, 2.0, 3.0], vec![0.1, 0.2], vec![0.5, f64::NAN]] {
            let scores: ArrayRef = Arc::new(Float64Array::from(scores));
            append = append.file(RecordBatch::try_from_iter([("score", scores)])?);
        }
        append.commit().await?;
        Ok(table)
    }

    // Table `readings`, unpartitioned. Each row is its own file, and the last row is all nulls.
    async fn readings() -> Result<FixtureTable, Box<dyn Error>> {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "day", Type::Primitive(PrimitiveType::Date)).into(),
                NestedField::optional(4, "n", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(5, "at", Type::Primitive(PrimitiveType::Timestamp)).into(),
                NestedField::optional(6, "x", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()?;
        let fixtures = FixtureCatalog::try_new().await?;
        let mut table = fixtures
            .table(TableIdent::from_strs(["default", "readings"])?, schema)
            .create()
            .await?;
        let id: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(6),
            None,
        ]));
        let name: ArrayRef = Arc::new(StringArray::from(vec![
            Some("apple"),
            Some("avocado"),
            Some("banana"),
            Some("éclair"),
            Some("cherry"),
            None,
        ]));
        // Days since 1970-01-01: 2024-01-31, 2024-02-01, 2024-02-29, 2024-03-01, 2023-12-31.
        let day: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(19753),
            Some(19754),
            Some(19782),
            Some(19783),
            Some(19722),
            None,
        ]));
        let n: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(10),
            Some(11),
            Some(12),
            Some(13),
            Some(-1),
            None,
        ]));
        // Microseconds since 1970-01-01: 2024-01-01 23:59:59.999999, 2024-01-02 00:00:00,
        // 2024-01-02 12:00:00, 2024-01-03 00:00:00, 2023-12-31 23:59:59.
        let at: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1704153599999999),
            Some(1704153600000000),
            Some(1704196800000000),
            Some(1704240000000000),
            Some(1704067199000000),
            None,
        ]));
        let x: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.5),
            Some(-0.0),
            Some(0.0),
            Some(f64::NAN),
            Some(2.5),
            None,
        ]));
        let rows = RecordBatch::try_from_iter([
            ("id", id),
            ("name", name),
            ("day", day),
            ("n", n),
            ("at", at),
            ("x", x),
        ])?;
        let mut append = table.append();
        for row in 0..rows.num_rows() {
            append = append.file(rows.slice(row, 1));
        }
        append.commit().await?;
        Ok(table)
    }

    /// A case that joins `readings` with `keys` on `column`. Each key is cast to the column's
    /// type, so the join filters the column itself, without a cast around it.
    fn join(schema: &SchemaRef, column: &str, keys: &[&str]) -> (String, String) {
        let data_type = schema.field_with_name(column).unwrap().data_type();
        let values = keys
            .iter()
            .map(|key| format!("(arrow_cast({key}, '{data_type}'))"))
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!("join on {column} with {}", keys.join(", ")),
            format!(
                "SELECT t.* FROM readings t JOIN (VALUES {values}) AS k(x) \
                 ON t.{column} = k.x ORDER BY t.id"
            ),
        )
    }

    /// A case that reads the first `limit` rows of `readings` in `order`.
    fn top_n(order: &str, limit: usize) -> (String, String) {
        (
            format!("ORDER BY {order} LIMIT {limit}"),
            format!("SELECT * FROM readings ORDER BY {order} LIMIT {limit}"),
        )
    }

    /// The metric `name` of the Iceberg scan in `plan`, summed over its partitions.
    fn scan_metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<MetricValue> {
        plan.metrics()
            .and_then(|metrics| metrics.sum_by_name(name))
            .or_else(|| {
                plan.children()
                    .into_iter()
                    .find_map(|child| scan_metric(child, name))
            })
    }
}
