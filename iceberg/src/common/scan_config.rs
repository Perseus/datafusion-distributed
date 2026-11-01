use crate::common::dynamic_domain::CompactedDynamicFilters;
use crate::common::partition_pruning::PartitionFilePruner;
use crate::{IcebergConfig, common::df_err};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, Statistics};
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{Partitioning, PhysicalExpr};
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, TryStreamExt, future};
use iceberg::arrow::{ArrowReader, ArrowReaderBuilder};
use iceberg::io::FileIO;
use iceberg::scan::FileScanTaskStream;
use iceberg::spec::SnapshotRef;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Execution settings shared by assigned-file and feed-backed scans.
/// Reader settings are captured from the planning or decoding context. Clones
/// share the process-local reader; decoding constructs a fresh worker reader.
#[derive(Clone)]
pub(crate) struct ScanConfig {
    pub(crate) schema: SchemaRef,
    pub(crate) partitioning: Partitioning,
    pub(crate) fetch: Option<usize>,
    pub(crate) metrics: ExecutionPlanMetricsSet,
    pub(crate) iceberg_file_io: FileIO,
    pub(crate) iceberg_runtime: iceberg::Runtime,
    // Clones share the delete loader across this source's local partitions.
    reader: ArrowReader,
    /// The session's `hash_join_inlist_pushdown_max_distinct_values` when
    /// `iceberg.dynamic_filter_pruning_enabled` is set, `None` otherwise.
    dynamic_filter_max_values: Option<usize>,
    /// The file pruner shared by every partition this config opens, with the filters it was
    /// built for. Clones share it; a clone opened with other filters replaces it.
    file_pruner: Arc<Mutex<Option<SharedPruner>>>,
}

/// A [`PartitionFilePruner`] and the kept filters it was built over.
type SharedPruner = (Vec<Arc<dyn PhysicalExpr>>, Option<Arc<PartitionFilePruner>>);

impl ScanConfig {
    pub(crate) fn new(
        schema: SchemaRef,
        partitioning: Partitioning,
        fetch: Option<usize>,
        iceberg_file_io: FileIO,
        iceberg_runtime: iceberg::Runtime,
        context: &TaskContext,
    ) -> Self {
        let config = IcebergConfig::from_config_options(context.session_config().options());
        let reader = ArrowReaderBuilder::new(iceberg_file_io.clone(), iceberg_runtime.clone())
            .with_batch_size(context.session_config().batch_size())
            .with_data_file_concurrency_limit(config.data_file_concurrency_limit)
            .with_row_group_filtering_enabled(config.row_group_filtering_enabled)
            .with_row_selection_enabled(config.row_selection_enabled)
            .build();
        let dynamic_filter_max_values = config.dynamic_filter_pruning_enabled.then(|| {
            context
                .session_config()
                .options()
                .optimizer
                .hash_join_inlist_pushdown_max_distinct_values
        });
        Self {
            schema,
            partitioning,
            fetch,
            metrics: Default::default(),
            iceberg_file_io,
            iceberg_runtime,
            reader,
            dynamic_filter_max_values,
            file_pruner: Default::default(),
        }
    }

    /// Reads `tasks`. `filters` are the source's kept filters, against [`Self::schema`].
    ///
    /// When `iceberg.dynamic_filter_pruning_enabled` is set and some of them are dynamic, data
    /// files whose identity partition values can't match the dynamic filters are dropped from
    /// `tasks` before the reader sees them, and counted in the `files_skipped` metric. Each
    /// task is checked when the reader pulls it, with the filters' value at that time. The
    /// reader pulls up to `iceberg.data_file_concurrency_limit` tasks ahead of the files it
    /// is reading, so a filter that tightens later doesn't skip tasks already pulled: it only
    /// skips fewer files, never a file that could match.
    pub(crate) fn open(
        &self,
        tasks: FileScanTaskStream,
        filters: &[Arc<dyn PhysicalExpr>],
        partition: usize,
    ) -> Result<SendableRecordBatchStream> {
        let tasks = match self.file_pruner(filters) {
            Some(pruner) => {
                let files_skipped =
                    MetricBuilder::new(&self.metrics).counter("files_skipped", partition);
                tasks
                    .filter(move |task| {
                        let skip = task.as_ref().is_ok_and(|task| pruner.should_skip(task));
                        if skip {
                            files_skipped.add(1);
                        }
                        future::ready(!skip)
                    })
                    .boxed()
            }
            None => tasks,
        };
        let stream = self
            .reader
            .clone()
            .read(tasks)
            .map(|result| result.stream())
            .map_err(df_err)?
            .map_err(df_err);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )) as SendableRecordBatchStream;

        let metrics = BaselineMetrics::new(&self.metrics, partition);

        Ok(Box::pin(LimitStream::new(stream, 0, self.fetch, metrics)))
    }
}

impl ScanConfig {
    /// The pruner for `filters`, built once and shared by every partition opened with the same
    /// filters, so their dynamic filters are compacted once per change of their value. `None`
    /// when pruning is disabled or no filter is dynamic.
    fn file_pruner(&self, filters: &[Arc<dyn PhysicalExpr>]) -> Option<Arc<PartitionFilePruner>> {
        let max_values = self.dynamic_filter_max_values?;
        let mut shared = self.file_pruner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((built_for, pruner)) = shared.as_ref()
            && built_for.len() == filters.len()
            && built_for
                .iter()
                .zip(filters)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        {
            return pruner.clone();
        }
        let pruner = CompactedDynamicFilters::try_new(filters, &self.schema, max_values)
            .map(|compacted| Arc::new(PartitionFilePruner::new(Arc::new(compacted), &self.schema)));
        *shared = Some((filters.to_vec(), pruner.clone()));
        pruner
    }
}

impl fmt::Debug for ScanConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanConfig")
            .field("schema", &self.schema)
            .field("partitioning", &self.partitioning)
            .field("fetch", &self.fetch)
            .field("metrics", &self.metrics)
            .field("iceberg_file_io", &self.iceberg_file_io)
            .field("iceberg_runtime", &self.iceberg_runtime)
            .finish_non_exhaustive()
    }
}

/// Snapshot summary keys defined by the Iceberg table spec:
/// https://iceberg.apache.org/spec/#optional-snapshot-summary-fields
///
/// iceberg-rust defines them privately:
/// https://github.com/apache/iceberg-rust/blob/4168a0b2950dc5f85588e5cb3ab6796e5228b309/crates/iceberg/src/spec/snapshot_summary.rs#L46-L47
const TOTAL_RECORDS: &str = "total-records";
const TOTAL_FILE_SIZE: &str = "total-files-size";

/// Getting stats out of snapshot's additional properties (no I/O overhead)
pub(crate) fn stats_from_snapshot(
    snapshot: Option<SnapshotRef>,
    schema: &SchemaRef,
) -> Result<Statistics> {
    let Some(snap) = snapshot else {
        // A table with no current snapshot has never had a commit. It was created, but zero data files were added
        return Ok(Statistics {
            num_rows: Precision::Exact(0),
            total_byte_size: Precision::Exact(0),
            column_statistics: vec![ColumnStatistics::new_unknown(); schema.fields().len()],
        });
    };
    let props = &snap.summary().additional_properties;

    let num_rows = props
        .get(TOTAL_RECORDS)
        .and_then(|v| v.parse().ok())
        .map(Precision::Exact)
        .unwrap_or(Precision::Absent);
    let total_byte_size = props
        .get(TOTAL_FILE_SIZE)
        .and_then(|v| v.parse().ok())
        .map(Precision::Exact)
        .unwrap_or(Precision::Absent);

    Ok(Statistics {
        num_rows,
        total_byte_size,
        column_statistics: vec![ColumnStatistics::new_unknown(); schema.fields().len()],
    })
}
