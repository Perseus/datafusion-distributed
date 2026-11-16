use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static DBG_REFRESH: AtomicUsize = AtomicUsize::new(0);
static DBG_READER: AtomicUsize = AtomicUsize::new(0);

fn dbg_now() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn dbg_cut(s: String) -> String {
    s.chars().take(4000).collect()
}

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::HashMap;
use datafusion::physical_expr::{PhysicalExpr, conjunction};
use datafusion::physical_expr_common::physical_expr::{
    snapshot_generation, snapshot_physical_expr,
};
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};
use datafusion::physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricType, PruningMetrics,
};
use iceberg::expr::Predicate;
use iceberg::scan::FileScanTask;

use crate::common::dynamic_filters::file_bounds::FileBounds;
use crate::common::dynamic_filters::partition_filter::PartitionFilter;
use crate::common::dynamic_filters::physical_predicate::convert_physical_filter_to_predicate;
use crate::common::dynamic_filters::row_group_filter::with_dynamic_filter;

/// Applies the scan's dynamic filters to each file just before the reader opens it. It skips
/// files whose partition value or column bounds can't match the filters. For the other files, it
/// can add the filters to the file's own filter, so the reader can skip row groups.
pub(crate) struct DynamicFilterPruner {
    /// The scan's dynamic filters, combined with AND.
    dynamic_filters: Arc<dyn PhysicalExpr>,

    /// The generation of `dynamic_filters` that `predicate` was built from.
    cached_snapshot_generation: u64,

    /// The dynamic filters are stored as iceberg Predicates after being converted
    /// `None` if the filters can't be converted
    predicate: Option<Predicate>,

    /// A map of partition spec IDs to their corresponding partition filter executors
    partition_filters: HashMap<i32, Option<PartitionFilter>>,

    /// The scan's output schema, which the filters' columns refer to.
    schema: SchemaRef,

    /// The filters as a predicate over file column bounds. `None` if the bounds can't rule
    /// anything out.
    bounds_predicate: Option<Arc<PruningPredicate>>,

    /// Metrics to track pruning
    files: PruningMetrics,

    /// Files that passed the partition check, pruned or matched by their column bounds.
    files_by_bounds: PruningMetrics,
}

impl DynamicFilterPruner {
    pub(crate) fn new(
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
        schema: SchemaRef,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Self {
        Self {
            dynamic_filters: conjunction(dynamic_filters),
            cached_snapshot_generation: 0,
            predicate: None,
            partition_filters: HashMap::new(),
            schema,
            bounds_predicate: None,
            files: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .pruning_metrics("files_pruned_by_partition", partition),
            files_by_bounds: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .pruning_metrics("files_pruned_by_bounds", partition),
        }
    }

    /// Whether the filters rule out every row in the task's partition. Counts the task as pruned
    /// or matched
    pub(crate) fn should_prune(&mut self, task: &FileScanTask) -> bool {
        if self.can_skip(task) {
            self.files.add_pruned(1);
            return true;
        }
        self.files.add_matched(1);
        let schema = Arc::clone(&self.schema);
        let Some(bounds) = FileBounds::new(&schema, task) else {
            return false;
        };
        let prune = self.bounds_rule_out(&bounds);
        if prune {
            self.files_by_bounds.add_pruned(1);
        } else {
            self.files_by_bounds.add_matched(1);
        }
        prune
    }

    /// Adds the dynamic filters to the task's own filter, so the reader can use them to skip row
    /// groups. Returns the task unchanged if that fails.
    pub(crate) fn with_reader_filter(&mut self, task: FileScanTask) -> FileScanTask {
        self.refresh();
        let Some(predicate) = &self.predicate else {
            if DBG_READER.fetch_add(1, Ordering::Relaxed) < 40 {
                eprintln!("DFDBG {} reader_filter none file={}", dbg_now(), task.data_file_path().rsplit('/').next().unwrap_or(""));
            }
            return task;
        };
        let out = with_dynamic_filter(&task, predicate);
        if DBG_READER.fetch_add(1, Ordering::Relaxed) < 40 {
            eprintln!("DFDBG {} reader_filter attached={} file={} task_predicate={}", dbg_now(), out.is_some(), task.data_file_path().rsplit('/').next().unwrap_or(""),
                dbg_cut(format!("{:?}", out.as_ref().and_then(|t| t.predicate()).map(|p| p.to_string()))));
        }
        out.unwrap_or(task)
    }

    /// Whether no row in the task's partition can match the filters.
    fn can_skip(&mut self, task: &FileScanTask) -> bool {
        let (Some(spec), Some(partition)) = (task.partition_spec(), task.partition()) else {
            return false;
        };
        if spec.is_unpartitioned() {
            return false;
        }
        self.refresh();
        let Some(predicate) = &self.predicate else {
            return false;
        };

        // Every `FileScanTask` within a scan has the same Iceberg schema, taken from the selected
        // Iceberg snapshot.
        // The only difference in partition filters can come from the partition spec being different
        // between data files.
        // The PartitionFilter struct -
        // 1. Binds the `Predicate`, which just contains column names and values, to the actual
        //    schema, resolving column names to Iceberg field IDs etc.
        // 2. Projects the column filter into a partition filter. A partition spec is not limited
        //    to just direct column filters; it can have expressions like `bucket(2, x)` or
        //    `day(ts)`. A column filter like `ts >= '2024-03-15 10:00'` becomes
        //    `ts_day >= 2024-03-15` which can be checked against every data file's partition value.
        // These steps can be expensive, so we maintain a cache of a `PartitionFilter` per spec ID.
        let partition_filter = self
            .partition_filters
            .entry(spec.spec_id())
            .or_insert_with(|| {
                PartitionFilter::try_new(
                    predicate.clone(),
                    task.schema_ref(),
                    spec,
                    task.case_sensitive(),
                )
                .ok()
            });
        let Some(filter) = partition_filter.as_ref() else {
            return false;
        };

        // Check whether the file can be dropped, based on the partition values.
        // If there's an error while evaluating the filter,
        // we return `false` so that the file is not skipped
        filter.can_match(partition).is_ok_and(|matches| !matches)
    }

    /// Whether no value within the file's column bounds can match the filters.
    fn bounds_rule_out(&mut self, bounds: &FileBounds) -> bool {
        self.refresh();
        let Some(predicate) = &self.bounds_predicate else {
            return false;
        };
        // On an error, the file is read.
        predicate
            .prune(bounds)
            .is_ok_and(|keep| keep.iter().all(|keep| !keep))
    }

    /// Rebuilds the predicates when any of the filters has changed.
    fn refresh(&mut self) {
        // Read the current generation before snapshotting the filters.
        // The two reads aren't atomic, so a filter can change between them.
        //
        // In this order, the snapshot can be newer than the generation we have stored
        // which is okay - because we're still doing the most 'efficient' pruning
        //
        // In the opposite order, we could store a newer generation but an older filter,
        // which would prune less until the filter updated.
        let generation = snapshot_generation(&self.dynamic_filters);
        if self.cached_snapshot_generation == generation {
            return;
        }
        self.cached_snapshot_generation = generation;
        self.partition_filters.clear();
        let snapshot = snapshot_physical_expr(Arc::clone(&self.dynamic_filters)).ok();
        self.predicate = snapshot
            .as_ref()
            .and_then(convert_physical_filter_to_predicate);
        if DBG_REFRESH.fetch_add(1, Ordering::Relaxed) < 40 {
            eprintln!("DFDBG {} filter_refresh generation={} snapshot={} predicate={}", dbg_now(), generation,
                dbg_cut(format!("{:?}", snapshot.as_ref().map(|s| s.to_string()))),
                dbg_cut(format!("{:?}", self.predicate.as_ref().map(|p| p.to_string()))));
        }
        self.bounds_predicate = snapshot.and_then(|snapshot| {
            PruningPredicateBuilder::new()
                .with_file_schema(Arc::clone(&self.schema))
                .build(snapshot)
        });
    }
}
