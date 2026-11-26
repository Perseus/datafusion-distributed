use std::sync::Arc;

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

use crate::common::dynamic_filters::file_statistics::FileStatistics;
use crate::common::dynamic_filters::partition_filter::PartitionFilter;
use crate::common::dynamic_filters::physical_predicate::convert_physical_filter_to_predicate;

/// Uses the scan's dynamic filters on each file just before the reader starts it. Skips files
/// whose partition value or column stats can't match the filters.
pub(crate) struct DynamicFilterPruner {
    /// The scan's dynamic filters, combined with AND.
    dynamic_filters: Arc<dyn PhysicalExpr>,

    /// The scan's output schema, which the dynamic filters' columns refer to.
    schema: SchemaRef,

    /// The largest IN list that `PruningPredicate` checks against column stats.
    max_in_list_size: usize,

    /// The generation of `dynamic_filters` that `predicate` was built from.
    cached_snapshot_generation: u64,

    /// The dynamic filters are stored as iceberg Predicates after being converted
    /// `None` if the filters can't be converted
    predicate: Option<Predicate>,

    /// The dynamic filters as a DataFusion pruning predicate, which we check against each
    /// file's column stats. `None` if no part of the filters can use stats.
    pruning_predicate: Option<Arc<PruningPredicate>>,

    /// A map of partition spec IDs to their corresponding partition filter executors
    partition_filters: HashMap<i32, Option<PartitionFilter>>,

    /// Files skipped or kept by their partition value
    files_by_partition: PruningMetrics,

    /// Files skipped or kept by their column stats. Files skipped by partition aren't counted.
    files_by_statistics: PruningMetrics,
}

impl DynamicFilterPruner {
    pub(crate) fn new(
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
        schema: SchemaRef,
        max_in_list_size: usize,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Self {
        Self {
            dynamic_filters: conjunction(dynamic_filters),
            schema,
            max_in_list_size,
            cached_snapshot_generation: 0,
            predicate: None,
            pruning_predicate: None,
            partition_filters: HashMap::new(),
            files_by_partition: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .pruning_metrics("files_pruned_by_partition", partition),
            files_by_statistics: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .pruning_metrics("files_pruned_by_statistics", partition),
        }
    }

    /// Whether the filters rule out every row in the task's file. Counts the task as pruned or
    /// matched by each check it reaches.
    pub(crate) fn should_prune(&mut self, task: &FileScanTask) -> bool {
        self.refresh();
        let by_partition = self.can_skip_by_partition(task);
        if count(&self.files_by_partition, by_partition) {
            return true;
        }
        count(&self.files_by_statistics, self.can_skip_by_statistics(task))
    }

    /// Whether no row in the task's partition can match the filters.
    fn can_skip_by_partition(&mut self, task: &FileScanTask) -> bool {
        let (Some(spec), Some(partition)) = (task.partition_spec(), task.partition()) else {
            return false;
        };
        if spec.is_unpartitioned() {
            return false;
        }
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

    /// Whether the task's column stats show that no row in its file can match the filters.
    fn can_skip_by_statistics(&self, task: &FileScanTask) -> bool {
        let (Some(pruning_predicate), Some(statistics)) =
            (&self.pruning_predicate, FileStatistics::try_new(task))
        else {
            return false;
        };
        // `prune` returns false for a file that can't match. On an error we keep the file.
        pruning_predicate
            .prune(&statistics)
            .is_ok_and(|keep| keep == [false])
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
        self.pruning_predicate = snapshot.and_then(|snapshot| {
            PruningPredicateBuilder::new()
                .with_file_schema(Arc::clone(&self.schema))
                .with_max_in_list_size(self.max_in_list_size)
                .build(snapshot)
        });
    }
}

/// Counts a file as pruned or matched, and returns `pruned`.
fn count(metrics: &PruningMetrics, pruned: bool) -> bool {
    if pruned {
        metrics.add_pruned(1);
    } else {
        metrics.add_matched(1);
    }
    pruned
}
