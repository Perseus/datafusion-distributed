use std::sync::Arc;

use datafusion::physical_expr::{PhysicalExpr, conjunction};
use datafusion::physical_expr_common::physical_expr::{
    snapshot_generation, snapshot_physical_expr,
};
use datafusion::physical_plan::metrics::Count;
use iceberg::expr::{Bind, BoundPredicate};
use iceberg::scan::FileScanTask;

use crate::common::dynamic_filters::partition_skipper::PartitionSkipper;
use crate::common::dynamic_filters::row_group_filter::RowGroupFilter;
use crate::table_provider::physical_predicate::convert_physical_filter_to_predicate;

/// Uses the scan's dynamic filters on each file as the reader starts it. It skips files whose
/// partition value can't match the filters. It gives the other files the filters, so the reader
/// can skip their row groups.
pub(crate) struct DynamicFilterPruner {
    /// The scan's dynamic filters, combined with AND.
    dynamic_filters: Arc<dyn PhysicalExpr>,

    /// The generation of `dynamic_filters` that `predicate` was built from.
    cached_snapshot_generation: u64,

    /// The filters as an Iceberg predicate bound to the table schema.
    predicate: Option<BoundPredicate>,

    partition_skipper: PartitionSkipper,

    /// `None` when the reader doesn't get the filters.
    row_group_filter: Option<RowGroupFilter>,

    /// Counts the files the reader got the filters for.
    files_with_dynamic_filter: Count,
}

impl DynamicFilterPruner {
    /// `filter_row_groups` says whether to give the reader the filters.
    pub(crate) fn new(
        dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
        filter_row_groups: bool,
        files_with_dynamic_filter: Count,
    ) -> Self {
        Self {
            dynamic_filters: conjunction(dynamic_filters),
            cached_snapshot_generation: 0,
            predicate: None,
            partition_skipper: PartitionSkipper::default(),
            row_group_filter: filter_row_groups.then(RowGroupFilter::default),
            files_with_dynamic_filter,
        }
    }

    /// The task to read, or `None` to skip its file. Any error keeps the task as it is.
    pub(crate) fn prune(&mut self, task: FileScanTask) -> Option<FileScanTask> {
        self.refresh(&task);
        let Some(predicate) = &self.predicate else {
            return Some(task);
        };
        if self.partition_skipper.should_skip_task(predicate, &task) {
            return None;
        }
        let Some(row_group_filter) = &mut self.row_group_filter else {
            return Some(task);
        };
        match row_group_filter.with_dynamic_filter(predicate, &task) {
            Some(filtered) => {
                self.files_with_dynamic_filter.add(1);
                Some(filtered)
            }
            None => Some(task),
        }
    }

    /// Rebuilds the predicate when any of the filters has changed.
    fn refresh(&mut self, task: &FileScanTask) {
        // Read the generation before taking the snapshot. If a filter changes in between, the
        // next file sees a newer generation and rebuilds.
        let generation = snapshot_generation(&self.dynamic_filters);
        if self.cached_snapshot_generation == generation {
            return;
        }
        self.cached_snapshot_generation = generation;
        self.partition_skipper.clear();
        if let Some(row_group_filter) = &mut self.row_group_filter {
            row_group_filter.clear();
        }
        self.predicate = snapshot_physical_expr(Arc::clone(&self.dynamic_filters))
            .ok()
            .and_then(|snapshot| convert_physical_filter_to_predicate(&snapshot))
            .and_then(|predicate| {
                predicate
                    .bind(task.schema_ref(), task.case_sensitive())
                    .ok()
            });
    }
}
