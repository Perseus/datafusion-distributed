use crate::DISTRIBUTED_DATAFUSION_TASK_ID_LABEL;
use datafusion::physical_plan::metrics::{Label, Metric, MetricsSet};
use std::sync::Arc;

use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    ReplaceChildrenOptions,
};
use delegate::delegate;
use std::fmt::{Debug, Formatter};

/// A transparent wrapper that delegates all execution to its child but returns custom metrics. This node is invisible during display.
/// The structure of a plan tree is closely tied to the [TaskMetricsRewriter].
pub(crate) struct MetricsWrapperExec {
    inner: Arc<dyn ExecutionPlan>,
    /// metrics for this plan node.
    metrics: MetricsSet,
}

impl MetricsWrapperExec {
    pub(crate) fn new(inner: Arc<dyn ExecutionPlan>, metrics: MetricsSet) -> Self {
        Self { inner, metrics }
    }

    #[cfg(test)]
    pub(crate) fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
}

/// MetricsWrapperExec is invisible during display.
impl DisplayAs for MetricsWrapperExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        self.inner.fmt_as(t, f)
    }
}

/// MetricsWrapperExec is visible when debugging.
impl Debug for MetricsWrapperExec {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "MetricsWrapperExec ({:?})", self.inner)
    }
}

impl ExecutionPlan for MetricsWrapperExec {
    delegate! {
        to self.inner {
            fn name(&self) -> &str;
            fn properties(&self) -> &Arc<PlanProperties>;
        }
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inner.children()
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        self.inner.apply_expressions(f)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    /// Passes `options` on to the wrapped node, so it can reuse its cached properties when the
    /// caller says the children's properties did not change.
    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MetricsWrapperExec {
            inner: Arc::clone(&self.inner).replace_children(children, options)?,
            metrics: self.metrics.clone(),
        }))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        unimplemented!("MetricsWrapperExec does not implement execute")
    }

    /// returns the wrapped metrics merged with any other present in
    /// the inner [ExecutionPlan].
    fn metrics(&self) -> Option<MetricsSet> {
        match self.inner.metrics() {
            None => Some(self.metrics.clone()),
            Some(local_metrics) => {
                let mut all_metrics = self.metrics.clone();

                for local_metric in local_metrics {
                    // When a node is executed in an in-process worker, the ExecutionPlan might not
                    // have suffered any [de]serialization. This means that execution metrics
                    // collected at runtime in the Worker will be automatically visible in the
                    // coordinator, as both happen to run in-process, and therefore, the pointer
                    // that holds the ExecutionPlan MetricsSet is the same.
                    //
                    // This means that we need to dedupe the metrics here, otherwise, we might
                    // double-count metrics:
                    // 1. What was collected in the local worker
                    // 2. What is automatically available in the coordinator because of pointer
                    //    equivalence.
                    //
                    // Workers sum each metric over partitions before reporting it, so a reported
                    // metric covers every local metric with the same name and labels.
                    if !self.metrics.iter().any(|wrapped| {
                        wrapped.value().name() == local_metric.value().name()
                            && same_labels_ignoring_task_id(wrapped, &local_metric)
                    }) {
                        all_metrics.push(local_metric);
                    }
                }
                Some(all_metrics)
            }
        }
    }

    fn downcast_delegate(&self) -> Option<&dyn ExecutionPlan> {
        Some(self.inner.as_ref())
    }
}

/// Whether `a` and `b` have the same labels, leaving out the task id that the coordinator adds to
/// reported metrics.
fn same_labels_ignoring_task_id(a: &Metric, b: &Metric) -> bool {
    let not_task_id = |label: &&Label| label.name() != DISTRIBUTED_DATAFUSION_TASK_ID_LABEL;
    a.labels()
        .iter()
        .filter(not_task_id)
        .eq(b.labels().iter().filter(not_task_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::empty::EmptyExec;

    #[test]
    fn replace_children_passes_options_to_inner() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let leaf = || Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>;
        let inner = Arc::new(CoalescePartitionsExec::new(leaf())) as Arc<dyn ExecutionPlan>;
        let wrapper = Arc::new(MetricsWrapperExec::new(
            Arc::clone(&inner),
            MetricsSet::new(),
        ));

        let keep = ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep);
        let replaced = Arc::clone(&wrapper).replace_children(vec![leaf()], keep)?;
        assert!(Arc::ptr_eq(inner.properties(), replaced.properties()));

        let recompute = ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
        let replaced = wrapper.replace_children(vec![leaf()], recompute)?;
        assert!(!Arc::ptr_eq(inner.properties(), replaced.properties()));
        Ok(())
    }
}
