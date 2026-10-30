//! Publishes a scan's kept dynamic filters to the Iceberg Arrow reader, so each data file can
//! be pruned with the filters' latest values when the reader starts it.

use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_expr::{PhysicalExpr, conjunction};
use datafusion::physical_expr_common::physical_expr::snapshot_generation;
use iceberg::arrow::{RuntimePredicateProvider, RuntimePredicateSnapshot, arrow_schema_to_schema};
use iceberg::spec::Schema;

use crate::common::physical_expr_to_predicate::physical_expr_to_predicate;

/// Supplies the AND of a scan's kept dynamic filters to the Iceberg Arrow reader as a runtime
/// predicate.
///
/// The reader asks for [`Self::generation`] when each data-file task starts and calls
/// [`Self::snapshot`] only when it changes. Nothing here waits for a filter to be updated or
/// completed: the reader uses whatever value the filters hold at that moment.
///
/// How this meets the [`RuntimePredicateProvider`] contract:
/// * **Generation.** The sum of the dynamic filters' generations. Each filter's generation
///   starts at 1 and grows by one on every update, so the sum grows on any change and is never
///   reused. [`Self::snapshot`] reads the generation *before* the filter values, so a snapshot
///   may carry a value newer (tighter) than its generation, never an older one.
/// * **Tighten-only.** Published predicates may only tighten. The dynamic filters DataFusion
///   produces (a hash join's build-side bounds and values, TopK and aggregate bounds) only
///   tighten over time; this is the same assumption DataFusion's own Parquet scan makes when it
///   prunes with them. The `FilterExec` above the scan still evaluates every filter, so pruning
///   only decides what is read, never which rows are returned.
/// * **Fail open.** Filters that can't be converted publish no predicate, and the reader treats
///   provider and binding errors as "no pruning".
pub(crate) struct DynamicFilterPredicateProvider {
    filters: Vec<Arc<dyn PhysicalExpr>>,
    arrow_schema: SchemaRef,
    iceberg_schema: Schema,
}

impl DynamicFilterPredicateProvider {
    /// Builds a provider over the dynamic filters found in `filters` (at the top level or nested
    /// inside a kept expression), all expressed against `arrow_schema`, the scan's output schema.
    ///
    /// Returns `None` when there is no dynamic filter, or when the output schema can't be mapped
    /// back to Iceberg fields; the scan then reads without runtime pruning.
    pub(crate) fn try_new(
        filters: &[Arc<dyn PhysicalExpr>],
        arrow_schema: &SchemaRef,
    ) -> Option<Self> {
        let mut dynamic = Vec::new();
        for filter in filters {
            filter
                .apply(|expr| {
                    if expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some() {
                        dynamic.push(Arc::clone(expr));
                        return Ok(TreeNodeRecursion::Jump);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .ok()?;
        }
        if dynamic.is_empty() {
            return None;
        }
        let iceberg_schema = arrow_schema_to_schema(arrow_schema).ok()?;
        Some(Self {
            filters: dynamic,
            arrow_schema: Arc::clone(arrow_schema),
            iceberg_schema,
        })
    }
}

impl RuntimePredicateProvider for DynamicFilterPredicateProvider {
    fn generation(&self) -> u64 {
        self.filters
            .iter()
            .fold(0u64, |acc, f| acc.wrapping_add(snapshot_generation(f)))
    }

    fn snapshot(&self) -> iceberg::Result<RuntimePredicateSnapshot> {
        let generation = self.generation();
        let expr = conjunction(self.filters.iter().cloned());
        let predicate = physical_expr_to_predicate(&expr, &self.arrow_schema, &self.iceberg_schema);
        Ok(RuntimePredicateSnapshot::new(predicate, generation))
    }
}
