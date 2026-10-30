//! Publishes a scan's kept dynamic filters to the Iceberg Arrow reader, so each data file can
//! be pruned with the filters' latest values when the reader starts it.

use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_expr::{PhysicalExpr, conjunction, split_conjunction};
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
    /// Builds a provider over the dynamic filters among `filters` and their top-level AND
    /// conjuncts, all expressed against `arrow_schema`, the scan's output schema.
    ///
    /// Returns `None` when there is no dynamic filter, or when the output schema can't be mapped
    /// back to Iceberg fields; the scan then reads without runtime pruning.
    pub(crate) fn try_new(
        filters: &[Arc<dyn PhysicalExpr>],
        arrow_schema: &SchemaRef,
    ) -> Option<Self> {
        // Only conjuncts may be used: a dynamic filter under OR, NOT or any other expression
        // doesn't restrict the rows the kept filter matches, so it is ignored (fail open).
        let dynamic: Vec<_> = filters
            .iter()
            .flat_map(split_conjunction)
            .filter(|expr| expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some())
            .cloned()
            .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{BinaryExpr, NotExpr, col, lit};
    use std::collections::HashMap;

    #[test]
    fn snapshot_follows_dynamic_filter_updates() {
        let schema = id_schema();
        let dynamic = id_dynamic_filter(&schema);
        let provider = DynamicFilterPredicateProvider::try_new(
            &[Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>],
            &schema,
        )
        .expect("a kept dynamic filter installs a provider");
        let before = provider.generation();
        insta::assert_snapshot!(snapshot(&provider), @r#"generation=1, predicate=Some("TRUE")"#);

        dynamic
            .update(Arc::new(BinaryExpr::new(
                col("id", &schema).unwrap(),
                Operator::LtEq,
                lit(10i64),
            )))
            .unwrap();

        assert!(provider.generation() > before);
        insta::assert_snapshot!(snapshot(&provider), @r#"generation=2, predicate=Some("id <= 10")"#);
    }

    #[test]
    fn dynamic_filter_and_static_conjunct_are_used() {
        let schema = id_schema();
        let dynamic = id_dynamic_filter(&schema);
        let filter = Arc::new(BinaryExpr::new(
            dynamic,
            Operator::And,
            Arc::new(BinaryExpr::new(
                col("id", &schema).unwrap(),
                Operator::Gt,
                lit(0i64),
            )),
        )) as Arc<dyn PhysicalExpr>;

        assert!(DynamicFilterPredicateProvider::try_new(&[filter], &schema).is_some());
    }

    #[test]
    fn dynamic_filter_under_or_is_ignored() {
        let schema = id_schema();
        let dynamic = id_dynamic_filter(&schema);
        let filter = Arc::new(BinaryExpr::new(
            dynamic,
            Operator::Or,
            Arc::new(BinaryExpr::new(
                col("id", &schema).unwrap(),
                Operator::Gt,
                lit(100i64),
            )),
        )) as Arc<dyn PhysicalExpr>;

        assert!(DynamicFilterPredicateProvider::try_new(&[filter], &schema).is_none());
    }

    #[test]
    fn dynamic_filter_under_not_is_ignored() {
        let schema = id_schema();
        let filter = Arc::new(NotExpr::new(id_dynamic_filter(&schema))) as Arc<dyn PhysicalExpr>;

        assert!(DynamicFilterPredicateProvider::try_new(&[filter], &schema).is_none());
    }

    /// A dynamic filter on `id`, starting as `true` like a hash join's placeholder.
    fn id_dynamic_filter(schema: &SchemaRef) -> Arc<DynamicFilterPhysicalExpr> {
        Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col("id", schema).unwrap()],
            lit(true),
        ))
    }

    /// An `id` column carrying its Iceberg field id, like a scan's output schema.
    fn id_schema() -> SchemaRef {
        let field = Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
            "PARQUET:field_id".to_string(),
            "1".to_string(),
        )]));
        Arc::new(ArrowSchema::new(vec![field]))
    }

    fn snapshot(provider: &DynamicFilterPredicateProvider) -> String {
        let snapshot = provider.snapshot().unwrap();
        format!(
            "generation={}, predicate={:?}",
            snapshot.generation(),
            snapshot.into_predicate().map(|p| p.to_string())
        )
    }
}
