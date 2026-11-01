//! Publishes a scan's kept dynamic filters to the Iceberg Arrow reader, so each data file can
//! be pruned with the filters' latest values when the reader starts it.

use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use iceberg::arrow::{RuntimePredicateProvider, RuntimePredicateSnapshot, arrow_schema_to_schema};
use iceberg::spec::Schema;

use crate::common::dynamic_domain::CompactedDynamicFilters;
use crate::common::physical_expr_to_predicate::physical_expr_to_predicate;

/// Supplies the scan's kept dynamic filters, compacted by [`CompactedDynamicFilters`], to the
/// Iceberg Arrow reader as a runtime predicate.
///
/// The reader asks for [`Self::generation`] when each data-file task starts and calls
/// [`Self::snapshot`] only when it changes. Nothing here waits for a filter to be updated or
/// completed: the reader uses whatever value the filters hold at that moment.
///
/// It converts the scan's one shared [`CompactedDynamicFilters`], the same instance the file
/// pruner uses, so the filters are compacted once per change for the whole scan. How this meets
/// the [`RuntimePredicateProvider`] contract:
/// * **Generation.** [`CompactedDynamicFilters::generation`], the sum of the dynamic filters'
///   generations. Each filter's generation starts at 1 and grows by one on every update, so the
///   sum grows on any change and is never reused. A snapshot carries the generation its value
///   was compacted for, which is never older than the generation the reader last read.
/// * **Tighten-only.** The dynamic filters reaching a scan only tighten within one execution,
///   and the compacted value is latched: each compaction is intersected with the one published
///   before, so compaction's widening (a float bound reaching zero, a value set crossing the
///   cap) can't loosen it. A later snapshot therefore never admits a row an earlier one ruled
///   out. See [`crate::common::dynamic_domain`] for the invariant and its limits.
/// * **Exactness.** There is no `FilterExec` above the scan for a dynamic filter. The operator
///   that produced it makes the result exact (the hash join probes its hash table, TopK and
///   aggregates re-check their threshold), so pruning only decides what is read. Conversions
///   that are not exact, such as float comparisons that must keep `NaN`, only admit more rows.
/// * **Fail open.** Filters that can't be converted publish no predicate, and the reader treats
///   provider and binding errors as "no pruning".
pub(crate) struct DynamicFilterPredicateProvider {
    filters: Arc<CompactedDynamicFilters>,
    arrow_schema: SchemaRef,
    iceberg_schema: Schema,
}

impl DynamicFilterPredicateProvider {
    /// Builds a provider over `filters`, expressed against `arrow_schema`, the scan's output
    /// schema.
    ///
    /// Returns `None` when the output schema can't be mapped back to Iceberg fields; the scan
    /// then reads without runtime pruning.
    pub(crate) fn try_new(
        filters: Arc<CompactedDynamicFilters>,
        arrow_schema: &SchemaRef,
    ) -> Option<Self> {
        let iceberg_schema = arrow_schema_to_schema(arrow_schema).ok()?;
        Some(Self {
            filters,
            arrow_schema: Arc::clone(arrow_schema),
            iceberg_schema,
        })
    }
}

impl RuntimePredicateProvider for DynamicFilterPredicateProvider {
    fn generation(&self) -> u64 {
        self.filters.generation()
    }

    fn snapshot(&self) -> iceberg::Result<RuntimePredicateSnapshot> {
        let (generation, expr) = self.filters.current();
        let predicate = physical_expr_to_predicate(&expr, &self.arrow_schema, &self.iceberg_schema);
        Ok(RuntimePredicateSnapshot::new(predicate, generation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, DynamicFilterPhysicalExpr, col, lit};

    #[test]
    fn snapshot_follows_dynamic_filter_updates() {
        let field = Field::new("id", DataType::Int64, false)
            .with_metadata([("PARQUET:field_id".to_string(), "1".to_string())].into());
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        // Starts as `true`, like a hash join's placeholder.
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col("id", &schema).unwrap()],
            lit(true),
        ));
        let filters = [Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>];
        let compacted = CompactedDynamicFilters::try_new(&filters, &schema, 150).unwrap();
        let provider =
            DynamicFilterPredicateProvider::try_new(Arc::new(compacted), &schema).unwrap();
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

    fn snapshot(provider: &DynamicFilterPredicateProvider) -> String {
        let snapshot = provider.snapshot().unwrap();
        format!(
            "generation={}, predicate={:?}",
            snapshot.generation(),
            snapshot.into_predicate().map(|p| p.to_string())
        )
    }
}
