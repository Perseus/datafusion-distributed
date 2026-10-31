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
/// How this meets the [`RuntimePredicateProvider`] contract:
/// * **Generation.** The sum of the dynamic filters' generations. Each filter's generation
///   starts at 1 and grows by one on every update, so the sum grows on any change and is never
///   reused. A snapshot may carry a value newer (tighter) than its generation, never an older
///   one.
/// * **Tighten-only.** Published predicates may only tighten. The dynamic filters DataFusion
///   produces (a hash join's build-side bounds and values, TopK and aggregate bounds) only
///   tighten over time; this is the same assumption DataFusion's own Parquet scan makes when it
///   prunes with them. The `FilterExec` above the scan still evaluates every filter, so pruning
///   only decides what is read, never which rows are returned.
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
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::error::Result;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, NotExpr, col, lit};
    use datafusion::physical_expr::expressions::{CaseExpr, InListExpr};
    use datafusion::physical_plan::joins::HashJoinExec;
    use datafusion::physical_plan::{ExecutionPlan, collect};
    use datafusion::prelude::SessionContext;
    use datafusion_distributed_iceberg_test_tools::create_taxi_table;
    use std::collections::HashMap;

    use crate::IcebergExt;

    #[test]
    fn snapshot_follows_dynamic_filter_updates() {
        let schema = id_schema();
        let dynamic = id_dynamic_filter(&schema);
        let provider = provider(&[Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>], &schema)
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

        assert!(provider(&[filter], &schema).is_some());
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

        assert!(provider(&[filter], &schema).is_none());
    }

    #[test]
    fn dynamic_filter_under_not_is_ignored() {
        let schema = id_schema();
        let filter = Arc::new(NotExpr::new(id_dynamic_filter(&schema))) as Arc<dyn PhysicalExpr>;

        assert!(provider(&[filter], &schema).is_none());
    }

    /// A partitioned hash join routes each probe row to its build partition's filter with a
    /// `CASE` on the row's hash, one branch per partition. The reader is handed one range and
    /// one value set for the join key instead of the OR of every branch.
    #[tokio::test]
    async fn partitioned_join_filter_reaches_the_reader_compacted() -> Result<()> {
        let table = create_taxi_table().await?;
        let mut state = SessionStateBuilder::new()
            .with_default_features()
            .with_iceberg_integration(Default::default());
        let options = state.config().get_or_insert_default().options_mut();
        options.set("datafusion.execution.target_partitions", "4")?;
        options.set(
            "datafusion.optimizer.hash_join_single_partition_threshold",
            "0",
        )?;
        options.set(
            "datafusion.optimizer.hash_join_single_partition_threshold_rows",
            "0",
        )?;
        options.set("iceberg.dynamic_filter_pruning_enabled", "true")?;
        let ctx = SessionContext::new_with_state(state.build());
        let location = table.metadata_location()?;
        ctx.sql(&format!(
            "CREATE EXTERNAL TABLE taxi STORED AS ICEBERG LOCATION '{location}'"
        ))
        .await?
        .collect()
        .await?;
        // The trips on two days that hash to different build partitions.
        let plan = ctx
            .sql(
                "SELECT COUNT(*) FROM taxi WHERE pickup_date IN (
                    SELECT pickup_date FROM taxi
                    WHERE pickup_date IN (DATE '2024-01-08', DATE '2024-01-14'))",
            )
            .await?
            .create_physical_plan()
            .await?;
        collect(Arc::clone(&plan), ctx.task_ctx()).await?;

        let mut join_filter = None;
        plan.apply(|node| {
            if let Some(join) = node.downcast_ref::<HashJoinExec>() {
                join_filter = Some((
                    Arc::clone(&join.dynamic_expressions_produced()[0]),
                    join.right().schema(),
                ));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let (filter, probe_schema) = join_filter.unwrap();
        // The join's filter as DataFusion built it: a CASE with a branch per build partition,
        // each with an IN list of its build rows' keys.
        let current = filter
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .unwrap()
            .current()?;
        let branches = current
            .downcast_ref::<CaseExpr>()
            .unwrap()
            .when_then_expr()
            .len();
        let mut in_list_items = 0;
        current.apply(|e| {
            if let Some(list) = e.downcast_ref::<InListExpr>() {
                in_list_items += list.list().len();
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let provider = provider(&[Arc::clone(&filter)], &probe_schema).unwrap();

        insta::assert_snapshot!(
            format!(
                "join filter: CASE with {branches} branches, {in_list_items} IN-list items\n\
                 reader predicate: {}",
                snapshot(&provider),
            ),
            @r#"
        join filter: CASE with 2 branches, 50000 IN-list items
        reader predicate: generation=2, predicate=Some("((pickup_date >= 2024-01-08) AND (pickup_date <= 2024-01-14)) AND (pickup_date IN (2024-01-14, 2024-01-08))")
        "#
        );
        Ok(())
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

    /// A provider over the dynamic filters among `filters`, as a scan installs it.
    fn provider(
        filters: &[Arc<dyn PhysicalExpr>],
        schema: &SchemaRef,
    ) -> Option<DynamicFilterPredicateProvider> {
        let compacted = CompactedDynamicFilters::try_new(filters, schema, 150)?;
        DynamicFilterPredicateProvider::try_new(Arc::new(compacted), schema)
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
