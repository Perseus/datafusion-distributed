//! Rewrites `CASE` expressions into the OR of their result branches, so that the routing
//! filters DataFusion builds for partitioned hash joins can be used for pruning.
//!
//! For a partitioned hash join, DataFusion's dynamic filter routes each probe row to the
//! filter of the build partition it hashes (or range-partitions) to:
//!
//! ```text
//! CASE hash_repartition(keys) % N WHEN 0 THEN F0 WHEN 1 THEN F1 ... ELSE <bool> END
//! ```
//!
//! Neither Iceberg predicates nor DataFusion's `PruningPredicate` can evaluate the routing
//! hash against statistics, so as written the filter prunes nothing. DataFusion 55 has no
//! simplification or pruning support for `CASE`, hence this small rewrite.

use std::sync::Arc;

use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, CaseExpr, lit};

/// The OR of `case`'s THEN branches and its ELSE branch, if any.
///
/// This over-approximates the rows for which `case` is `true`, whatever the base and WHEN
/// expressions are, even when their values can't all be enumerated: every row takes exactly
/// one THEN branch or the ELSE, so if the CASE is `true` for a row, the branch it took is
/// `true` for that row, and so is the OR. A missing ELSE yields NULL, which never passes a
/// filter, so it is left out. The WHEN and base expressions are dropped, which is why the
/// result may match more rows than the CASE, never fewer.
pub(crate) fn case_branches_disjunction(case: &CaseExpr) -> Arc<dyn PhysicalExpr> {
    case.when_then_expr()
        .iter()
        .map(|(_, then)| Arc::clone(then))
        .chain(case.else_expr().cloned())
        .reduce(|acc, branch| Arc::new(BinaryExpr::new(acc, Operator::Or, branch)))
        .unwrap_or_else(|| lit(false))
}

/// Replaces each `CASE` reachable from the root of `expr` through AND and OR (including
/// inside the branches of a replaced `CASE`) by [`case_branches_disjunction`].
///
/// AND and OR are monotonic, so widening an operand widens the whole expression, and the
/// result matches every row `expr` keeps. A `CASE` anywhere else, e.g. under NOT, is left as
/// is: widening it there could narrow the result.
pub(crate) fn rewrite_case_as_or(expr: &Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        return rewrite_case_as_or(&case_branches_disjunction(case));
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>()
        && matches!(binary.op(), Operator::And | Operator::Or)
    {
        return Arc::new(BinaryExpr::new(
            rewrite_case_as_or(binary.left()),
            *binary.op(),
            rewrite_case_as_or(binary.right()),
        ));
    }
    Arc::clone(expr)
}
