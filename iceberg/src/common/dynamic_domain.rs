//! Compacts a scan's kept dynamic filters into one small per-column expression that is cheap
//! to evaluate and usable for pruning, the way runtime filters are compacted into per-column
//! domains.
//!
//! The filters DataFusion builds for joins can be large. A partitioned hash join routes each
//! probe row to its build partition's filter with
//! `CASE hash(keys) % N WHEN i THEN Fi ... ELSE false END`, a distributed query ORs one such
//! filter per producer task, and every `Fi` holds its own bounds and IN list. Evaluated as
//! written, such a predicate costs O(rows × size).
//!
//! Instead, the filters' current value is normalized into a [`Domain`]: for each probe column,
//! the non-null values that may pass (a range, or an exact value set that implies its range)
//! and whether NULL may pass. The domain is then turned back into a small expression,
//! `col >= min AND col <= max [AND col IN (set)] [OR col IS NULL]`, AND-ed across columns.
//!
//! # Only widens
//!
//! The domain of an expression contains every row for which the expression is `true`, and may
//! contain more. Each rule only keeps or widens:
//! * AND intersects the domains of its children; an empty intersection is `false`. A child
//!   that can't be described (a hash lookup, a cast, a NOT, ...) constrains nothing.
//! * OR, and the branches of a `CASE`, union the domains of their children: the range spans
//!   every branch, NULL passes if it passes any branch, a value set survives only if every
//!   branch has one and the union stays within the cap, and a column unconstrained in any
//!   branch is unconstrained. A `false` branch adds nothing; a `true` branch means no
//!   constraint. Every row takes exactly one THEN branch or the ELSE of a `CASE`, so the
//!   union of those branches covers every row the `CASE` keeps; the WHEN and base
//!   expressions are ignored. A missing ELSE yields NULL, which never passes.
//! * A multi-column constraint becomes one constraint per column, which is wider.
//!
//! Widening is sound only where widening the expression widens the filter: compaction is
//! applied to dynamic filters that stand alone or are top-level AND conjuncts of the scan's
//! kept filters (see [`dynamic_conjuncts`]), never to one under NOT, OR or any other
//! expression, and within a filter it only descends through AND, OR and `CASE` branches.
//!
//! There is no `FilterExec` above the scan for a dynamic filter. The query stays exact because
//! the filter's producer re-checks rows itself: the hash join probes its hash table, and TopK
//! and aggregates re-evaluate their own threshold. A compacted domain may therefore only decide
//! what is read, never which rows are returned.
//!
//! # Tighten-only invariant
//!
//! A consumer may skip data (a file, a row group) only if data skipped under one generation
//! stays skippable under every later one. The filters themselves only tighten for the dynamic
//! filters reaching a scan today:
//! * DataFusion's producers: the hash join updates its filter once, after its build barrier;
//!   TopK and aggregate filters only tighten.
//! * The distributed merge modes in `src/coordinator/dynamic_filter_registry.rs`:
//!   `UnionWhenAllComplete` and `ForwardFirstComplete` forward one value, and
//!   `IntersectOnUpdate` only tightens.
//!
//! A tightening filter does not imply a tightening compacted value. Compaction widens, and
//! can widen a later, tighter value more than an earlier one: a float bound that moves to NaN
//! or zero constrains nothing (TopK's `price < 1.0` compacts to `price <= 1.0`, its later
//! `price < 0.0` to `true`), and a union of value sets that crosses the cap becomes a range.
//! [`CompactedDynamicFilters`] therefore latches: it intersects each newly compacted domain
//! with the one it published before, which is sound because the filters only tighten, so the
//! value it publishes only tightens too. A per-file skip decision would be safe without the
//! latch, since each one uses a value covering every row the filters keep at that time and
//! later values keep a subset of those rows; the latch exists so that consumers which keep
//! reading one value while the filters change see it only tighten.
//!
//! Literals are used only when their type is exactly the column's, and float NaN and zero
//! literals constrain nothing, so float comparisons never depend on how NaN or the sign of
//! zero is ordered.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::Schema;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
    Literal, in_list, lit,
};
use datafusion::physical_expr::{PhysicalExpr, conjunction, split_conjunction};
use datafusion::physical_expr_common::physical_expr::{
    snapshot_generation, snapshot_physical_expr,
};

/// A scan's kept dynamic filters, compacted into per-column domains once per change of their value,
/// and latched: each compacted domain is intersected with the previously published one, so
/// the published value only tightens (see the module documentation).
///
/// The cache sits behind a lock, so one instance can be shared (in an `Arc`) by every
/// partition of a scan and still compacts once per generation; each instance it is built for
/// compacts on its own.
pub(crate) struct CompactedDynamicFilters {
    /// The AND of the dynamic filters among the scan's top-level conjuncts.
    filters: Arc<dyn PhysicalExpr>,
    schema: Arc<Schema>,
    max_values: usize,
    /// The latched domain and its expression for a generation of `filters`.
    cached: Mutex<Option<Published>>,
}

/// The value [`CompactedDynamicFilters`] published for one generation of its filters.
struct Published {
    generation: u64,
    domain: Domain,
    expr: Arc<dyn PhysicalExpr>,
}

impl CompactedDynamicFilters {
    /// Builds over the dynamic filters among `filters` and their top-level AND conjuncts,
    /// expressed against `schema`, the scan's output schema. Value sets larger than
    /// `max_values`, the session's `hash_join_inlist_pushdown_max_distinct_values`, are reduced
    /// to their range. Returns `None` when there is no dynamic filter.
    pub(crate) fn try_new(
        filters: &[Arc<dyn PhysicalExpr>],
        schema: &Arc<Schema>,
        max_values: usize,
    ) -> Option<Self> {
        let dynamic = dynamic_conjuncts(filters);
        if dynamic.is_empty() {
            return None;
        }
        Some(Self {
            filters: conjunction(dynamic),
            schema: Arc::clone(schema),
            max_values,
            cached: Mutex::new(None),
        })
    }

    /// Changes whenever the dynamic filters are updated, and is never reused.
    pub(crate) fn generation(&self) -> u64 {
        snapshot_generation(&self.filters)
    }

    /// The compacted expression for the filters' current value, intersected with every value
    /// published before, with the generation it was built for. It is rebuilt only when the
    /// generation changed since the last call. The generation is read before the value, so the
    /// expression is never older than its generation. It is read under the lock, so a caller
    /// delayed on the lock can't replace a newer cached expression with one tagged by an older
    /// generation.
    pub(crate) fn current(&self) -> (u64, Arc<dyn PhysicalExpr>) {
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        let generation = self.generation();
        if let Some(published) = cached.as_ref()
            && published.generation == generation
        {
            return (generation, Arc::clone(&published.expr));
        }
        let compactor = Compactor {
            schema: &self.schema,
            max_values: self.max_values,
        };
        let domain = match snapshot_physical_expr(Arc::clone(&self.filters)) {
            Ok(current) => compactor.domain(&current),
            // Fail open: no constraint.
            Err(_) => Domain::all(),
        };
        let domain = match cached.take() {
            Some(previous) => previous.domain.intersect(domain),
            None => domain,
        };
        let expr = domain.clone().into_expr(&self.schema);
        *cached = Some(Published {
            generation,
            domain,
            expr: Arc::clone(&expr),
        });
        (generation, expr)
    }
}

/// Returns the dynamic filters among `filters` and their top-level AND conjuncts.
///
/// Only conjuncts may be used for pruning: a dynamic filter under OR, NOT or any other
/// expression doesn't restrict the rows the kept filter matches, so it is ignored (fail open).
pub(crate) fn dynamic_conjuncts(filters: &[Arc<dyn PhysicalExpr>]) -> Vec<Arc<dyn PhysicalExpr>> {
    filters
        .iter()
        .flat_map(split_conjunction)
        .filter(|expr| expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some())
        .cloned()
        .collect()
}

/// Compacts `expr`, a snapshot of dynamic filters against `schema`, into an expression that is
/// `true` for every row `expr` is `true` for: one constraint per column, see
/// [`Domain::into_expr`]. Value sets larger than `max_values` are reduced to their range.
///
/// `expr` must be used where widening it is safe, see the module documentation.
#[cfg(test)]
pub(crate) fn compact(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    max_values: usize,
) -> Arc<dyn PhysicalExpr> {
    let compactor = Compactor { schema, max_values };
    compactor.domain(expr).into_expr(schema)
}

/// The rows that may pass a filter, as a constraint per column. A column absent from the map is
/// unconstrained.
#[derive(Debug, Clone)]
enum Domain {
    /// No row passes.
    None,
    Columns(BTreeMap<(usize, String), ColumnDomain>),
}

/// The values of one column that may pass.
#[derive(Debug, Clone)]
struct ColumnDomain {
    column: Column,
    values: Values,
    /// Whether NULL may pass.
    nulls: bool,
}

/// The non-null values that may pass.
#[derive(Debug, Clone)]
enum Values {
    /// Exactly these values, sorted and distinct. Empty means no non-null value passes.
    Set(Vec<ScalarValue>),
    /// The values within these inclusive bounds; a missing bound is unbounded.
    Range {
        min: Option<ScalarValue>,
        max: Option<ScalarValue>,
    },
}

impl Domain {
    fn all() -> Self {
        Self::Columns(BTreeMap::new())
    }

    fn column(domain: ColumnDomain) -> Self {
        let key = (domain.column.index(), domain.column.name().to_string());
        Self::Columns(BTreeMap::from([(key, domain)])).normalized()
    }

    /// Drops unconstrained columns, and becomes [`Domain::None`] if a column admits nothing.
    fn normalized(self) -> Self {
        let Self::Columns(mut columns) = self else {
            return self;
        };
        if columns.values().any(ColumnDomain::is_empty) {
            return Self::None;
        }
        columns.retain(|_, domain| !domain.is_unconstrained());
        Self::Columns(columns)
    }

    fn intersect(self, other: Self) -> Self {
        let (Self::Columns(mut left), Self::Columns(right)) = (self, other) else {
            return Self::None;
        };
        for (key, domain) in right {
            let merged = match left.remove(&key) {
                Some(existing) => existing.intersect(domain),
                None => domain,
            };
            left.insert(key, merged);
        }
        Self::Columns(left).normalized()
    }

    fn union(self, other: Self, max_values: usize) -> Self {
        let (mut left, right) = match (self, other) {
            (Self::None, other) | (other, Self::None) => return other,
            (Self::Columns(left), Self::Columns(right)) => (left, right),
        };
        // Only columns constrained in both branches stay constrained.
        let mut right = right;
        left.retain(|key, _| right.contains_key(key));
        let columns = left
            .into_iter()
            .map(|(key, domain)| {
                let other = right.remove(&key).expect("kept keys are in both");
                (key, domain.union(other, max_values))
            })
            .collect();
        Self::Columns(columns).normalized()
    }

    /// For each constrained column `col`, `col >= min AND col <= max [AND col IN (set)]`, or
    /// `col = v` for a single value, with `OR col IS NULL` when NULL may pass, AND-ed across
    /// columns. A set's range is its smallest and largest value.
    fn into_expr(self, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        match self {
            Self::None => lit(false),
            Self::Columns(columns) => {
                conjunction(columns.into_values().map(|domain| domain.into_expr(schema)))
            }
        }
    }
}

impl ColumnDomain {
    fn is_empty(&self) -> bool {
        !self.nulls && matches!(&self.values, Values::Set(values) if values.is_empty())
    }

    fn is_unconstrained(&self) -> bool {
        self.nulls
            && matches!(
                &self.values,
                Values::Range {
                    min: None,
                    max: None
                }
            )
    }

    fn intersect(self, other: Self) -> Self {
        Self {
            column: self.column,
            values: self.values.intersect(other.values),
            nulls: self.nulls && other.nulls,
        }
    }

    fn union(self, other: Self, max_values: usize) -> Self {
        Self {
            column: self.column,
            values: self.values.union(other.values, max_values),
            nulls: self.nulls || other.nulls,
        }
    }

    fn into_expr(self, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        let column: Arc<dyn PhysicalExpr> = Arc::new(self.column);
        let compare = |op, value: ScalarValue| -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(Arc::clone(&column), op, lit(value)))
        };
        let non_null: Option<Arc<dyn PhysicalExpr>> = match self.values {
            Values::Set(values) => match values.as_slice() {
                [] => None,
                [value] => Some(compare(Operator::Eq, value.clone())),
                [first, .., last] => {
                    let range = [
                        compare(Operator::GtEq, first.clone()),
                        compare(Operator::LtEq, last.clone()),
                    ];
                    let list = values.iter().cloned().map(lit).collect();
                    // Building the IN list only fails on mismatched types; the range alone
                    // still holds every value.
                    let set = in_list(Arc::clone(&column), list, &false, schema).ok();
                    Some(conjunction(range.into_iter().chain(set)))
                }
            },
            Values::Range { min, max } => {
                let bounds: Vec<_> = min
                    .map(|min| compare(Operator::GtEq, min))
                    .into_iter()
                    .chain(max.map(|max| compare(Operator::LtEq, max)))
                    .collect();
                Some(if bounds.is_empty() {
                    Arc::new(IsNotNullExpr::new(Arc::clone(&column)))
                } else {
                    conjunction(bounds)
                })
            }
        };
        let is_null =
            || -> Arc<dyn PhysicalExpr> { Arc::new(IsNullExpr::new(Arc::clone(&column))) };
        match (non_null, self.nulls) {
            (Some(non_null), true) => Arc::new(BinaryExpr::new(non_null, Operator::Or, is_null())),
            (Some(non_null), false) => non_null,
            (None, true) => is_null(),
            // Empty domains are normalized to `Domain::None`.
            (None, false) => lit(false),
        }
    }
}

impl Values {
    fn any() -> Self {
        Self::Range {
            min: None,
            max: None,
        }
    }

    fn intersect(self, other: Self) -> Self {
        match (self, other) {
            (Self::Set(left), Self::Set(right)) => {
                Self::Set(left.into_iter().filter(|v| right.contains(v)).collect())
            }
            (Self::Set(values), Self::Range { min, max })
            | (Self::Range { min, max }, Self::Set(values)) => Self::Set(
                values
                    .into_iter()
                    .filter(|v| within(v, min.as_ref(), max.as_ref()))
                    .collect(),
            ),
            (
                Self::Range {
                    min: min_a,
                    max: max_a,
                },
                Self::Range {
                    min: min_b,
                    max: max_b,
                },
            ) => {
                // Keeping either bound when they can't be compared only widens.
                let min = tighter(min_a, min_b, Ordering::Greater);
                let max = tighter(max_a, max_b, Ordering::Less);
                match (&min, &max) {
                    (Some(lo), Some(hi)) if lo.partial_cmp(hi) == Some(Ordering::Greater) => {
                        Self::Set(Vec::new())
                    }
                    _ => Self::Range { min, max },
                }
            }
        }
    }

    fn union(self, other: Self, max_values: usize) -> Self {
        match (self, other) {
            (Self::Set(empty), other) | (other, Self::Set(empty)) if empty.is_empty() => other,
            (Self::Set(left), Self::Set(right)) => {
                let merged = sorted_distinct(left.into_iter().chain(right).collect());
                if merged.len() <= max_values {
                    Self::Set(merged)
                } else {
                    Self::Set(merged).into_range()
                }
            }
            (left, right) => match (left.into_range(), right.into_range()) {
                (
                    Self::Range {
                        min: min_a,
                        max: max_a,
                    },
                    Self::Range {
                        min: min_b,
                        max: max_b,
                    },
                ) => Self::Range {
                    min: wider(min_a, min_b, Ordering::Less),
                    max: wider(max_a, max_b, Ordering::Greater),
                },
                _ => Self::any(),
            },
        }
    }

    /// The range of a non-empty set, from its smallest to its largest value.
    fn into_range(self) -> Self {
        match self {
            Self::Set(values) => Self::Range {
                min: values.first().cloned(),
                max: values.last().cloned(),
            },
            range => range,
        }
    }
}

/// Of two bounds, the one further in `direction`; a missing bound is unbounded and loses.
fn tighter(
    a: Option<ScalarValue>,
    b: Option<ScalarValue>,
    direction: Ordering,
) -> Option<ScalarValue> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if b.partial_cmp(&a) == Some(direction) {
            b
        } else {
            a
        }),
        (a, None) | (None, a) => a,
    }
}

/// Of two bounds, the one further in `direction`; a missing bound is unbounded and wins, and so
/// do bounds that can't be compared.
fn wider(
    a: Option<ScalarValue>,
    b: Option<ScalarValue>,
    direction: Ordering,
) -> Option<ScalarValue> {
    let (a, b) = (a?, b?);
    match b.partial_cmp(&a)? {
        ordering if ordering == direction => Some(b),
        _ => Some(a),
    }
}

/// Whether `value` is within the inclusive bounds. Values that can't be compared are kept.
fn within(value: &ScalarValue, min: Option<&ScalarValue>, max: Option<&ScalarValue>) -> bool {
    let below = min.is_some_and(|min| value.partial_cmp(min) == Some(Ordering::Less));
    let above = max.is_some_and(|max| value.partial_cmp(max) == Some(Ordering::Greater));
    !below && !above
}

fn sorted_distinct(mut values: Vec<ScalarValue>) -> Vec<ScalarValue> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values.dedup();
    values
}

struct Compactor<'a> {
    schema: &'a Schema,
    max_values: usize,
}

impl Compactor<'_> {
    fn domain(&self, expr: &Arc<dyn PhysicalExpr>) -> Domain {
        if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
            return match binary.op() {
                Operator::And => self
                    .domain(binary.left())
                    .intersect(self.domain(binary.right())),
                Operator::Or => self
                    .domain(binary.left())
                    .union(self.domain(binary.right()), self.max_values),
                op => self.comparison(binary.left(), *op, binary.right()),
            };
        }
        if let Some(case) = expr.downcast_ref::<CaseExpr>() {
            return case
                .when_then_expr()
                .iter()
                .map(|(_, then)| then)
                .chain(case.else_expr())
                .map(|branch| self.domain(branch))
                .fold(Domain::None, |acc, branch| {
                    acc.union(branch, self.max_values)
                });
        }
        if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
            return self.in_list(in_list);
        }
        if let Some(is_null) = expr.downcast_ref::<IsNullExpr>()
            && let Some(column) = self.column(is_null.arg())
        {
            return Domain::column(ColumnDomain {
                column: column.clone(),
                values: Values::Set(Vec::new()),
                nulls: true,
            });
        }
        if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>()
            && let Some(column) = self.column(is_not_null.arg())
        {
            return Domain::column(ColumnDomain {
                column: column.clone(),
                values: Values::any(),
                nulls: false,
            });
        }
        if let Some(literal) = expr.downcast_ref::<Literal>() {
            return match literal.value() {
                ScalarValue::Boolean(Some(true)) => Domain::all(),
                // `false` and NULL never pass.
                ScalarValue::Boolean(_) | ScalarValue::Null => Domain::None,
                _ => Domain::all(),
            };
        }
        // Anything else (hash lookups, casts, NOT, ...) constrains nothing.
        Domain::all()
    }

    fn comparison(
        &self,
        left: &Arc<dyn PhysicalExpr>,
        op: Operator,
        right: &Arc<dyn PhysicalExpr>,
    ) -> Domain {
        let (column, literal, op) = match (self.column(left), self.column(right)) {
            (Some(column), None) => (column, right, op),
            (None, Some(column)) => match op.swap() {
                Some(op) => (column, left, op),
                None => return Domain::all(),
            },
            _ => return Domain::all(),
        };
        let Some(value) = self.literal(column, literal) else {
            return Domain::all();
        };
        // Strict bounds are kept inclusive, which only widens.
        let values = match op {
            Operator::Eq => Values::Set(vec![value]),
            Operator::Lt | Operator::LtEq => Values::Range {
                min: None,
                max: Some(value),
            },
            Operator::Gt | Operator::GtEq => Values::Range {
                min: Some(value),
                max: None,
            },
            _ => return Domain::all(),
        };
        Domain::column(ColumnDomain {
            column: column.clone(),
            values,
            nulls: false,
        })
    }

    fn in_list(&self, in_list: &InListExpr) -> Domain {
        let Some(column) = self.column(in_list.expr()) else {
            return Domain::all();
        };
        if in_list.negated() {
            return Domain::all();
        }
        let mut values = Vec::with_capacity(in_list.list().len());
        for item in in_list.list() {
            // A NULL item never matches. Any other unusable item makes the set unknown:
            // leaving it out would narrow the set.
            if item
                .downcast_ref::<Literal>()
                .is_some_and(|l| l.value().is_null())
            {
                continue;
            }
            match self.literal(column, item) {
                Some(value) => values.push(value),
                None => return Domain::all(),
            }
        }
        let values = Values::Set(sorted_distinct(values));
        let values = match &values {
            Values::Set(set) if set.len() > self.max_values => values.into_range(),
            _ => values,
        };
        Domain::column(ColumnDomain {
            column: column.clone(),
            values,
            nulls: false,
        })
    }

    /// A bare column whose name agrees with the input field at its index.
    fn column<'e>(&self, expr: &'e Arc<dyn PhysicalExpr>) -> Option<&'e Column> {
        let column = expr.downcast_ref::<Column>()?;
        let field = self.schema.fields().get(column.index())?;
        (field.name() == column.name()).then_some(column)
    }

    /// A non-null literal of exactly `column`'s type. Float NaN and zeros are not used.
    fn literal(&self, column: &Column, expr: &Arc<dyn PhysicalExpr>) -> Option<ScalarValue> {
        let value = expr.downcast_ref::<Literal>()?.value();
        if value.is_null() || &value.data_type() != self.schema.field(column.index()).data_type() {
            return None;
        }
        match value {
            ScalarValue::Float16(_) => None,
            ScalarValue::Float32(Some(v)) if v.is_nan() || *v == 0.0 => None,
            ScalarValue::Float64(Some(v)) if v.is_nan() || *v == 0.0 => None,
            value => Some(value.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{
        Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    };
    use datafusion::arrow::datatypes::{DataType, Field};
    use datafusion::physical_expr::expressions::{CastExpr, NotExpr};
    use insta::assert_snapshot;

    /// DataFusion's partitioned-join filter: one branch per build partition, ELSE false.
    #[test]
    fn hash_routed_case_unions_ranges_and_value_sets() {
        let filter = hash_case(
            vec![
                partition_filter("id", 10, 20, &[10, 15, 20]),
                partition_filter("id", 31, 35, &[31, 35]),
            ],
            lit(false),
        );
        assert_snapshot!(show(&filter, 150), @"id@0 >= 10 AND id@0 <= 35 AND id@0 IN (SET) ([10, 15, 20, 31, 35])");
    }

    #[test]
    fn union_over_the_cap_keeps_only_the_range() {
        let filter = hash_case(
            vec![
                partition_filter("id", 10, 20, &[10, 15, 20]),
                partition_filter("id", 31, 35, &[31, 35]),
            ],
            lit(false),
        );
        assert_snapshot!(show(&filter, 4), @"id@0 >= 10 AND id@0 <= 35");
    }

    /// A build partition too large for an IN list probes a hash table instead, which can't be
    /// described: that branch has bounds only, so the union keeps only the range.
    #[test]
    fn branch_without_a_value_set_keeps_only_the_range() {
        let lookup_branch = binary(bounds("id", 100, 900), Operator::And, hash_lookup());
        let filter = hash_case(
            vec![partition_filter("id", 10, 20, &[10, 20]), lookup_branch],
            lit(false),
        );
        assert_snapshot!(show(&filter, 150), @"id@0 >= 10 AND id@0 <= 900");
    }

    /// An ELSE true (a canceled build partition) lets every row through.
    #[test]
    fn else_true_constrains_nothing() {
        let filter = hash_case(vec![partition_filter("id", 10, 20, &[10, 20])], lit(true));
        assert_snapshot!(show(&filter, 150), @"true");
    }

    /// Range partitioning: the last partition's filter is the ELSE.
    #[test]
    fn range_routed_case_unions_its_branches_and_else() {
        let filter: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                Some(binary(col("id"), Operator::Divide, lit(100i64))),
                vec![(lit(0i64), partition_filter("id", 1, 5, &[1, 5]))],
                Some(partition_filter("id", 150, 160, &[150, 160])),
            )
            .unwrap(),
        );
        assert_snapshot!(show(&filter, 150), @"id@0 >= 1 AND id@0 <= 160 AND id@0 IN (SET) ([1, 5, 150, 160])");
    }

    /// In a distributed query the coordinator ORs together the filter each producer task built,
    /// each with its own routing CASE.
    #[test]
    fn merged_per_task_cases_union_into_one_domain() {
        let task = |min, max| {
            hash_case(
                vec![partition_filter("id", min, max, &[min, max]), lit(false)],
                lit(false),
            )
        };
        let merged = binary(task(10, 20), Operator::Or, task(31, 35));
        assert_snapshot!(show(&merged, 150), @"id@0 >= 10 AND id@0 <= 35 AND id@0 IN (SET) ([10, 20, 31, 35])");
    }

    /// A multi-column join key: each column is constrained on its own, which is wider than the
    /// tuples but holds them all.
    #[test]
    fn multi_column_keys_are_constrained_per_column() {
        let branch = |id: (i64, i64), small: (i32, i32)| {
            let small_bounds = binary(
                binary(col("small"), Operator::GtEq, lit(small.0)),
                Operator::And,
                binary(col("small"), Operator::LtEq, lit(small.1)),
            );
            binary(bounds("id", id.0, id.1), Operator::And, small_bounds)
        };
        let filter = hash_case(
            vec![branch((1, 5), (7, 9)), branch((3, 8), (2, 4))],
            lit(false),
        );
        assert_snapshot!(show(&filter, 150), @"id@0 >= 1 AND id@0 <= 8 AND small@1 >= 2 AND small@1 <= 9");
    }

    /// A null-aware join ORs `key IS NULL` around its filter.
    #[test]
    fn null_admitting_filter_keeps_nulls() {
        let routing = hash_case(
            vec![
                partition_filter("id", 10, 20, &[10, 20]),
                partition_filter("id", 31, 35, &[31, 35]),
            ],
            lit(false),
        );
        let filter = binary(routing, Operator::Or, Arc::new(IsNullExpr::new(col("id"))));
        assert_snapshot!(show(&filter, 150), @"id@0 >= 10 AND id@0 <= 35 AND id@0 IN (SET) ([10, 20, 31, 35]) OR id@0 IS NULL");
    }

    /// AND intersects: bounds narrow the value set, and the range follows the set.
    #[test]
    fn and_intersects_ranges_and_value_sets() {
        let filter = binary(
            binary(bounds("id", 0, 30), Operator::And, bounds("id", 12, 100)),
            Operator::And,
            in_list("id", &[5, 12, 18, 25, 40]),
        );
        assert_snapshot!(show(&filter, 150), @"id@0 >= 12 AND id@0 <= 25 AND id@0 IN (SET) ([12, 18, 25])");
        let disjoint = binary(bounds("id", 0, 10), Operator::And, bounds("id", 20, 30));
        assert_snapshot!(show(&disjoint, 150), @"false");
    }

    /// Hash lookups, casts and NOT can't be described, so they are dropped from an AND.
    #[test]
    fn unconvertible_conjuncts_are_dropped() {
        let cast_eq = binary(
            Arc::new(CastExpr::new(col("id"), DataType::Int32, None)),
            Operator::Eq,
            lit(1i32),
        );
        let not_small = Arc::new(NotExpr::new(binary(col("small"), Operator::Gt, lit(3i32))));
        let filter = binary(
            binary(bounds("id", 10, 20), Operator::And, cast_eq),
            Operator::And,
            binary(hash_lookup(), Operator::And, not_small),
        );
        assert_snapshot!(show(&filter, 150), @"id@0 >= 10 AND id@0 <= 20");
    }

    /// Literals of another type and float NaN and zeros constrain nothing.
    /// An IN list with one such item is not used at all, since dropping the item would narrow it.
    #[test]
    fn unusable_literals_constrain_nothing() {
        let filter = binary(
            binary(col("small"), Operator::Gt, lit(3i64)),
            Operator::And,
            binary(
                binary(col("price"), Operator::LtEq, lit(0.0f64)),
                Operator::And,
                Arc::new(
                    InListExpr::try_new(
                        col("price"),
                        vec![lit(1.5f64), lit(f64::NAN)],
                        false,
                        &schema(),
                    )
                    .unwrap(),
                ),
            ),
        );
        assert_snapshot!(show(&filter, 150), @"true");
    }

    /// NaN, 0.0 and -0.0 compare differently depending on how NaN and the sign of zero are
    /// ordered, so a bound on any of them constrains nothing; other float bounds are kept.
    #[test]
    fn float_nan_and_zero_literals_constrain_nothing() {
        let price = |op, value: f64| binary(col("price"), op, lit(value));
        let filter = binary(
            binary(
                price(Operator::GtEq, f64::NAN),
                Operator::And,
                price(Operator::GtEq, -0.0),
            ),
            Operator::And,
            binary(
                price(Operator::LtEq, 0.0),
                Operator::And,
                price(Operator::LtEq, 9.5),
            ),
        );
        assert_snapshot!(show(&filter, 150), @"price@2 <= 9.5");
    }

    /// TopK replaces `price < 1.0` with the tighter `price < 0.0`, but a zero bound constrains
    /// nothing, so the compacted value loosens. Consumers must guard against this.
    #[test]
    fn tighter_float_bound_at_zero_compacts_looser() {
        let earlier = binary(col("price"), Operator::Lt, lit(1.0f64));
        let later = binary(col("price"), Operator::Lt, lit(0.0f64));
        assert_snapshot!(show(&earlier, 150), @"price@2 <= 1");
        assert_snapshot!(show(&later, 150), @"true");
    }

    /// The hash join's filter goes from `true` to its final value, and then an intersecting
    /// merge narrows it further. Each compacted value must keep a subset of the rows the
    /// previous one kept.
    #[test]
    fn compaction_of_tightening_filters_never_loosens() {
        let ids: Vec<i64> = (0..60).collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema()),
            vec![
                Arc::new(Int64Array::from_iter(
                    ids.iter().copied().map(Some).chain([None]),
                )),
                Arc::new(Int32Array::from(vec![Some(0); 61])),
                Arc::new(Float64Array::from(vec![Some(1.0); 61])),
            ],
        )
        .unwrap();
        let routed = hash_case(
            vec![
                partition_filter("id", 10, 20, &[10, 15, 20]),
                partition_filter("id", 31, 35, &[31, 35]),
            ],
            lit(false),
        );
        let narrowed = binary(Arc::clone(&routed), Operator::And, bounds("id", 12, 40));
        let narrowest = binary(
            Arc::clone(&narrowed),
            Operator::And,
            in_list("id", &[15, 35, 50]),
        );
        let mut kept_ids = Vec::new();
        let mut previous: Option<Vec<Option<i64>>> = None;
        for filter in [lit(true), routed, narrowed, narrowest] {
            let compacted = compact(&filter, &schema(), 4);
            let mask = compacted.evaluate(&batch).unwrap().into_array(61).unwrap();
            let mask = mask.as_any().downcast_ref::<BooleanArray>().unwrap();
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let kept: Vec<Option<i64>> = (0..61)
                .filter(|i| mask.is_valid(*i) && mask.value(*i))
                .map(|i| ids.is_valid(i).then(|| ids.value(i)))
                .collect();
            if let Some(previous) = &previous {
                assert!(
                    kept.iter().all(|row| previous.contains(row)),
                    "{compacted} loosened"
                );
            }
            kept_ids.push(format!("{compacted} keeps {} rows", kept.len()));
            previous = Some(kept);
        }
        assert_snapshot!(kept_ids.join("\n"), @r"
        true keeps 61 rows
        id@0 >= 10 AND id@0 <= 35 keeps 26 rows
        id@0 >= 12 AND id@0 <= 35 keeps 24 rows
        id@0 >= 15 AND id@0 <= 35 AND id@0 IN (SET) ([15, 35]) keeps 2 rows
        ");
    }

    #[test]
    fn dynamic_filters_under_not_or_or_are_not_selected() {
        let dynamic: Arc<dyn PhysicalExpr> = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col("id")],
            bounds("id", 1, 2),
        ));
        let under_not: Arc<dyn PhysicalExpr> = Arc::new(NotExpr::new(Arc::clone(&dynamic)));
        let under_or = binary(Arc::clone(&dynamic), Operator::Or, lit(true));
        let schema = Arc::new(schema());
        assert!(CompactedDynamicFilters::try_new(&[under_not, under_or], &schema, 150).is_none());
        let kept = binary(Arc::clone(&dynamic), Operator::And, bounds("small", 0, 1));
        assert!(CompactedDynamicFilters::try_new(&[kept], &schema, 150).is_some());
    }

    #[test]
    fn compacts_once_per_generation() {
        let schema = Arc::new(schema());
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(vec![col("id")], lit(true)));
        let compacted =
            CompactedDynamicFilters::try_new(&[Arc::clone(&dynamic) as _], &schema, 150).unwrap();
        let (_, first) = compacted.current();
        assert!(Arc::ptr_eq(&first, &compacted.current().1));
        dynamic.update(bounds("id", 1, 2)).unwrap();
        let (_, second) = compacted.current();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_snapshot!(second, @"id@0 >= 1 AND id@0 <= 2");
    }

    /// A caller that waits on the cache while the filter is updated and recompacted by another
    /// caller must reuse that newer expression, not overwrite it with an older generation's.
    #[test]
    fn caller_delayed_on_the_cache_reuses_the_newer_expression() {
        let schema = Arc::new(schema());
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(vec![col("id")], lit(true)));
        let compacted = Arc::new(
            CompactedDynamicFilters::try_new(&[Arc::clone(&dynamic) as _], &schema, 150).unwrap(),
        );

        let held = compacted.cached.lock().unwrap();
        let delayed = {
            let compacted = Arc::clone(&compacted);
            std::thread::spawn(move || compacted.current())
        };
        std::thread::sleep(std::time::Duration::from_millis(200));
        // While the delayed caller waits, the filter is updated and another caller caches it.
        dynamic.update(bounds("id", 1, 2)).unwrap();
        let mut held = held;
        let newer = compact(&dynamic.current().unwrap(), &schema, 150);
        *held = Some(Published {
            generation: compacted.generation(),
            domain: Domain::all(),
            expr: Arc::clone(&newer),
        });
        drop(held);

        let (generation, expr) = delayed.join().unwrap();
        assert_eq!(generation, compacted.generation());
        assert!(Arc::ptr_eq(&expr, &newer));
        assert_eq!(
            compacted
                .cached
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .generation,
            compacted.generation()
        );
    }

    /// Feeds `values` to one dynamic filter in turn and returns what consumers see after each.
    fn published_after(values: Vec<Arc<dyn PhysicalExpr>>, max_values: usize) -> String {
        let schema = Arc::new(schema());
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col("id"), col("price")],
            lit(true),
        ));
        let compacted =
            CompactedDynamicFilters::try_new(&[Arc::clone(&dynamic) as _], &schema, max_values)
                .unwrap();
        values
            .into_iter()
            .map(|value| {
                dynamic.update(value).unwrap();
                compacted.current().1.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// TopK's later `price < 0.0` compacts to `true` on its own; the latch keeps `price <= 1`.
    #[test]
    fn latch_keeps_float_bound_when_tighter_bound_reaches_zero() {
        let published = published_after(
            vec![
                binary(col("price"), Operator::Lt, lit(1.0f64)),
                binary(col("price"), Operator::Lt, lit(0.0f64)),
            ],
            150,
        );
        assert_snapshot!(published, @"
        price@2 <= 1
        price@2 <= 1
        ");
    }

    /// A value set that crosses the cap becomes a range on its own; the latch keeps the set.
    #[test]
    fn latch_keeps_value_set_when_a_union_crosses_the_cap() {
        let published = published_after(
            vec![
                in_list("id", &[1, 5, 9]),
                // The same rows, but the union of its branches' sets crosses the cap.
                binary(
                    binary(
                        in_list("id", &[1, 5]),
                        Operator::Or,
                        in_list("id", &[9, 20]),
                    ),
                    Operator::And,
                    bounds("id", 1, 9),
                ),
            ],
            3,
        );
        assert_snapshot!(published, @"
        id@0 >= 1 AND id@0 <= 9 AND id@0 IN (SET) ([1, 5, 9])
        id@0 >= 1 AND id@0 <= 9 AND id@0 IN (SET) ([1, 5, 9])
        ");
    }

    /// A column constrained earlier but no longer constrained keeps its earlier constraint.
    #[test]
    fn latch_keeps_constraints_on_columns_a_later_value_drops() {
        let published = published_after(
            vec![
                binary(
                    bounds("id", 1, 9),
                    Operator::And,
                    binary(col("price"), Operator::Lt, lit(5.0f64)),
                ),
                bounds("id", 2, 4),
            ],
            150,
        );
        assert_snapshot!(published, @"
        id@0 >= 1 AND id@0 <= 9 AND price@2 <= 5
        id@0 >= 2 AND id@0 <= 4 AND price@2 <= 5
        ");
    }

    /// Once nothing can pass, nothing passes for every later value.
    #[test]
    fn latch_stays_false_once_a_value_compacts_to_false() {
        let published = published_after(vec![lit(false), bounds("id", 1, 9), lit(true)], 150);
        assert_snapshot!(published, @"
        false
        false
        false
        ");
    }

    fn show(expr: &Arc<dyn PhysicalExpr>, max_values: usize) -> String {
        compact(expr, &schema(), max_values).to_string()
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("small", DataType::Int32, true),
            Field::new("price", DataType::Float64, true),
        ])
    }

    fn col(name: &str) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, schema().index_of(name).unwrap()))
    }

    fn binary(
        l: Arc<dyn PhysicalExpr>,
        op: Operator,
        r: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(l, op, r))
    }

    fn bounds(column: &str, min: i64, max: i64) -> Arc<dyn PhysicalExpr> {
        binary(
            binary(col(column), Operator::GtEq, lit(min)),
            Operator::And,
            binary(col(column), Operator::LtEq, lit(max)),
        )
    }

    fn in_list(column: &str, values: &[i64]) -> Arc<dyn PhysicalExpr> {
        let list = values.iter().map(|v| lit(*v)).collect();
        Arc::new(InListExpr::try_new(col(column), list, false, &schema()).unwrap())
    }

    /// One build partition's filter: bounds AND IN-list membership.
    fn partition_filter(column: &str, min: i64, max: i64, values: &[i64]) -> Arc<dyn PhysicalExpr> {
        binary(
            bounds(column, min, max),
            Operator::And,
            in_list(column, values),
        )
    }

    /// `CASE id % N WHEN 0 THEN branches[0] ... ELSE else_expr END`.
    fn hash_case(
        branches: Vec<Arc<dyn PhysicalExpr>>,
        else_expr: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn PhysicalExpr> {
        let routing = binary(col("id"), Operator::Modulo, lit(branches.len() as i64));
        let when_then = branches
            .into_iter()
            .enumerate()
            .map(|(i, then)| (lit(i as i64), then))
            .collect();
        Arc::new(CaseExpr::try_new(Some(routing), when_then, Some(else_expr)).unwrap())
    }

    /// Stands in for a hash table lookup, which can't be described: `CAST(id AS INT) = 1`.
    fn hash_lookup() -> Arc<dyn PhysicalExpr> {
        binary(
            Arc::new(CastExpr::new(col("id"), DataType::Int32, None)),
            Operator::Eq,
            lit(1i32),
        )
    }
}
