//! Conversion of DataFusion physical filter expressions (notably dynamic filters pushed down
//! from hash joins) into Iceberg [`Predicate`]s usable for row-group, page and row pruning.
//!
//! The conversion is conservative: the returned predicate matches every row the physical
//! expression could keep, and may match more. `None` means the expression can't be used to
//! prune.
//!
//! Every converted subtree is tracked as exact or approximate (over-approximating):
//! * AND keeps the children it can convert; if it drops one, the result is approximate.
//! * OR converts only when every child converts, exact or approximate; an OR of
//!   over-approximations still over-approximates. The result is approximate if any child is.
//! * NOT, and negated IN, convert only exact children: negating an over-approximation would
//!   under-approximate.
//!
//! Float columns: DataFusion compares floats with a total order in which NaN sorts above every
//! value and negative NaN below every value, so `price > 1.5`, `price < 1.5`, `price != 1.5`
//! and `price NOT IN (..)` all keep NaN rows. Iceberg's Parquet row-group and page pruning
//! uses min/max statistics that exclude NaN, so the plain Iceberg comparison could skip a row
//! group holding `[1.0, NaN]` under `price > 1.5`. Those shapes are therefore converted as
//! `(<comparison>) OR is_nan(price)` and marked approximate. Every evaluator the reader uses
//! treats `is_nan` conservatively: the row-group metrics and bloom filter evaluators answer
//! "might match", the page index evaluator selects every row of a float column, and the row
//! filter evaluates it exactly. `=` and IN with non-NaN literals never keep NaN rows and stay
//! exact, but a float comparison is never negated: `NOT (price = 1.5)` keeps NaN rows too.
//! NaN and ±0 float literals are not converted at all, matching the float rule of
//! [`crate::common::dynamic_domain`] and file skipping by partition values.
//!
//! Not converted: decimal and nanosecond timestamp literals, and anything that isn't a bare
//! column compared with a literal of exactly the column's Iceberg type.
//!
//! Dynamic filters are compacted before they get here, see
//! [`crate::common::dynamic_domain`], which is where a partitioned hash join's routing `CASE`
//! is handled.

use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr, Literal,
    NotExpr,
};
use iceberg::expr::{Predicate, Reference};

use iceberg::spec::{Datum, PrimitiveType, Schema, Type};

/// Converts `expr`, evaluated against `input_schema`, into an Iceberg [`Predicate`] over the
/// fields of `schema`.
///
/// The predicate matches a superset of the rows for which `expr` evaluates to `true`.
/// Returns `None` when no part of `expr` can be expressed as a predicate.
pub(crate) fn physical_expr_to_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &ArrowSchema,
    schema: &Schema,
) -> Option<Predicate> {
    let schemas = Schemas {
        input: input_schema,
        iceberg: schema,
    };
    convert(expr, &schemas).map(|c| c.predicate)
}

struct Schemas<'a> {
    input: &'a ArrowSchema,
    iceberg: &'a Schema,
}

/// A converted subtree. `exact` is true when `predicate` matches exactly the rows the subtree
/// evaluates to `true` (with the same NULL semantics), and false when it over-approximates.
/// `negatable` is true when negating `predicate` matches exactly the rows the negated subtree
/// keeps; it implies `exact`, and is false for float comparisons (see the module docs).
struct Converted {
    predicate: Predicate,
    exact: bool,
    negatable: bool,
}

impl Converted {
    fn exact(predicate: Predicate) -> Self {
        Self {
            predicate,
            exact: true,
            negatable: true,
        }
    }

    fn approximate(predicate: Predicate) -> Self {
        Self {
            predicate,
            exact: false,
            negatable: false,
        }
    }

    /// An exact comparison on `column`. On float columns it is not negatable, and shapes that
    /// keep NaN rows (`nan_kept`) also match NaN and become approximate.
    fn comparison(predicate: Predicate, column: &Reference, ty: &Type, nan_kept: bool) -> Self {
        match (is_float(ty), nan_kept) {
            (false, _) => Self::exact(predicate),
            (true, false) => Self {
                predicate,
                exact: true,
                negatable: false,
            },
            (true, true) => Self::approximate(predicate.or(column.clone().is_nan())),
        }
    }
}

fn is_float(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Primitive(PrimitiveType::Float | PrimitiveType::Double)
    )
}

fn convert(expr: &Arc<dyn PhysicalExpr>, schema: &Schemas) -> Option<Converted> {
    if let Some(dynamic) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        return convert(&dynamic.current().ok()?, schema);
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return convert_binary(binary, schema);
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        // NOT of an over-approximation would under-approximate, so only exact children.
        let child = convert(not.arg(), schema)?;
        return child
            .negatable
            .then(|| Converted::exact(child.predicate.negate()));
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return Some(Converted::exact(
            column_ref(is_null.arg(), schema)?.0.is_null(),
        ));
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return Some(Converted::exact(
            column_ref(is_not_null.arg(), schema)?.0.is_not_null(),
        ));
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        return convert_in_list(in_list, schema);
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return match literal.value() {
            ScalarValue::Boolean(Some(true)) => Some(Converted::exact(Predicate::AlwaysTrue)),
            ScalarValue::Boolean(Some(false)) => Some(Converted::exact(Predicate::AlwaysFalse)),
            _ => None,
        };
    }
    None
}

fn convert_binary(binary: &BinaryExpr, schema: &Schemas) -> Option<Converted> {
    match binary.op() {
        Operator::And => {
            let left = convert(binary.left(), schema);
            let right = convert(binary.right(), schema);
            match (left, right) {
                (Some(l), Some(r)) => Some(Converted {
                    exact: l.exact && r.exact,
                    negatable: l.negatable && r.negatable,
                    predicate: l.predicate.and(r.predicate),
                }),
                // Dropping a conjunct only widens the predicate.
                (Some(c), None) | (None, Some(c)) => Some(Converted::approximate(c.predicate)),
                (None, None) => None,
            }
        }
        Operator::Or => {
            let left = convert(binary.left(), schema)?;
            let right = convert(binary.right(), schema)?;
            Some(Converted {
                exact: left.exact && right.exact,
                negatable: left.negatable && right.negatable,
                predicate: left.predicate.or(right.predicate),
            })
        }
        op => convert_comparison(binary.left(), *op, binary.right(), schema),
    }
}

fn convert_comparison(
    left: &Arc<dyn PhysicalExpr>,
    op: Operator,
    right: &Arc<dyn PhysicalExpr>,
    schema: &Schemas,
) -> Option<Converted> {
    let (column, literal, op) = match (column_ref(left, schema), column_ref(right, schema)) {
        (Some(column), None) => (column, right, op),
        (None, Some(column)) => (column, left, op.swap()?),
        _ => return None,
    };
    let (reference, field_type) = column;
    let datum = literal_datum(literal, field_type)?;
    let predicate = match op {
        Operator::Eq => reference.clone().equal_to(datum),
        Operator::NotEq => reference.clone().not_equal_to(datum),
        Operator::Lt => reference.clone().less_than(datum),
        Operator::LtEq => reference.clone().less_than_or_equal_to(datum),
        Operator::Gt => reference.clone().greater_than(datum),
        Operator::GtEq => reference.clone().greater_than_or_equal_to(datum),
        _ => return None,
    };
    let nan_kept = op != Operator::Eq;
    Some(Converted::comparison(
        predicate, &reference, field_type, nan_kept,
    ))
}

fn convert_in_list(in_list: &InListExpr, schema: &Schemas) -> Option<Converted> {
    let (reference, field_type) = column_ref(in_list.expr(), schema)?;
    let mut datums = Vec::with_capacity(in_list.list().len());
    for item in in_list.list() {
        let literal = item.downcast_ref::<Literal>()?;
        if literal.value().is_null() {
            // `x NOT IN (.., NULL)` is never true; leave that to DataFusion. In the
            // non-negated form a NULL item can never match, so it is simply skipped.
            if in_list.negated() {
                return None;
            }
            continue;
        }
        datums.push(literal_datum(item, field_type)?);
    }
    let predicate = match (in_list.negated(), datums.is_empty()) {
        (false, true) => Predicate::AlwaysFalse,
        (false, false) => reference.clone().is_in(datums),
        (true, true) => return None,
        (true, false) => reference.clone().is_not_in(datums),
    };
    Some(Converted::comparison(
        predicate,
        &reference,
        field_type,
        in_list.negated(),
    ))
}

/// Resolves a bare column to an Iceberg reference and its field type. Columns not present in
/// the Iceberg schema, and anything that is not a bare column (casts, functions, ...), don't
/// resolve.
fn column_ref<'a>(
    expr: &Arc<dyn PhysicalExpr>,
    schemas: &Schemas<'a>,
) -> Option<(Reference, &'a Type)> {
    let column = expr.downcast_ref::<Column>()?;
    // The column's index is what DataFusion evaluates; its name must agree with the input field
    // at that index, otherwise the name can't be trusted to identify the Iceberg field.
    let input_field = schemas.input.fields().get(column.index())?;
    if input_field.name() != column.name() {
        return None;
    }
    let field = schemas.iceberg.field_by_name(column.name())?;
    Some((Reference::new(column.name()), field.field_type.as_ref()))
}

/// Converts a non-null literal into a [`Datum`] whose type is exactly `field_type`. No lossy
/// conversion is attempted: a literal of a different type doesn't convert.
fn literal_datum(expr: &Arc<dyn PhysicalExpr>, field_type: &Type) -> Option<Datum> {
    let datum = scalar_value_to_datum(expr.downcast_ref::<Literal>()?.value())?;
    match field_type {
        Type::Primitive(primitive) if datum.data_type() == primitive => Some(datum),
        _ => None,
    }
}

fn scalar_value_to_datum(value: &ScalarValue) -> Option<Datum> {
    const MICROS_PER_SECOND: i64 = 1_000_000;
    const MICROS_PER_MILLI: i64 = 1_000;
    let timestamp = |micros: i64, tz: &Option<Arc<str>>| match tz {
        Some(_) => Datum::timestamptz_micros(micros),
        None => Datum::timestamp_micros(micros),
    };
    Some(match value {
        ScalarValue::Boolean(Some(v)) => Datum::bool(*v),
        ScalarValue::Int8(Some(v)) => Datum::int(*v as i32),
        ScalarValue::Int16(Some(v)) => Datum::int(*v as i32),
        ScalarValue::Int32(Some(v)) => Datum::int(*v),
        ScalarValue::Int64(Some(v)) => Datum::long(*v),
        // NaN ordering differs between engines, and Iceberg may collapse +0.0 and -0.0 (e.g.
        // in IN sets) where DataFusion distinguishes them; don't prune on either.
        ScalarValue::Float32(Some(v)) if !v.is_nan() && *v != 0.0 => Datum::float(*v),
        ScalarValue::Float64(Some(v)) if !v.is_nan() && *v != 0.0 => Datum::double(*v),
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => Datum::string(v.clone()),
        ScalarValue::Date32(Some(v)) => Datum::date(*v),
        ScalarValue::TimestampSecond(Some(v), tz) => {
            timestamp(v.checked_mul(MICROS_PER_SECOND)?, tz)
        }
        ScalarValue::TimestampMillisecond(Some(v), tz) => {
            timestamp(v.checked_mul(MICROS_PER_MILLI)?, tz)
        }
        ScalarValue::TimestampMicrosecond(Some(v), tz) => timestamp(*v, tz),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use datafusion::physical_expr::expressions::{CaseExpr, CastExpr, lit};
    use iceberg::spec::NestedField;
    use insta::assert_snapshot;

    #[test]
    fn comparisons_with_column_on_the_left() {
        let schema = iceberg_schema();
        let preds = [
            Operator::Eq,
            Operator::NotEq,
            Operator::Lt,
            Operator::LtEq,
            Operator::Gt,
            Operator::GtEq,
        ]
        .map(|op| show(&binary(col("id"), op, lit(5i64)), &schema));
        assert_snapshot!(preds.join("\n"), @r"
        id = 5
        id != 5
        id < 5
        id <= 5
        id > 5
        id >= 5
        ");
    }

    #[test]
    fn comparisons_with_literal_on_the_left_flip_the_operator() {
        let schema = iceberg_schema();
        let preds = [
            Operator::Eq,
            Operator::NotEq,
            Operator::Lt,
            Operator::LtEq,
            Operator::Gt,
            Operator::GtEq,
        ]
        .map(|op| show(&binary(lit(5i64), op, col("id")), &schema));
        assert_snapshot!(preds.join("\n"), @r"
        id = 5
        id != 5
        id > 5
        id >= 5
        id < 5
        id <= 5
        ");
    }

    #[test]
    fn literal_types_map_to_matching_iceberg_types() {
        let schema = iceberg_schema();
        let preds = [
            binary(col("small"), Operator::Eq, lit(3i32)),
            binary(col("name"), Operator::Eq, lit("x")),
            binary(
                col("name"),
                Operator::Eq,
                lit(ScalarValue::LargeUtf8(Some("y".into()))),
            ),
            binary(
                col("name"),
                Operator::Eq,
                lit(ScalarValue::Utf8View(Some("z".into()))),
            ),
            binary(col("flag"), Operator::Eq, lit(true)),
            binary(col("price"), Operator::Gt, lit(1.5f64)),
            binary(
                col("day"),
                Operator::Lt,
                lit(ScalarValue::Date32(Some(19000))),
            ),
            binary(
                col("ts"),
                Operator::GtEq,
                lit(ScalarValue::TimestampMillisecond(Some(1_000), None)),
            ),
        ]
        .map(|e| show(&e, &schema));
        assert_snapshot!(preds.join("\n"), @r#"
        small = 3
        name = "x"
        name = "y"
        name = "z"
        flag = true
        (price > 1.5) OR (price IS NAN)
        day < 2022-01-08
        ts >= 1970-01-01 00:00:01
        "#);
    }

    #[test]
    fn mismatched_or_unsupported_literal_types_do_not_convert() {
        let schema = iceberg_schema();
        // Int32 literal against a long column: no lossy coercion.
        assert!(convert_expr(&binary(col("id"), Operator::Eq, lit(5i32)), &schema).is_none());
        // Binary literal type unsupported.
        let bin = lit(ScalarValue::Binary(Some(vec![1])));
        assert!(convert_expr(&binary(col("name"), Operator::Eq, bin), &schema).is_none());
        // NULL literal.
        let null = lit(ScalarValue::Int64(None));
        assert!(convert_expr(&binary(col("id"), Operator::Eq, null), &schema).is_none());
        // NaN.
        let nan = lit(f64::NAN);
        assert!(convert_expr(&binary(col("price"), Operator::Lt, nan), &schema).is_none());
    }

    #[test]
    fn unknown_columns_and_non_column_operands_do_not_convert() {
        let schema = iceberg_schema();
        assert!(convert_expr(&binary(col("missing"), Operator::Eq, lit(1i64)), &schema).is_none());
        assert!(convert_expr(&binary(col("id"), Operator::Eq, col("small")), &schema).is_none());
        assert!(convert_expr(&binary(cast_id(), Operator::Eq, lit(1i32)), &schema).is_none());
        assert!(convert_expr(&binary(col("id"), Operator::Plus, lit(1i64)), &schema).is_none());
    }

    #[test]
    fn and_drops_unconvertible_children() {
        let schema = iceberg_schema();
        let both = binary(gt_id(1), Operator::And, lt_id(9));
        assert_snapshot!(show(&both, &schema), @"(id > 1) AND (id < 9)");
        let one = binary(gt_id(1), Operator::And, cast_eq());
        assert_snapshot!(show(&one, &schema), @"id > 1");
        let none = binary(cast_eq(), Operator::And, cast_eq());
        assert!(convert_expr(&none, &schema).is_none());
    }

    #[test]
    fn or_requires_every_child_to_convert() {
        let schema = iceberg_schema();
        let exact = binary(lt_id(1), Operator::Or, gt_id(9));
        assert_snapshot!(show(&exact, &schema), @"(id < 1) OR (id > 9)");
        let unconvertible = binary(lt_id(1), Operator::Or, cast_eq());
        assert!(convert_expr(&unconvertible, &schema).is_none());
        let approximate = binary(
            lt_id(1),
            Operator::Or,
            binary(gt_id(9), Operator::And, cast_eq()),
        );
        assert_snapshot!(show(&approximate, &schema), @"(id < 1) OR (id > 9)");
    }

    #[test]
    fn not_of_an_approximate_or_does_not_convert() {
        let schema = iceberg_schema();
        let approximate_or = binary(
            lt_id(1),
            Operator::Or,
            binary(gt_id(9), Operator::And, cast_eq()),
        );
        assert!(convert_expr(&not(approximate_or), &schema).is_none());
    }

    #[test]
    fn not_requires_an_exact_child() {
        let schema = iceberg_schema();
        let exact = not(binary(gt_id(1), Operator::And, lt_id(9)));
        assert_snapshot!(show(&exact, &schema), @"(id <= 1) OR (id >= 9)");
        let approximate = not(binary(gt_id(1), Operator::And, cast_eq()));
        assert!(convert_expr(&approximate, &schema).is_none());
    }

    #[test]
    fn null_checks() {
        let schema = iceberg_schema();
        let is_null: Arc<dyn PhysicalExpr> = Arc::new(IsNullExpr::new(col("name")));
        let is_not_null: Arc<dyn PhysicalExpr> = Arc::new(IsNotNullExpr::new(col("name")));
        assert_snapshot!(show(&is_null, &schema), @"name IS NULL");
        assert_snapshot!(show(&is_not_null, &schema), @"name IS NOT NULL");
    }

    #[test]
    fn in_list_converts_to_is_in_and_not_in() {
        let schema = iceberg_schema();
        let values = [lit(1i64), lit(ScalarValue::Int64(None)), lit(3i64)];
        assert_snapshot!(show(&in_list(&values, false), &schema), @"id IN (1, 3)");
        let values = [lit(1i64), lit(3i64)];
        assert_snapshot!(show(&in_list(&values, true), &schema), @"id NOT IN (1, 3)");
    }

    #[test]
    fn negated_in_list_only_when_exact() {
        let schema = iceberg_schema();
        // `NOT IN (.., NULL)` never matches; not representable exactly here.
        let with_null = [lit(1i64), lit(ScalarValue::Int64(None))];
        assert!(convert_expr(&in_list(&with_null, true), &schema).is_none());
        // A non-literal item.
        let with_bad_type = [lit(1i64), binary(col("id"), Operator::Plus, lit(1i64))];
        assert!(convert_expr(&in_list(&with_bad_type, true), &schema).is_none());
        assert!(convert_expr(&in_list(&with_bad_type, false), &schema).is_none());
    }

    #[test]
    fn boolean_literals() {
        let schema = iceberg_schema();
        assert_snapshot!(show(&lit(true), &schema), @"TRUE");
        assert_snapshot!(show(&lit(false), &schema), @"FALSE");
        assert!(convert_expr(&lit(1i64), &schema).is_none());
    }

    #[test]
    fn dynamic_filter_converts_its_current_value() {
        let schema = iceberg_schema();
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(vec![col("id")], lit(true)));
        let outer = binary(
            binary(col("small"), Operator::Gt, lit(0i32)),
            Operator::And,
            Arc::clone(&dynamic) as _,
        );
        assert_snapshot!(show(&outer, &schema), @"small > 0");
        dynamic
            .update(binary(gt_id(1), Operator::And, lt_id(9)))
            .unwrap();
        assert_snapshot!(show(&outer, &schema), @"(small > 0) AND ((id > 1) AND (id < 9))");
        // The snapshot's exactness propagates: an approximate current value under NOT fails.
        dynamic
            .update(binary(gt_id(1), Operator::And, cast_eq()))
            .unwrap();
        assert!(convert_expr(&not(Arc::clone(&dynamic) as _), &schema).is_none());
    }

    #[test]
    fn hash_join_style_dynamic_filter() {
        let schema = iceberg_schema();
        // Min/max bounds AND IN-list membership AND an unconvertible routing CASE, which is dropped.
        let bounds = binary(
            binary(col("id"), Operator::GtEq, lit(10i64)),
            Operator::And,
            binary(col("id"), Operator::LtEq, lit(20i64)),
        );
        let membership = in_list(&[lit(10i64), lit(15i64), lit(20i64)], false);
        let routing: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                Some(binary(col("id"), Operator::Modulo, lit(4i64))),
                vec![(lit(0i64), lit(true))],
                Some(lit(false)),
            )
            .unwrap(),
        );
        let filter = binary(
            binary(bounds, Operator::And, membership),
            Operator::And,
            routing,
        );
        let dynamic: Arc<dyn PhysicalExpr> =
            Arc::new(DynamicFilterPhysicalExpr::new(vec![col("id")], lit(true)));
        dynamic
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .unwrap()
            .update(filter)
            .unwrap();
        assert_snapshot!(show(&dynamic, &schema), @"((id >= 10) AND (id <= 20)) AND (id IN (20, 15, 10))");
    }

    #[test]
    fn column_name_must_match_the_input_field_at_its_index() {
        let schema = iceberg_schema();
        // Named `id` but pointing at `small` (index 1), or past the end of the input schema.
        let mismatched: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 1));
        let out_of_range: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 99));
        let results = [mismatched, out_of_range].map(|c| {
            format!(
                "{:?}",
                convert_expr(&binary(c, Operator::Eq, lit(5i64)), &schema).map(|p| p.to_string())
            )
        });
        assert_snapshot!(results.join("\n"), @"
        None
        None
        ");
    }

    #[test]
    fn float_comparisons_that_keep_nan_rows_also_match_nan() {
        let schema = iceberg_schema();
        // Under DataFusion's total order NaN > 1.5, -NaN < 1.5 and NaN != 1.5, while Parquet
        // min/max exclude NaN; each shape also matches NaN and is approximate.
        let results = [
            binary(col("price"), Operator::Gt, lit(1.5f64)),
            binary(col("price"), Operator::GtEq, lit(1.5f64)),
            binary(col("price"), Operator::Lt, lit(1.5f64)),
            binary(col("price"), Operator::LtEq, lit(1.5f64)),
            binary(col("price"), Operator::NotEq, lit(1.5f64)),
            binary(lit(1.5f64), Operator::Lt, col("price")),
            in_list_on("price", &[lit(1.5f64), lit(2.5f64)], true),
            binary(col("ratio"), Operator::Gt, lit(1.5f32)),
        ]
        .map(|e| show(&e, &schema));
        assert_snapshot!(results.join("\n"), @"
        (price > 1.5) OR (price IS NAN)
        (price >= 1.5) OR (price IS NAN)
        (price < 1.5) OR (price IS NAN)
        (price <= 1.5) OR (price IS NAN)
        (price != 1.5) OR (price IS NAN)
        (price > 1.5) OR (price IS NAN)
        (price NOT IN (1.5, 2.5)) OR (price IS NAN)
        (ratio > 1.5) OR (ratio IS NAN)
        ");
    }

    #[test]
    fn float_equality_and_in_stay_exact_but_do_not_negate() {
        let schema = iceberg_schema();
        let eq = binary(col("price"), Operator::Eq, lit(1.5f64));
        let in_set = in_list_on("price", &[lit(1.5f64), lit(2.5f64)], false);
        assert_snapshot!(show(&eq, &schema), @"price = 1.5");
        assert_snapshot!(show(&in_set, &schema), @"price IN (1.5, 2.5)");
        // Exact, so an AND with it is exact too.
        let both = binary(eq.clone(), Operator::And, gt_id(1));
        assert_snapshot!(show(&both, &schema), @"(price = 1.5) AND (id > 1)");
        // `NOT (price = 1.5)` keeps NaN rows, `NOT (price < 1.5)` keeps NaN rows and NOT of an
        // approximate OR would drop them: none of these convert.
        let results = [
            not(eq),
            not(in_set),
            not(binary(col("price"), Operator::Lt, lit(1.5f64))),
            not(binary(
                binary(col("price"), Operator::Eq, lit(1.5f64)),
                Operator::And,
                gt_id(1),
            )),
        ]
        .map(|e| format!("{:?}", convert_expr(&e, &schema).map(|p| p.to_string())));
        assert_snapshot!(results.join("\n"), @"
        None
        None
        None
        None
        ");
        // Control: the same shape on a non-float column negates exactly.
        let control = not(binary(col("id"), Operator::Lt, lit(5i64)));
        assert_snapshot!(show(&control, &schema), @"id >= 5");
    }

    #[test]
    fn unsupported_literal_types_do_not_convert() {
        let schema = iceberg_schema();
        let results = [
            binary(
                col("ts"),
                Operator::Gt,
                lit(ScalarValue::TimestampNanosecond(Some(1_000), None)),
            ),
            binary(
                col("amount"),
                Operator::Gt,
                lit(ScalarValue::Decimal128(Some(150), 10, 2)),
            ),
        ]
        .map(|e| format!("{:?}", convert_expr(&e, &schema).map(|p| p.to_string())));
        assert_snapshot!(results.join("\n"), @"
        None
        None
        ");
    }

    #[test]
    fn float_zero_literals_do_not_convert() {
        let schema = iceberg_schema();
        // DataFusion distinguishes +0.0 and -0.0; Iceberg may not, so don't prune on zeros.
        let results = [
            in_list_on("price", &[lit(-0.0f64), lit(1.5f64)], false),
            in_list_on("price", &[lit(0.0f64), lit(1.5f64)], true),
            binary(col("price"), Operator::Eq, lit(-0.0f64)),
            binary(col("price"), Operator::Gt, lit(0.0f64)),
        ]
        .map(|e| format!("{:?}", convert_expr(&e, &schema).map(|p| p.to_string())));
        assert_snapshot!(results.join("\n"), @"
        None
        None
        None
        None
        ");
    }

    fn iceberg_schema() -> Schema {
        let field = |id, name, ty| NestedField::optional(id, name, Type::Primitive(ty)).into();
        Schema::builder()
            .with_fields(vec![
                field(1, "id", PrimitiveType::Long),
                field(2, "small", PrimitiveType::Int),
                field(3, "name", PrimitiveType::String),
                field(4, "flag", PrimitiveType::Boolean),
                field(5, "price", PrimitiveType::Double),
                field(6, "day", PrimitiveType::Date),
                field(7, "ts", PrimitiveType::Timestamp),
                field(8, "ratio", PrimitiveType::Float),
                field(
                    9,
                    "amount",
                    PrimitiveType::Decimal {
                        precision: 10,
                        scale: 2,
                    },
                ),
            ])
            .build()
            .unwrap()
    }

    fn arrow_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("small", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new("price", DataType::Float64, true),
            Field::new("day", DataType::Date32, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
            Field::new("ratio", DataType::Float32, true),
            Field::new("amount", DataType::Decimal128(10, 2), true),
            // Present in the input but not in the Iceberg schema.
            Field::new("missing", DataType::Int64, true),
        ])
    }

    fn convert_expr(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Option<Predicate> {
        physical_expr_to_predicate(expr, &arrow_schema(), schema)
    }

    fn show(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> String {
        convert_expr(expr, schema)
            .expect("expected a predicate")
            .to_string()
    }

    fn col(name: &str) -> Arc<dyn PhysicalExpr> {
        let index = arrow_schema()
            .index_of(name)
            .expect("column in the input schema");
        Arc::new(Column::new(name, index))
    }

    fn binary(
        l: Arc<dyn PhysicalExpr>,
        op: Operator,
        r: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(l, op, r))
    }

    fn not(e: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(NotExpr::new(e))
    }

    fn gt_id(v: i64) -> Arc<dyn PhysicalExpr> {
        binary(col("id"), Operator::Gt, lit(v))
    }

    fn lt_id(v: i64) -> Arc<dyn PhysicalExpr> {
        binary(col("id"), Operator::Lt, lit(v))
    }

    fn cast_id() -> Arc<dyn PhysicalExpr> {
        Arc::new(CastExpr::new(col("id"), DataType::Int32, None))
    }

    /// An unconvertible leaf: `CAST(id AS INT) = 1`.
    fn cast_eq() -> Arc<dyn PhysicalExpr> {
        binary(cast_id(), Operator::Eq, lit(1i32))
    }

    fn in_list(values: &[Arc<dyn PhysicalExpr>], negated: bool) -> Arc<dyn PhysicalExpr> {
        in_list_on("id", values, negated)
    }

    fn in_list_on(
        column: &str,
        values: &[Arc<dyn PhysicalExpr>],
        negated: bool,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(
            InListExpr::try_new(col(column), values.to_vec(), negated, &arrow_schema()).unwrap(),
        )
    }
}
