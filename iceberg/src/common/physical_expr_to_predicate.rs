//! Conversion of DataFusion physical filter expressions (notably dynamic filters pushed down
//! from hash joins) into Iceberg [`Predicate`]s usable for row-group, page and row pruning.
//!
//! The conversion is conservative: the returned predicate matches every row the physical
//! expression could keep, and may match more. `None` means the expression can't be used to
//! prune.

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
use iceberg::spec::{Datum, Schema, Type};

/// Converts `expr`, evaluated against `input_schema`, into an Iceberg [`Predicate`] over the
/// fields of `schema`.
///
/// The predicate matches a superset of the rows for which `expr` evaluates to `true`.
/// Returns `None` when no part of `expr` can be expressed as a predicate.
#[cfg_attr(not(test), allow(dead_code))]
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

/// A converted subtree. `exact` is true when `predicate` matches exactly the rows the subtree
/// evaluates to `true` (with the same NULL semantics), and false when it over-approximates.
struct Schemas<'a> {
    input: &'a ArrowSchema,
    iceberg: &'a Schema,
}

struct Converted {
    predicate: Predicate,
    exact: bool,
}

impl Converted {
    fn exact(predicate: Predicate) -> Self {
        Self {
            predicate,
            exact: true,
        }
    }
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
            .exact
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
                    predicate: l.predicate.and(r.predicate),
                }),
                // Dropping a conjunct only widens the predicate.
                (Some(c), None) | (None, Some(c)) => Some(Converted {
                    predicate: c.predicate,
                    exact: false,
                }),
                (None, None) => None,
            }
        }
        Operator::Or => {
            let left = convert(binary.left(), schema)?;
            let right = convert(binary.right(), schema)?;
            (left.exact && right.exact)
                .then(|| Converted::exact(left.predicate.or(right.predicate)))
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
        Operator::Eq => reference.equal_to(datum),
        Operator::NotEq => reference.not_equal_to(datum),
        Operator::Lt => reference.less_than(datum),
        Operator::LtEq => reference.less_than_or_equal_to(datum),
        Operator::Gt => reference.greater_than(datum),
        Operator::GtEq => reference.greater_than_or_equal_to(datum),
        _ => return None,
    };
    Some(Converted::exact(predicate))
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
        (false, false) => reference.is_in(datums),
        (true, true) => return None,
        (true, false) => reference.is_not_in(datums),
    };
    Some(Converted::exact(predicate))
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
        ScalarValue::TimestampNanosecond(Some(v), Some(_)) => Datum::timestamptz_nanos(*v),
        ScalarValue::TimestampNanosecond(Some(v), None) => Datum::timestamp_nanos(*v),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use datafusion::physical_expr::expressions::{CaseExpr, CastExpr, lit};
    use iceberg::spec::{NestedField, PrimitiveType};
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
        price > 1.5
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
    fn or_requires_every_child_exact() {
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
        assert!(convert_expr(&approximate, &schema).is_none());
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
        // Shape built by DataFusion's hash join: per-partition CASE routing over a hash, plus
        // min/max bounds AND IN-list membership. The CASE is dropped, the rest converts.
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
