use std::sync::Arc;

use datafusion::common::HashSet;
use iceberg::expr::{Bind, BoundPredicate, Predicate};
use iceberg::scan::FileScanTask;
use iceberg::spec::{PartitionSpec, SchemaRef, Transform};

use crate::common::dynamic_filters::task_filter::and_predicate;

/// The most IN list values the reader gets, counted across all the IN lists in the filter. The
/// reader checks every row once per value, so long lists are slow. Above this limit, we drop the
/// IN lists and keep only the join's min/max bounds.
const MAX_LISTED_VALUES: usize = 200;

/// Returns a copy of `task` with the parts of `predicate` the reader can use added to its filter.
/// The reader then skips the row groups those parts rule out.
///
/// `predicate` is the dynamic filters, converted to an Iceberg predicate. Returns `None` if the
/// reader can't use any part of it, or on any error.
pub(crate) fn with_dynamic_filter(
    task: &FileScanTask,
    predicate: &Predicate,
) -> Option<FileScanTask> {
    // The partition spec tells us which columns are partition columns.
    let spec = task.partition_spec()?;
    let filter = reader_filter(predicate, spec, task.schema_ref(), task.case_sensitive())?;
    // AND onto the task's existing filter, so the reader still applies the query's WHERE clause.
    and_predicate(task, filter).ok()
}

/// The parts of `predicate` the reader can check correctly for a file written with `spec`.
/// Returns `None` if no part is left.
///
/// We drop comparisons on columns the reader can get wrong:
/// - Identity partition columns. A data file doesn't have to store these, because the value is
///   in the file's partition. The reader reads a column the file doesn't store as NULL, so it
///   could skip row groups that match. The partition filter already checks these columns.
/// - Columns with a default value. Files written before the column was added don't store it, so
///   the reader reads NULL where it should use the default.
/// - Nested columns. The reader's row filter only supports top-level columns, and fails the read
///   on a nested one.
fn reader_filter(
    predicate: &Predicate,
    spec: &PartitionSpec,
    schema: SchemaRef,
    case_sensitive: bool,
) -> Option<Predicate> {
    let identity_columns: HashSet<i32> = spec
        .fields()
        .iter()
        .filter(|field| field.transform == Transform::Identity)
        .map(|field| field.source_id)
        .collect();
    let keep_lists = listed_values(predicate) <= MAX_LISTED_VALUES;
    let readable = |comparison: &Predicate| {
        if matches!(comparison, Predicate::Set(_)) && !keep_lists {
            return false;
        }
        // Bind the comparison by itself to find its column. Drop it if it doesn't bind.
        let Ok(bound) = comparison.clone().bind(Arc::clone(&schema), case_sensitive) else {
            return false;
        };
        let column = match &bound {
            BoundPredicate::Unary(expr) => expr.term(),
            BoundPredicate::Binary(expr) => expr.term(),
            BoundPredicate::Set(expr) => expr.term(),
            _ => return true,
        };
        let field = column.field();
        schema.as_struct().field_by_id(field.id).is_some()
            && field.field_type.is_primitive()
            && field.initial_default.is_none()
            && !identity_columns.contains(&field.id)
    };
    let filter = keep_readable(predicate, &readable);
    (!matches!(filter, Predicate::AlwaysTrue)).then_some(filter)
}

/// The number of values in all of the predicate's IN and NOT IN lists.
fn listed_values(predicate: &Predicate) -> usize {
    match predicate {
        Predicate::And(expr) | Predicate::Or(expr) => {
            let [left, right] = expr.inputs();
            listed_values(left) + listed_values(right)
        }
        Predicate::Not(expr) => {
            let [inner] = expr.inputs();
            listed_values(inner)
        }
        Predicate::Set(expr) => expr.literals().len(),
        _ => 0,
    }
}

/// Replaces each comparison that `readable` rejects with "matches everything". The filter then
/// matches more rows, which is safe.
///
/// A NOT is replaced as a whole. Under a NOT, "matches everything" turns into "matches nothing",
/// which would skip rows that match. The filter converter never makes a NOT, so this costs
/// nothing.
fn keep_readable(predicate: &Predicate, readable: &dyn Fn(&Predicate) -> bool) -> Predicate {
    match predicate {
        Predicate::And(expr) => {
            let [left, right] = expr.inputs();
            keep_readable(left, readable).and(keep_readable(right, readable))
        }
        // Debug switch: leave ORs out of the reader filter, to measure their per-row cost.
        Predicate::Or(_) if std::env::var_os("DFDBG_READER_KEEP_OR").is_none() => {
            Predicate::AlwaysTrue
        }
        Predicate::Or(expr) => {
            let [left, right] = expr.inputs();
            keep_readable(left, readable).or(keep_readable(right, readable))
        }
        Predicate::Not(_) => Predicate::AlwaysTrue,
        comparison if readable(comparison) => comparison.clone(),
        _ => Predicate::AlwaysTrue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::expr::Reference;
    use iceberg::spec::{Datum, Literal, NestedField, PrimitiveType, Schema, StructType, Type};
    use std::error::Error;

    #[test]
    fn leaves_out_columns_the_reader_gets_wrong() -> Result<(), Box<dyn Error>> {
        let schema = Arc::new(schema()?);
        let spec = PartitionSpec::builder(Arc::clone(&schema))
            .add_partition_field("day", "day", Transform::Identity)?
            .build()?;
        let predicate = id()
            .less_than(Datum::long(5))
            .and(Reference::new("day").equal_to(Datum::int(1)))
            .and(Reference::new("region").equal_to(Datum::string("eu")))
            .and(Reference::new("location.lat").equal_to(Datum::int(3)));

        let filter = reader_filter(&predicate, &spec, schema, true).ok_or("no filter is left")?;

        insta::assert_snapshot!(filter, @"id < 5");
        Ok(())
    }

    #[test]
    fn leaves_out_long_lists_and_keeps_their_bounds() -> Result<(), Box<dyn Error>> {
        let schema = Arc::new(schema()?);
        let spec = PartitionSpec::unpartition_spec();
        let values = (0..=MAX_LISTED_VALUES as i64).map(Datum::long);
        let predicate = id()
            .greater_than_or_equal_to(Datum::long(0))
            .and(id().less_than_or_equal_to(Datum::long(1000)))
            .and(id().is_in(values));

        let filter = reader_filter(&predicate, &spec, schema, true).ok_or("no filter is left")?;

        insta::assert_snapshot!(filter, @"(id >= 0) AND (id <= 1000)");
        Ok(())
    }

    #[test]
    fn leaves_out_a_not() -> Result<(), Box<dyn Error>> {
        let schema = Arc::new(schema()?);
        let spec = PartitionSpec::unpartition_spec();
        let predicate = !id().equal_to(Datum::long(1));

        assert_eq!(reader_filter(&predicate, &spec, schema, true), None);
        Ok(())
    }

    fn id() -> Reference {
        Reference::new("id")
    }

    // `day` is the partition column. `region` has a default value, and `location.lat` is nested.
    fn schema() -> iceberg::Result<Schema> {
        let location = StructType::new(vec![
            NestedField::optional(6, "lat", Type::Primitive(PrimitiveType::Int)).into(),
        ]);
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "day", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(4, "region", Type::Primitive(PrimitiveType::String))
                    .with_initial_default(Literal::string("us"))
                    .into(),
                NestedField::optional(5, "location", Type::Struct(location)).into(),
            ])
            .build()
    }
}
