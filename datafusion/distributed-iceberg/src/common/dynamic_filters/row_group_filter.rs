use std::sync::Arc;

use datafusion::common::HashSet;
use iceberg::expr::{Bind, BoundPredicate, Predicate};
use iceberg::scan::FileScanTask;
use iceberg::spec::{PartitionSpec, SchemaRef, Transform};

use crate::common::dynamic_filters::task_filter::and_predicate;

/// The most values the reader's dynamic filter may list, across all of its IN lists. The reader
/// checks every row of a file against its filter, and long lists make that slow. Above this, the
/// lists are left out, and the filter keeps only the bounds that come with them.
const MAX_LISTED_VALUES: usize = 200;

/// A copy of `task` whose filter also has the part of `predicate` the reader can use, so the
/// reader skips the row groups `predicate` rules out. `predicate` is the dynamic filters as an
/// Iceberg predicate. `None` when the reader can't use any part of it, and on any error.
pub(crate) fn with_dynamic_filter(
    task: &FileScanTask,
    predicate: &Predicate,
) -> Option<FileScanTask> {
    // The spec says which columns the reader fills in from the partition value.
    let spec = task.partition_spec()?;
    let filter = reader_filter(predicate, spec, task.schema_ref(), task.case_sensitive())?;
    // Keep the file's own filter, so no file gets a weaker filter than it has today.
    and_predicate(task, filter).ok()
}

/// The part of `predicate` the reader checks correctly in files written with `spec`. `None` when
/// no part is left.
///
/// The reader gets some columns wrong, so comparisons on them are left out:
/// - Identity partition columns. A file doesn't have to store them, and the reader treats a
///   column the file doesn't store as null. The partition filter checks these columns instead.
/// - Columns with a default value. In files written before the column was added, the reader
///   treats them as null too.
/// - Nested columns.
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
        // Binding the comparison on its own gives its column. One that doesn't bind is left out.
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

/// `predicate` with only the comparisons `readable` accepts. The others become "matches
/// everything", which keeps more rows. A NOT becomes "matches everything" as a whole, because
/// leaving out a comparison under it would keep fewer rows. The filter translator never makes a
/// NOT, so this loses nothing.
fn keep_readable(predicate: &Predicate, readable: &dyn Fn(&Predicate) -> bool) -> Predicate {
    match predicate {
        Predicate::And(expr) => {
            let [left, right] = expr.inputs();
            keep_readable(left, readable).and(keep_readable(right, readable))
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
