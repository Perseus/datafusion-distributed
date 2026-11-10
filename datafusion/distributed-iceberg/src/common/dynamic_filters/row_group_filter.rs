use datafusion::common::{HashMap, HashSet};
use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, BoundReference, Predicate, Reference, SetExpression,
    UnaryExpression,
};
use iceberg::scan::FileScanTask;
use iceberg::spec::{PartitionSpec, Schema, Transform};

/// The most values the reader's dynamic filter may list, across all of its IN lists. The reader
/// checks every row of a file against its filter, and long lists make that slow. Above this, the
/// lists are left out, and the filter keeps only the bounds that come with them.
const MAX_LISTED_VALUES: usize = 200;

/// Gives each file the scan's dynamic filters, combined with the file's own filter, so the reader
/// can skip the row groups the filters rule out.
#[derive(Default)]
pub(crate) struct RowGroupFilter {
    /// The part of the filters the reader can use, by partition spec id. `None` when no part is
    /// left. Every task in a scan has the same table schema, so the spec id is enough as a key.
    reader_filters: HashMap<i32, Option<Predicate>>,
}

impl RowGroupFilter {
    /// A copy of `task` whose filter also has `predicate`, the dynamic filters bound to the table
    /// schema. `None` when the reader can't use any part of `predicate`, and on any error.
    pub(crate) fn with_dynamic_filter(
        &mut self,
        predicate: &BoundPredicate,
        task: &FileScanTask,
    ) -> Option<FileScanTask> {
        // The spec says which columns the reader fills in from the partition value.
        let spec = task.partition_spec()?;
        let reader_filter = self
            .reader_filters
            .entry(spec.spec_id())
            .or_insert_with(|| reader_filter(predicate, spec, task.schema()))
            .as_ref()?;
        // Keep the file's own filter, so no file gets a weaker filter than it has today.
        let combined = match task.predicate() {
            Some(existing) => unbind(existing, task.schema())?.and(reader_filter.clone()),
            None => reader_filter.clone(),
        };
        let combined = combined.bind(task.schema_ref(), true).ok()?;
        with_predicate(task, combined).ok()
    }

    /// Forgets the reader filters. Called when the dynamic filters change.
    pub(crate) fn clear(&mut self) {
        self.reader_filters.clear();
    }
}

/// The part of `predicate` the reader checks correctly in files written with `spec`. `None` when
/// no part is left.
///
/// The reader gets some columns wrong, so comparisons on them are left out:
/// - Identity partition columns. A file doesn't have to store them, and the reader treats a
///   column the file doesn't store as null. The partition skipper checks these columns instead.
/// - Columns with a default value. In files written before the column was added, the reader
///   treats them as null too.
/// - Nested columns.
fn reader_filter(
    predicate: &BoundPredicate,
    spec: &PartitionSpec,
    schema: &Schema,
) -> Option<Predicate> {
    let identity_columns: HashSet<i32> = spec
        .fields()
        .iter()
        .filter(|field| field.transform == Transform::Identity)
        .map(|field| field.source_id)
        .collect();
    let keep_lists = listed_values(predicate) <= MAX_LISTED_VALUES;
    let readable = |comparison: &BoundPredicate| {
        let column = match comparison {
            BoundPredicate::Unary(expr) => expr.term(),
            BoundPredicate::Binary(expr) => expr.term(),
            BoundPredicate::Set(expr) if keep_lists => expr.term(),
            BoundPredicate::Set(_) => return false,
            _ => return true,
        };
        let field = column.field();
        schema.as_struct().field_by_id(field.id).is_some()
            && field.field_type.is_primitive()
            && field.initial_default.is_none()
            && !identity_columns.contains(&field.id)
    };
    let filter = keep_readable(predicate, schema, &readable)?;
    (!matches!(filter, Predicate::AlwaysTrue)).then_some(filter)
}

/// The number of values in all of the predicate's IN and NOT IN lists.
fn listed_values(predicate: &BoundPredicate) -> usize {
    match predicate {
        BoundPredicate::And(expr) | BoundPredicate::Or(expr) => {
            let [left, right] = expr.inputs();
            listed_values(left) + listed_values(right)
        }
        BoundPredicate::Not(expr) => {
            let [inner] = expr.inputs();
            listed_values(inner)
        }
        BoundPredicate::Set(expr) => expr.literals().len(),
        _ => 0,
    }
}

/// Rebuilds `predicate` as an unbound predicate with only the comparisons `readable` accepts.
/// The others become "matches everything", which keeps more rows. A NOT becomes "matches
/// everything" as a whole, because leaving out a comparison under it would keep fewer rows. The
/// filter translator never makes a NOT, so this loses nothing. `None` if a column isn't in
/// `schema`.
fn keep_readable(
    predicate: &BoundPredicate,
    schema: &Schema,
    readable: &dyn Fn(&BoundPredicate) -> bool,
) -> Option<Predicate> {
    Some(match predicate {
        BoundPredicate::And(expr) => {
            let [left, right] = expr.inputs();
            keep_readable(left, schema, readable)?.and(keep_readable(right, schema, readable)?)
        }
        BoundPredicate::Or(expr) => {
            let [left, right] = expr.inputs();
            keep_readable(left, schema, readable)?.or(keep_readable(right, schema, readable)?)
        }
        BoundPredicate::Not(_) => Predicate::AlwaysTrue,
        comparison if !readable(comparison) => Predicate::AlwaysTrue,
        comparison => unbind(comparison, schema)?,
    })
}

/// Rebuilds a bound predicate as an unbound one, so it can be combined with another predicate
/// and bound again. iceberg-rust can only combine unbound predicates. Columns get their full name
/// in `schema`, so binding again is case-sensitive. `None` if a column isn't in `schema`.
fn unbind(predicate: &BoundPredicate, schema: &Schema) -> Option<Predicate> {
    let reference = |column: &BoundReference| {
        schema
            .name_by_field_id(column.field().id)
            .map(Reference::new)
    };
    Some(match predicate {
        BoundPredicate::AlwaysTrue => Predicate::AlwaysTrue,
        BoundPredicate::AlwaysFalse => Predicate::AlwaysFalse,
        BoundPredicate::And(expr) => {
            let [left, right] = expr.inputs();
            unbind(left, schema)?.and(unbind(right, schema)?)
        }
        BoundPredicate::Or(expr) => {
            let [left, right] = expr.inputs();
            unbind(left, schema)?.or(unbind(right, schema)?)
        }
        BoundPredicate::Not(expr) => {
            let [inner] = expr.inputs();
            !unbind(inner, schema)?
        }
        BoundPredicate::Unary(expr) => {
            Predicate::Unary(UnaryExpression::new(expr.op(), reference(expr.term())?))
        }
        BoundPredicate::Binary(expr) => Predicate::Binary(BinaryExpression::new(
            expr.op(),
            reference(expr.term())?,
            expr.literal().clone(),
        )),
        BoundPredicate::Set(expr) => Predicate::Set(SetExpression::new(
            expr.op(),
            reference(expr.term())?,
            expr.literals().clone(),
        )),
    })
}

/// A copy of `task` that reads with `predicate`. iceberg-rust has no way to change a task's
/// filter, so this rebuilds the task from every one of its fields.
fn with_predicate(task: &FileScanTask, predicate: BoundPredicate) -> iceberg::Result<FileScanTask> {
    FileScanTask::builder()
        .with_file_size_in_bytes(task.file_size_in_bytes())
        .with_start(task.start())
        .with_length(task.length())
        .with_record_count(task.record_count())
        .with_first_row_id(task.first_row_id())
        .with_data_sequence_number(task.data_sequence_number())
        .with_data_file_path(task.data_file_path().to_string())
        .with_data_file_format(task.data_file_format())
        .with_schema(task.schema_ref())
        .with_project_field_ids(task.project_field_ids().to_vec())
        .with_predicate(Some(predicate))
        .with_deletes(task.deletes().to_vec())
        .with_partition(task.partition().cloned())
        .with_partition_spec(task.partition_spec().cloned())
        .with_name_mapping(task.name_mapping().cloned())
        .with_unified_partition_type(task.unified_partition_type().cloned())
        .with_sort_order_id(task.sort_order_id())
        .with_sort_order(task.sort_order().cloned())
        .with_column_stats(task.column_stats().cloned())
        .with_case_sensitive(task.case_sensitive())
        .with_key_metadata(task.key_metadata().map(Box::from))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_distributed_iceberg_test_tools::create_taxi_table;
    use futures::TryStreamExt;
    use iceberg::spec::{Datum, Literal, NestedField, PrimitiveType, StructType, Type};
    use std::error::Error;
    use std::sync::Arc;

    #[test]
    fn unbinding_and_binding_again_gives_the_same_predicate() -> Result<(), Box<dyn Error>> {
        let schema = Arc::new(schema()?);
        let predicate = id()
            .less_than(Datum::long(5))
            .and(!id().is_in([Datum::long(1), Datum::long(2)]))
            .or(Reference::new("name").starts_with(Datum::string("a")))
            .or(Reference::new("location.lat").is_null())
            .bind(Arc::clone(&schema), true)?;

        let unbound = unbind(&predicate, &schema).ok_or("a column isn't in the schema")?;

        assert_eq!(unbound.bind(schema, true)?, predicate);
        Ok(())
    }

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
            .and(Reference::new("location.lat").equal_to(Datum::int(3)))
            .bind(Arc::clone(&schema), true)?;

        let filter = reader_filter(&predicate, &spec, &schema).ok_or("no filter is left")?;

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
            .and(id().is_in(values))
            .bind(Arc::clone(&schema), true)?;

        let filter = reader_filter(&predicate, &spec, &schema).ok_or("no filter is left")?;

        insta::assert_snapshot!(filter, @"(id >= 0) AND (id <= 1000)");
        Ok(())
    }

    #[test]
    fn leaves_out_a_not() -> Result<(), Box<dyn Error>> {
        let schema = Arc::new(schema()?);
        let spec = PartitionSpec::unpartition_spec();
        let predicate = (!id().equal_to(Datum::long(1))).bind(Arc::clone(&schema), true)?;

        assert_eq!(reader_filter(&predicate, &spec, &schema), None);
        Ok(())
    }

    /// `with_predicate` copies a task field by field. If iceberg-rust adds a field, the copy
    /// would silently drop it, so this lists the fields it copies.
    #[tokio::test]
    async fn the_copy_of_a_task_keeps_every_field() -> Result<(), Box<dyn Error>> {
        let table = create_taxi_table().await?;
        let tasks: Vec<FileScanTask> = table
            .table()
            .scan()
            .with_filter(Reference::new("pickup_date").is_not_null())
            .build()?
            .plan_files()
            .await?
            .try_collect()
            .await?;
        let task = tasks.first().ok_or("the scan planned no files")?;
        let predicate = task.predicate().ok_or("the task has no filter")?;

        let copy = with_predicate(task, predicate.clone())?;

        assert_eq!(&copy, task);
        insta::assert_snapshot!(field_names(task).join("\n"), @"
        file_size_in_bytes
        start
        length
        record_count
        first_row_id
        data_sequence_number
        data_file_path
        data_file_format
        schema
        project_field_ids
        predicate
        deletes
        partition
        partition_spec
        name_mapping
        unified_partition_type
        sort_order_id
        sort_order
        column_stats
        case_sensitive
        key_metadata
        ");
        Ok(())
    }

    /// The names of a struct's fields, read from its pretty `Debug` output, where they are the
    /// lines indented by four spaces.
    fn field_names(value: &impl std::fmt::Debug) -> Vec<String> {
        format!("{value:#?}")
            .lines()
            .filter_map(|line| line.strip_prefix("    "))
            .filter(|line| !line.starts_with(' '))
            .filter_map(|line| line.split_once(':'))
            .map(|(name, _)| name.to_string())
            .collect()
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
                NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(4, "region", Type::Primitive(PrimitiveType::String))
                    .with_initial_default(Literal::string("us"))
                    .into(),
                NestedField::optional(5, "location", Type::Struct(location)).into(),
            ])
            .build()
    }
}
