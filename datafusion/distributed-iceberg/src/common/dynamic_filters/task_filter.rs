//! Adds a filter to a file scan task. The iceberg-rust version this crate uses has no way to
//! change a task's filter, and can't AND two bound filters from outside, so this file rebuilds
//! the task. Once iceberg-rust has a way, delete this file and call it.

use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, BoundReference, Predicate, Reference, SetExpression,
    UnaryExpression,
};
use iceberg::scan::FileScanTask;
use iceberg::spec::Schema;
use iceberg::{Error, ErrorKind, Result};

/// A copy of `task` whose filter is its own filter AND `predicate`.
pub(crate) fn and_predicate(task: &FileScanTask, predicate: Predicate) -> Result<FileScanTask> {
    let combined = match task.predicate() {
        Some(existing) => unbind(existing, task.schema())?.and(predicate),
        None => predicate,
    };
    let combined = combined.bind(task.schema_ref(), task.case_sensitive())?;
    with_predicate(task, combined)
}

/// Rebuilds a bound predicate as an unbound one, so it can be combined with another predicate
/// and bound again. iceberg-rust can only combine unbound predicates. Columns get their full name
/// in `schema`. Fails if a column isn't in `schema`.
fn unbind(predicate: &BoundPredicate, schema: &Schema) -> Result<Predicate> {
    let reference = |column: &BoundReference| {
        let id = column.field().id;
        schema
            .name_by_field_id(id)
            .map(Reference::new)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("column {id} isn't in the schema"),
                )
            })
    };
    Ok(match predicate {
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

/// A copy of `task` that reads with `predicate`, rebuilt from every one of the task's fields.
fn with_predicate(task: &FileScanTask, predicate: BoundPredicate) -> Result<FileScanTask> {
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
        .with_case_sensitive(task.case_sensitive())
        .with_key_metadata(task.key_metadata().map(Box::from))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_distributed_iceberg_test_tools::create_taxi_table;
    use futures::TryStreamExt;
    use iceberg::spec::{Datum, NestedField, PrimitiveType, StructType, Type};
    use std::error::Error;
    use std::sync::Arc;

    #[test]
    fn unbinding_and_binding_again_gives_the_same_predicate()
    -> std::result::Result<(), Box<dyn Error>> {
        let location = StructType::new(vec![
            NestedField::optional(3, "lat", Type::Primitive(PrimitiveType::Int)).into(),
        ]);
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(4, "location", Type::Struct(location)).into(),
                ])
                .build()?,
        );
        let id = || Reference::new("id");
        let predicate = id()
            .less_than(Datum::long(5))
            .and(!id().is_in([Datum::long(1), Datum::long(2)]))
            .or(Reference::new("name").starts_with(Datum::string("a")))
            .or(Reference::new("location.lat").is_null())
            .bind(Arc::clone(&schema), true)?;

        let unbound = unbind(&predicate, &schema)?;

        assert_eq!(unbound.bind(schema, true)?, predicate);
        Ok(())
    }

    /// `with_predicate` copies a task field by field. If iceberg-rust adds a field, the copy
    /// would silently drop it, so this lists the fields it copies.
    #[tokio::test]
    async fn the_copy_of_a_task_keeps_every_field() -> std::result::Result<(), Box<dyn Error>> {
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
}
