//! Skips whole data files whose partition values can't match a scan's kept dynamic filters,
//! before the Iceberg Arrow reader opens them.

use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use datafusion::common::pruning::PrunableStatistics;
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, ScalarValue, Statistics};
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use datafusion::physical_expr::{PhysicalExpr, conjunction};
use datafusion::physical_expr_common::physical_expr::{
    snapshot_generation, snapshot_physical_expr,
};
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};
use iceberg::scan::FileScanTask;
use iceberg::spec::{Literal, PrimitiveLiteral, PrimitiveType, Transform, Type};

use crate::common::case_as_or::rewrite_case_as_or;
use crate::common::runtime_predicate::dynamic_conjuncts;

/// Decides, per data-file task, whether the scan's kept dynamic filters rule out every row of
/// the file given its partition values.
///
/// It evaluates a DataFusion [`PruningPredicate`], the check DataFusion's Parquet scan runs
/// before opening a file, over statistics built by [`file_statistics`]. A file is skipped only
/// when the predicate proves no row can match; any error or unsupported case keeps the file.
/// The filters are read when the task is checked, never waited on, and are still applied by
/// the `FilterExec` above the scan, so skipping only decides which files are read.
///
/// Before building the predicate, the filters' current value goes through
/// [`rewrite_case_as_or`], so a partitioned hash join's routing `CASE` can prune too.
/// DataFusion's `FilePruner` snapshots the dynamic filters itself and can't take a rewritten
/// value, so this uses [`PruningPredicate`] directly and reproduces its change tracking: the
/// predicate is rebuilt only when the filters' generation changes.
pub(crate) struct PartitionFilePruner {
    predicate: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    /// The pruning predicate built for a generation of `predicate`, `None` when it can't
    /// prune.
    cached: Mutex<Option<(u64, Option<Arc<PruningPredicate>>)>>,
}

impl PartitionFilePruner {
    /// Builds a pruner over the dynamic filters among `filters` and their top-level AND
    /// conjuncts, expressed against `schema`, the scan's output schema. Returns `None` when
    /// there is no dynamic filter.
    pub(crate) fn try_new(filters: &[Arc<dyn PhysicalExpr>], schema: &SchemaRef) -> Option<Self> {
        let dynamic = dynamic_conjuncts(filters);
        if dynamic.is_empty() {
            return None;
        }
        Some(Self {
            predicate: conjunction(dynamic),
            schema: Arc::clone(schema),
            cached: Mutex::new(None),
        })
    }

    /// Whether `task`'s file can be skipped: `true` only if no row in it can pass the filters.
    pub(crate) fn should_skip(&self, task: &FileScanTask) -> bool {
        let Some(pruning_predicate) = self.pruning_predicate() else {
            return false;
        };
        let statistics = PrunableStatistics::new(
            vec![Arc::new(file_statistics(task, &self.schema))],
            Arc::clone(&self.schema),
        );
        match pruning_predicate.prune(&statistics) {
            Ok(keep) => keep.iter().all(|keep| !keep),
            Err(_) => false,
        }
    }

    /// The pruning predicate for the filters' current value, rebuilt only when their
    /// generation changed since the last call. The generation is read before the value, so
    /// the cached predicate is never older than its generation.
    fn pruning_predicate(&self) -> Option<Arc<PruningPredicate>> {
        let generation = snapshot_generation(&self.predicate);
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((cached_generation, predicate)) = cached.as_ref()
            && *cached_generation == generation
        {
            return predicate.clone();
        }
        let predicate = snapshot_physical_expr(Arc::clone(&self.predicate))
            .ok()
            .and_then(|current| {
                PruningPredicateBuilder::new()
                    .with_file_schema(Arc::clone(&self.schema))
                    .build(rewrite_case_as_or(&current))
            });
        *cached = Some((generation, predicate.clone()));
        predicate
    }
}

/// File-level statistics for `task`, one column per field of `schema`, the scan's output schema.
///
/// Only partition values are used for now; every column they don't describe is unknown.
/// Column bounds from the task's manifest entry could later be merged in per column.
pub(crate) fn file_statistics(task: &FileScanTask, schema: &SchemaRef) -> Statistics {
    let num_rows = task
        .record_count()
        .map_or(Precision::Absent, |n| Precision::Exact(n as usize));
    let mut column_statistics = vec![ColumnStatistics::new_unknown(); schema.fields().len()];
    for (index, stats) in partition_column_statistics(task, schema, num_rows) {
        column_statistics[index] = stats;
    }
    Statistics {
        num_rows,
        total_byte_size: Precision::Absent,
        column_statistics,
    }
}

/// Statistics, by output column index, of the columns that are the source of an identity
/// field in `task`'s own partition spec, so each file is read with the spec it was written
/// with. A value sets min = max = the value; a null value means the column is all nulls, which
/// is only stated when the file's row count is known. Values whose type doesn't exactly match
/// the output column are left out.
fn partition_column_statistics(
    task: &FileScanTask,
    schema: &SchemaRef,
    num_rows: Precision<usize>,
) -> Vec<(usize, ColumnStatistics)> {
    let (Some(spec), Some(partition)) = (task.partition_spec(), task.partition()) else {
        return Vec::new();
    };
    let mut columns = Vec::new();
    for (field, value) in spec.fields().iter().zip(partition.iter()) {
        if field.transform != Transform::Identity {
            continue;
        }
        let Some(index) = output_column(schema, field.source_id) else {
            continue;
        };
        let Some(source) = task.schema().field_by_id(field.source_id) else {
            continue;
        };
        let Type::Primitive(source_type) = &*source.field_type else {
            continue;
        };
        let output_type = schema.field(index).data_type();
        let stats = match value {
            None => {
                let Precision::Exact(rows) = num_rows else {
                    continue;
                };
                ColumnStatistics::new_unknown().with_null_count(Precision::Exact(rows))
            }
            Some(literal) => {
                let Some(scalar) = literal_to_scalar(literal, source_type, output_type) else {
                    continue;
                };
                ColumnStatistics::new_unknown()
                    .with_null_count(Precision::Exact(0))
                    .with_min_value(Precision::Exact(scalar.clone()))
                    .with_max_value(Precision::Exact(scalar))
            }
        };
        columns.push((index, stats));
    }
    columns
}

/// The index of the output column carrying Iceberg field id `field_id`.
fn output_column(schema: &SchemaRef, field_id: i32) -> Option<usize> {
    let field_id = field_id.to_string();
    schema
        .fields()
        .iter()
        .position(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY) == Some(&field_id))
}

/// Converts a partition value of Iceberg type `source_type` to a value of `output_type`, or
/// `None` when the types don't exactly correspond, or when a float value is zero or NaN.
fn literal_to_scalar(
    literal: &Literal,
    source_type: &PrimitiveType,
    output_type: &DataType,
) -> Option<ScalarValue> {
    let Literal::Primitive(value) = literal else {
        return None;
    };
    let scalar = match (source_type, value, output_type) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(v), DataType::Boolean) => {
            ScalarValue::Boolean(Some(*v))
        }
        (PrimitiveType::Int, PrimitiveLiteral::Int(v), DataType::Int32) => {
            ScalarValue::Int32(Some(*v))
        }
        (PrimitiveType::Long, PrimitiveLiteral::Long(v), DataType::Int64) => {
            ScalarValue::Int64(Some(*v))
        }
        // Iceberg puts -0 and +0, and every NaN, in one partition, so a zero or NaN partition
        // value doesn't bound the file's rows exactly: leave it unknown.
        (PrimitiveType::Float, PrimitiveLiteral::Float(v), DataType::Float32)
            if v.0 != 0.0 && !v.0.is_nan() =>
        {
            ScalarValue::Float32(Some(v.0))
        }
        (PrimitiveType::Double, PrimitiveLiteral::Double(v), DataType::Float64)
            if v.0 != 0.0 && !v.0.is_nan() =>
        {
            ScalarValue::Float64(Some(v.0))
        }
        (PrimitiveType::Date, PrimitiveLiteral::Int(v), DataType::Date32) => {
            ScalarValue::Date32(Some(*v))
        }
        (PrimitiveType::String, PrimitiveLiteral::String(v), DataType::Utf8) => {
            ScalarValue::Utf8(Some(v.clone()))
        }
        (PrimitiveType::String, PrimitiveLiteral::String(v), DataType::LargeUtf8) => {
            ScalarValue::LargeUtf8(Some(v.clone()))
        }
        (PrimitiveType::String, PrimitiveLiteral::String(v), DataType::Utf8View) => {
            ScalarValue::Utf8View(Some(v.clone()))
        }
        (
            PrimitiveType::Timestamp,
            PrimitiveLiteral::Long(v),
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ) => ScalarValue::TimestampMicrosecond(Some(*v), None),
        (
            PrimitiveType::Timestamptz,
            PrimitiveLiteral::Long(v),
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)),
        ) => ScalarValue::TimestampMicrosecond(Some(*v), Some(Arc::clone(tz))),
        (
            PrimitiveType::TimestampNs,
            PrimitiveLiteral::Long(v),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
        ) => ScalarValue::TimestampNanosecond(Some(*v), None),
        (
            PrimitiveType::TimestamptzNs,
            PrimitiveLiteral::Long(v),
            DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)),
        ) => ScalarValue::TimestampNanosecond(Some(*v), Some(Arc::clone(tz))),
        _ => return None,
    };
    Some(scalar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema};
    use iceberg::arrow::schema_to_arrow_schema;
    use iceberg::spec::{DataFileFormat, NestedField, PartitionSpec, Schema, Struct};

    #[test]
    fn identity_partition_value_is_exact_bounds() {
        let task = task(
            spec(&[("day", Transform::Identity)]),
            [Some(Literal::date(19732))],
        );

        insta::assert_snapshot!(describe(&task, &output_schema()), @"
        rows=Exact(10)
        id: unknown
        day: min=Exact(Date32(\"2024-01-10\")) max=Exact(Date32(\"2024-01-10\")) nulls=Exact(0)
        name: unknown
        score: unknown
        ");
    }

    #[test]
    fn non_identity_transform_is_ignored() {
        let task = task(
            spec(&[("id", Transform::Bucket(4))]),
            [Some(Literal::int(1))],
        );

        insta::assert_snapshot!(describe(&task, &output_schema()), @"
        rows=Exact(10)
        id: unknown
        day: unknown
        name: unknown
        score: unknown
        ");
    }

    #[test]
    fn null_partition_value_means_all_nulls() {
        let task = task(spec(&[("name", Transform::Identity)]), [None]);

        insta::assert_snapshot!(describe(&task, &output_schema()), @"
        rows=Exact(10)
        id: unknown
        day: unknown
        name: min=Absent max=Absent nulls=Exact(10)
        score: unknown
        ");
    }

    /// Each file is described by its own spec: an older file partitioned by `id` and a newer
    /// one partitioned by `name` describe different columns.
    #[test]
    fn each_file_uses_its_own_partition_spec() {
        let old = task(
            spec(&[("id", Transform::Identity)]),
            [Some(Literal::int(7))],
        );
        let new = task(
            spec(&[("name", Transform::Identity)]),
            [Some(Literal::string("a"))],
        );

        insta::assert_snapshot!(describe(&old, &output_schema()), @"
        rows=Exact(10)
        id: min=Exact(Int32(7)) max=Exact(Int32(7)) nulls=Exact(0)
        day: unknown
        name: unknown
        score: unknown
        ");
        insta::assert_snapshot!(describe(&new, &output_schema()), @"
        rows=Exact(10)
        id: unknown
        day: unknown
        name: min=Exact(Utf8(\"a\")) max=Exact(Utf8(\"a\")) nulls=Exact(0)
        score: unknown
        ");
    }

    /// The output reads `id` as Int64, which doesn't match the Iceberg Int partition value.
    #[test]
    fn type_mismatch_is_left_unknown() {
        let task = task(
            spec(&[("id", Transform::Identity)]),
            [Some(Literal::int(7))],
        );
        let mut fields: Vec<Field> = output_schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields[0] = fields[0].clone().with_data_type(DataType::Int64);
        let output = Arc::new(ArrowSchema::new(fields));

        insta::assert_snapshot!(describe(&task, &output), @"
        rows=Exact(10)
        id: unknown
        day: unknown
        name: unknown
        score: unknown
        ");
    }

    /// Iceberg puts -0.0 and 0.0 rows in one partition, and every NaN in another, so those
    /// values are left unknown; other float values are exact.
    #[test]
    fn zero_and_nan_float_partition_values_are_left_unknown() {
        let describe_score = |value: f64| {
            let task = task(
                spec(&[("score", Transform::Identity)]),
                [Some(Literal::double(value))],
            );
            describe(&task, &output_schema())
                .lines()
                .find(|line| line.starts_with("score"))
                .unwrap()
                .to_string()
        };

        insta::assert_snapshot!(
            [0.0, -0.0, f64::NAN, 1.5].map(describe_score).join("\n"),
            @"
        score: unknown
        score: unknown
        score: unknown
        score: min=Exact(Float64(1.5)) max=Exact(Float64(1.5)) nulls=Exact(0)
        "
        );
    }

    fn table_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "day", Type::Primitive(PrimitiveType::Date)).into(),
                    NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(4, "score", Type::Primitive(PrimitiveType::Double))
                        .into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn output_schema() -> SchemaRef {
        Arc::new(schema_to_arrow_schema(&table_schema()).unwrap())
    }

    fn spec(fields: &[(&str, Transform)]) -> Arc<PartitionSpec> {
        let mut builder = PartitionSpec::builder(table_schema());
        for (column, transform) in fields {
            builder = builder
                .add_partition_field(*column, format!("{column}_part"), *transform)
                .unwrap();
        }
        Arc::new(builder.build().unwrap())
    }

    fn task<const N: usize>(
        spec: Arc<PartitionSpec>,
        values: [Option<Literal>; N],
    ) -> FileScanTask {
        FileScanTask::builder()
            .with_file_size_in_bytes(100)
            .with_start(0)
            .with_length(100)
            .with_record_count(Some(10))
            .with_data_file_path("file:///t/data/f.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(table_schema())
            .with_project_field_ids(vec![1, 2, 3, 4])
            .with_partition(Some(Struct::from_iter(values)))
            .with_partition_spec(Some(spec))
            .with_case_sensitive(true)
            .build()
            .unwrap()
    }

    fn describe(task: &FileScanTask, schema: &SchemaRef) -> String {
        let stats = file_statistics(task, schema);
        let mut lines = vec![format!("rows={:?}", stats.num_rows)];
        for (field, column) in schema.fields().iter().zip(&stats.column_statistics) {
            let line = if column == &ColumnStatistics::new_unknown() {
                "unknown".to_string()
            } else {
                format!(
                    "min={:?} max={:?} nulls={:?}",
                    column.min_value, column.max_value, column.null_count
                )
            };
            lines.push(format!("{}: {line}", field.name()));
        }
        lines.join("\n")
    }
}
