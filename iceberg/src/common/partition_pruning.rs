//! Skips whole data files whose partition values can't match a scan's kept dynamic filters,
//! before the Iceberg Arrow reader opens them.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, ScalarValue, Statistics};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use datafusion::physical_expr::{PhysicalExpr, conjunction};
use datafusion::physical_optimizer::pruning::FilePruner;
use datafusion::physical_plan::metrics::Count;
use iceberg::scan::FileScanTask;
use iceberg::spec::{Literal, PrimitiveLiteral, PrimitiveType, Transform, Type};

use crate::common::runtime_predicate::dynamic_conjuncts;

/// Decides, per data-file task, whether the scan's kept dynamic filters rule out every row of
/// the file given its partition values.
///
/// It uses DataFusion's [`FilePruner`], the same check DataFusion's Parquet scan runs before
/// opening a file, over statistics built by [`file_statistics`]. A file is skipped only when
/// the pruner proves no row can match; any error or unsupported case keeps the file. The
/// filters are read when the task is checked, never waited on, and are still applied by the
/// `FilterExec` above the scan, so skipping only decides which files are read.
pub(crate) struct PartitionFilePruner {
    predicate: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
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
        })
    }

    /// Whether `task`'s file can be skipped: `true` only if no row in it can pass the filters.
    pub(crate) fn should_skip(&self, task: &FileScanTask) -> bool {
        let statistics = file_statistics(task, &self.schema);
        let file = PartitionedFile::new(task.data_file_path(), task.file_size_in_bytes())
            .with_statistics(Arc::new(statistics));
        let Some(mut pruner) =
            FilePruner::try_new(Arc::clone(&self.predicate), &self.schema, &file, Count::new())
        else {
            return false;
        };
        pruner.should_prune().unwrap_or(false)
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
/// `None` when the types don't exactly correspond.
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
        (PrimitiveType::Float, PrimitiveLiteral::Float(v), DataType::Float32) => {
            ScalarValue::Float32(Some(v.0))
        }
        (PrimitiveType::Double, PrimitiveLiteral::Double(v), DataType::Float64) => {
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
