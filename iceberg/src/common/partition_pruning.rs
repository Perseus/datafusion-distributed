//! Skips whole data files whose partition values or column bounds can't match a scan's kept
//! dynamic filters, before the Iceberg Arrow reader opens them.

use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{ArrayRef, BooleanArray};
use datafusion::arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use datafusion::common::Column;
use datafusion::common::pruning::{PrunableStatistics, PruningStatistics};
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, ScalarValue, Statistics};
use datafusion::parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};
use iceberg::scan::FileScanTask;
use iceberg::spec::{Datum, Literal, PrimitiveLiteral, PrimitiveType, Transform, Type};

use crate::common::dynamic_domain::CompactedDynamicFilters;

/// Decides, per data-file task, whether the scan's kept dynamic filters rule out every row of
/// the file given its partition values and column bounds.
///
/// It evaluates a DataFusion [`PruningPredicate`], the check DataFusion's Parquet scan runs
/// before opening a file, over statistics built by [`file_statistics`]. A file is skipped only
/// when the predicate proves no row can match; any error or unsupported case keeps the file
/// (fail open). The filters are read when the task is checked, never waited on.
///
/// There is no `FilterExec` above the scan for a dynamic filter: the query stays exact because
/// the filter's producer re-checks rows itself (the hash join probes its hash table, TopK and
/// aggregates re-evaluate their threshold). Skipping a file only removes rows the filter
/// already rules out, so it decides which files are read, never which rows are returned.
/// Delete files only remove rows, so skipping a data file with them is safe too.
///
/// The predicate is built from the filters' compacted, latched value, see
/// [`CompactedDynamicFilters`], so a partitioned hash join's routing `CASE` can skip files too.
/// DataFusion's `FilePruner` snapshots the dynamic filters itself and can't take a rewritten
/// value, so this uses [`PruningPredicate`] directly and reproduces its change tracking: the
/// predicate is rebuilt only when the filters' generation changes.
pub(crate) struct PartitionFilePruner {
    filters: Arc<CompactedDynamicFilters>,
    schema: SchemaRef,
    /// The pruning predicate built for a generation of `filters`, `None` when it can't prune.
    cached: Mutex<Option<(u64, Option<Arc<PruningPredicate>>)>>,
}

impl PartitionFilePruner {
    /// Builds a pruner over `filters`, expressed against `schema`, the scan's output schema.
    pub(crate) fn new(filters: Arc<CompactedDynamicFilters>, schema: &SchemaRef) -> Self {
        Self {
            filters,
            schema: Arc::clone(schema),
            cached: Mutex::new(None),
        }
    }

    /// Whether `task`'s file can be skipped: `true` only if no row in it can pass the filters.
    #[cfg(test)]
    pub(crate) fn should_skip(&self, task: &FileScanTask) -> bool {
        self.skip_reason(task).is_some()
    }

    /// Why `task`'s file can be skipped, `None` when some row in it may pass the filters.
    ///
    /// Its identity partition values are checked first; only a file they keep and that carries
    /// column statistics is checked again with its column bounds, so a skip is attributed to
    /// bounds only when the partition values alone couldn't rule the file out.
    pub(crate) fn skip_reason(&self, task: &FileScanTask) -> Option<SkipReason> {
        let pruning_predicate = self.pruning_predicate()?;
        let prunes = |with_bounds| {
            let statistics = FileStatistics::new(
                statistics_for(task, &self.schema, with_bounds),
                &self.schema,
            );
            match pruning_predicate.prune(&statistics) {
                Ok(keep) => keep.iter().all(|keep| !keep),
                Err(_) => false,
            }
        };
        if prunes(false) {
            Some(SkipReason::PartitionValues)
        } else if task.column_stats().is_some() && prunes(true) {
            Some(SkipReason::ColumnBounds)
        } else {
            None
        }
    }

    /// The pruning predicate for the filters' current value, rebuilt only when their
    /// generation changed since the last call.
    fn pruning_predicate(&self) -> Option<Arc<PruningPredicate>> {
        let generation = self.filters.generation();
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((cached_generation, predicate)) = cached.as_ref()
            && *cached_generation == generation
        {
            return predicate.clone();
        }
        let (generation, current) = self.filters.current();
        let predicate = PruningPredicateBuilder::new()
            .with_file_schema(Arc::clone(&self.schema))
            .build(current);
        *cached = Some((generation, predicate.clone()));
        predicate
    }
}

/// What ruled a skipped file out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipReason {
    /// Its identity partition values.
    PartitionValues,
    /// Its column bounds, which its partition values alone couldn't rule out.
    ColumnBounds,
}

/// The columns to request column statistics for, so files can be skipped by their bounds: the
/// scan's projected columns, or every column when `projection` is `None`, that are primitive
/// fields of `schema`, the scanned snapshot's schema. These are the only columns a filter on
/// the scan's output can reference; other names, such as metadata columns, have no statistics.
pub(crate) fn column_stats_columns(
    schema: &iceberg::spec::Schema,
    projection: Option<&[String]>,
) -> Vec<String> {
    let names: Vec<String> = match projection {
        Some(names) => names.to_vec(),
        None => schema
            .as_struct()
            .fields()
            .iter()
            .map(|f| f.name.clone())
            .collect(),
    };
    names
        .into_iter()
        .filter(|name| {
            schema
                .field_by_name(name)
                .is_some_and(|field| field.field_type.is_primitive())
        })
        .collect()
}

/// File-level statistics for `task`, one column per field of `schema`, the scan's output schema.
///
/// Columns are described by the task's column bounds, see [`column_bound_statistics`], and by
/// its identity partition values, see [`partition_column_statistics`]. A partition value is
/// exact, so where both describe a column it replaces the bounds. Every other column is unknown.
#[cfg(test)]
pub(crate) fn file_statistics(task: &FileScanTask, schema: &SchemaRef) -> Statistics {
    statistics_for(task, schema, true)
}

/// [`file_statistics`], leaving out the column bounds unless `with_bounds`.
fn statistics_for(task: &FileScanTask, schema: &SchemaRef, with_bounds: bool) -> Statistics {
    let num_rows = task
        .record_count()
        .map_or(Precision::Absent, |n| Precision::Exact(n as usize));
    let mut column_statistics = vec![ColumnStatistics::new_unknown(); schema.fields().len()];
    if with_bounds {
        for (index, stats) in column_bound_statistics(task, schema) {
            column_statistics[index] = stats;
        }
    }
    for (index, stats) in partition_column_statistics(task, schema, num_rows) {
        column_statistics[index] = stats;
    }
    Statistics {
        num_rows,
        total_byte_size: Precision::Absent,
        column_statistics,
    }
}

/// Statistics, by output column index, from the column statistics the task carries from its
/// manifest entry (requested through `TableScanBuilder::with_column_stats`).
///
/// Bounds become inexact min and max: Iceberg may truncate string and binary bounds, but a
/// truncated lower bound still bounds the values from below and a truncated upper bound is
/// rounded up, and DataFusion compares them byte-wise as Iceberg does. A missing bound is
/// unknown, never empty. A bound is used only when its type exactly matches the output column.
///
/// Float and double bounds exclude NaN, which DataFusion orders above every value, or below it
/// for a negative NaN, so they are used only when the file is known to hold no NaN. Their zero
/// bounds are widened, a lower bound to -0.0 and an upper bound to +0.0, because Iceberg doesn't
/// order -0.0 before +0.0 while DataFusion does.
///
/// Null counts are exact for the file. Delete files only remove rows, so these statistics
/// cover more rows than the scan returns, which can only keep files.
fn column_bound_statistics(
    task: &FileScanTask,
    schema: &SchemaRef,
) -> Vec<(usize, ColumnStatistics)> {
    let Some(stats) = task.column_stats() else {
        return Vec::new();
    };
    let mut columns = Vec::new();
    for (index, field) in schema.fields().iter().enumerate() {
        let Some(field_id) = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id| id.parse::<i32>().ok())
        else {
            continue;
        };
        let output_type = field.data_type();
        let mut column = ColumnStatistics::new_unknown();
        if let Some(nulls) = stats.null_value_counts().get(&field_id) {
            column = column.with_null_count(Precision::Exact(*nulls as usize));
        }
        let is_float = matches!(output_type, DataType::Float32 | DataType::Float64);
        let bounds_usable = !is_float
            || stats
                .nan_value_counts()
                .get(&field_id)
                .is_some_and(|n| *n == 0);
        if bounds_usable {
            let bound = |bound: Option<&Datum>, lower: bool| {
                widen_float_zero(bound_to_scalar(bound?, output_type)?, lower)
            };
            if let Some(min) = bound(stats.lower_bounds().get(&field_id), true) {
                column = column.with_min_value(Precision::Inexact(min));
            }
            if let Some(max) = bound(stats.upper_bounds().get(&field_id), false) {
                column = column.with_max_value(Precision::Inexact(max));
            }
        }
        if column != ColumnStatistics::new_unknown() {
            columns.push((index, column));
        }
    }
    columns
}

/// Converts a column bound to a value of `output_type`, `None` when the types don't exactly
/// correspond.
fn bound_to_scalar(bound: &Datum, output_type: &DataType) -> Option<ScalarValue> {
    primitive_to_scalar(bound.data_type(), bound.literal(), output_type)
}

/// Widens a float zero bound to include both zeros: -0.0 for a lower bound, +0.0 for an upper
/// one. A NaN bound is unknown. Other values are returned unchanged.
fn widen_float_zero(value: ScalarValue, lower: bool) -> Option<ScalarValue> {
    match value {
        ScalarValue::Float32(Some(v)) if v.is_nan() => None,
        ScalarValue::Float64(Some(v)) if v.is_nan() => None,
        ScalarValue::Float32(Some(0.0)) => {
            Some(ScalarValue::Float32(Some(if lower { -0.0 } else { 0.0 })))
        }
        ScalarValue::Float64(Some(0.0)) => {
            Some(ScalarValue::Float64(Some(if lower { -0.0 } else { 0.0 })))
        }
        value => Some(value),
    }
}

/// [`PruningStatistics`] for one file's [`Statistics`]. Unlike [`PrunableStatistics`], which
/// only reads exact values, min and max may be inexact: pruning only needs them to bound the
/// file's values. Null and row counts are still read only when exact.
struct FileStatistics {
    statistics: Arc<Statistics>,
    schema: SchemaRef,
    exact_counts: PrunableStatistics,
}

impl FileStatistics {
    fn new(statistics: Statistics, schema: &SchemaRef) -> Self {
        let statistics = Arc::new(statistics);
        Self {
            exact_counts: PrunableStatistics::new(
                vec![Arc::clone(&statistics)],
                Arc::clone(schema),
            ),
            statistics,
            schema: Arc::clone(schema),
        }
    }

    fn bound(
        &self,
        column: &Column,
        get: impl Fn(&ColumnStatistics) -> &Precision<ScalarValue>,
    ) -> Option<ArrayRef> {
        let index = self.schema.index_of(column.name()).ok()?;
        let value = get(self.statistics.column_statistics.get(index)?).get_value()?;
        value.to_array().ok()
    }
}

impl PruningStatistics for FileStatistics {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, |stats| &stats.min_value)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, |stats| &stats.max_value)
    }

    fn num_containers(&self) -> usize {
        1
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        self.exact_counts.null_counts(column)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        self.exact_counts.row_counts()
    }

    // The trait's signature takes std's `HashSet`.
    #[allow(clippy::disallowed_types)]
    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
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
    // Iceberg puts -0 and +0, and every NaN, in one partition, so a zero or NaN partition
    // value doesn't bound the file's rows exactly: leave it unknown.
    match value {
        PrimitiveLiteral::Float(v) if v.0 == 0.0 || v.0.is_nan() => return None,
        PrimitiveLiteral::Double(v) if v.0 == 0.0 || v.0.is_nan() => return None,
        _ => {}
    }
    primitive_to_scalar(source_type, value, output_type)
}

/// Converts an Iceberg value of type `source_type` to a value of `output_type`, or `None` when
/// the types don't exactly correspond.
fn primitive_to_scalar(
    source_type: &PrimitiveType,
    value: &PrimitiveLiteral,
    output_type: &DataType,
) -> Option<ScalarValue> {
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
        (PrimitiveType::Binary, PrimitiveLiteral::Binary(v), DataType::Binary) => {
            ScalarValue::Binary(Some(v.clone()))
        }
        (PrimitiveType::Binary, PrimitiveLiteral::Binary(v), DataType::LargeBinary) => {
            ScalarValue::LargeBinary(Some(v.clone()))
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

// Iceberg's column statistics are built from std's `HashMap`.
#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{
        BinaryExpr, CaseExpr, Column, DynamicFilterPhysicalExpr, lit,
    };
    use iceberg::arrow::schema_to_arrow_schema;
    use iceberg::scan::FileScanTaskColumnStats;
    use iceberg::spec::{DataFileFormat, NestedField, PartitionSpec, Schema, Struct};
    use std::collections::HashMap;

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

    #[test]
    fn column_bounds_are_inexact_min_and_max() {
        let stats = FileScanTaskColumnStats::builder()
            .with_lower_bounds(HashMap::from([
                (1, Datum::int(3)),
                (2, Datum::date(19730)),
                (3, Datum::string("apple")),
            ]))
            .with_upper_bounds(HashMap::from([
                (1, Datum::int(9)),
                (2, Datum::date(19736)),
                (3, Datum::string("melon")),
            ]))
            .with_null_value_counts(HashMap::from([(1, 0), (3, 2)]))
            .build();

        insta::assert_snapshot!(describe(&task_with_stats(stats), &output_schema()), @"
        rows=Exact(10)
        id: min=Inexact(Int32(3)) max=Inexact(Int32(9)) nulls=Exact(0)
        day: min=Inexact(Date32(\"2024-01-08\")) max=Inexact(Date32(\"2024-01-14\")) nulls=Absent
        name: min=Inexact(Utf8(\"apple\")) max=Inexact(Utf8(\"melon\")) nulls=Exact(2)
        score: unknown
        ");
    }

    /// Iceberg truncates long string bounds: the lower bound to a prefix, the upper bound to a
    /// prefix with its last character incremented. Both still bound the values, so a filter
    /// on a value between them keeps the file and one past them skips it.
    #[test]
    fn truncated_string_bounds_still_bound_the_values() {
        // Values from "carbon-long" to "carton-long", truncated to 3 characters.
        let stats = FileScanTaskColumnStats::builder()
            .with_lower_bounds(HashMap::from([(3, Datum::string("car"))]))
            .with_upper_bounds(HashMap::from([(3, Datum::string("cas"))]))
            .build();
        let task = task_with_stats(stats);
        let name: Arc<dyn PhysicalExpr> = Arc::new(Column::new("name", 2));
        let skips = |value: &str| {
            let filter = Arc::new(DynamicFilterPhysicalExpr::new(
                vec![Arc::clone(&name)],
                Arc::new(BinaryExpr::new(Arc::clone(&name), Operator::Eq, lit(value))),
            ));
            let pruner = pruner(&[filter as _], &output_schema());
            format!("name = {value}: skip={}", pruner.should_skip(&task))
        };

        insta::assert_snapshot!(
            ["carbon-long", "carton-long", "cast-long", "cap"].map(skips).join("\n"),
            @"
        name = carbon-long: skip=false
        name = carton-long: skip=false
        name = cast-long: skip=true
        name = cap: skip=true
        "
        );
    }

    /// The output reads `id` as Int64, which doesn't match the Iceberg Int bounds.
    #[test]
    fn column_bounds_of_another_type_are_left_unknown() {
        let stats = FileScanTaskColumnStats::builder()
            .with_lower_bounds(HashMap::from([(1, Datum::int(3))]))
            .with_upper_bounds(HashMap::from([(1, Datum::int(9))]))
            .build();
        let mut fields: Vec<Field> = output_schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields[0] = fields[0].clone().with_data_type(DataType::Int64);
        let output = Arc::new(ArrowSchema::new(fields));

        insta::assert_snapshot!(describe(&task_with_stats(stats), &output), @"
        rows=Exact(10)
        id: unknown
        day: unknown
        name: unknown
        score: unknown
        ");
    }

    /// Missing bounds are unknown, not empty; a column that is all nulls may have no bounds
    /// and is described by its null count alone.
    #[test]
    fn missing_bounds_are_left_unknown() {
        let stats = FileScanTaskColumnStats::builder()
            .with_null_value_counts(HashMap::from([(3, 10)]))
            .build();

        insta::assert_snapshot!(describe(&task_with_stats(stats), &output_schema()), @"
        rows=Exact(10)
        id: unknown
        day: unknown
        name: min=Absent max=Absent nulls=Exact(10)
        score: unknown
        ");
    }

    /// Float bounds exclude NaN, so they are used only when the file is known to hold none.
    /// Zero bounds are widened to cover both -0.0 and +0.0.
    #[test]
    fn float_bounds_need_a_zero_nan_count_and_widen_zeros() {
        let describe_score = |lower: f64, upper: f64, nans: Option<u64>| {
            let stats = FileScanTaskColumnStats::builder()
                .with_lower_bounds(HashMap::from([(4, Datum::double(lower))]))
                .with_upper_bounds(HashMap::from([(4, Datum::double(upper))]))
                .with_nan_value_counts(nans.map(|n| HashMap::from([(4, n)])).unwrap_or_default())
                .build();
            let line = describe(&task_with_stats(stats), &output_schema())
                .lines()
                .find(|line| line.starts_with("score"))
                .unwrap()
                .to_string();
            format!("[{lower:?}, {upper:?}] nans={nans:?} -> {line}")
        };

        insta::assert_snapshot!(
            [
                describe_score(1.5, 2.5, Some(0)),
                describe_score(1.5, 2.5, Some(3)),
                describe_score(1.5, 2.5, None),
                describe_score(0.0, -0.0, Some(0)),
                describe_score(-0.0, 0.0, Some(0)),
            ]
            .join("\n"),
            @"
        [1.5, 2.5] nans=Some(0) -> score: min=Inexact(Float64(1.5)) max=Inexact(Float64(2.5)) nulls=Absent
        [1.5, 2.5] nans=Some(3) -> score: unknown
        [1.5, 2.5] nans=None -> score: unknown
        [0.0, -0.0] nans=Some(0) -> score: min=Inexact(Float64(-0)) max=Inexact(Float64(0)) nulls=Absent
        [-0.0, 0.0] nans=Some(0) -> score: min=Inexact(Float64(-0)) max=Inexact(Float64(0)) nulls=Absent
        "
        );
    }

    /// An identity partition value is exact, so it replaces the column's bounds; other
    /// columns keep their bounds.
    #[test]
    fn partition_values_take_precedence_over_column_bounds() {
        let stats = FileScanTaskColumnStats::builder()
            .with_lower_bounds(HashMap::from([(1, Datum::int(3)), (2, Datum::date(19700))]))
            .with_upper_bounds(HashMap::from([(1, Datum::int(9)), (2, Datum::date(19800))]))
            .build();
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(100)
            .with_start(0)
            .with_length(100)
            .with_record_count(Some(10))
            .with_data_file_path("file:///t/data/f.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(table_schema())
            .with_project_field_ids(vec![1, 2, 3, 4])
            .with_partition(Some(Struct::from_iter([Some(Literal::date(19732))])))
            .with_partition_spec(Some(spec(&[("day", Transform::Identity)])))
            .with_column_stats(Some(stats))
            .with_case_sensitive(true)
            .build()
            .unwrap();

        insta::assert_snapshot!(describe(&task, &output_schema()), @r#"
        rows=Exact(10)
        id: min=Inexact(Int32(3)) max=Inexact(Int32(9)) nulls=Absent
        day: min=Exact(Date32("2024-01-10")) max=Exact(Date32("2024-01-10")) nulls=Exact(0)
        name: unknown
        score: unknown
        "#);
    }

    #[test]
    fn requests_stats_for_projected_primitive_columns() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "tags",
                    Type::List(iceberg::spec::ListType::new(
                        NestedField::list_element(3, Type::Primitive(PrimitiveType::String), false)
                            .into(),
                    )),
                )
                .into(),
                NestedField::optional(4, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        assert_eq!(column_stats_columns(&schema, None), vec!["id", "name"]);
        assert_eq!(
            column_stats_columns(
                &schema,
                Some(&["name".to_string(), "tags".to_string(), "_file".to_string()])
            ),
            vec!["name"]
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

    #[test]
    fn pruner_rebuilds_its_predicate_only_when_the_filter_changes() {
        let schema = output_schema();
        let day = |d: i32| lit(ScalarValue::Date32(Some(d)));
        let day_col: Arc<dyn PhysicalExpr> = Arc::new(Column::new("day", 1));
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&day_col)],
            lit(true),
        ));
        let pruner = pruner(&[Arc::clone(&dynamic) as _], &schema);
        let file = task(
            spec(&[("day", Transform::Identity)]),
            [Some(Literal::date(19732))],
        );
        let mut decisions = vec![format!(
            "before any update: skip={}",
            pruner.should_skip(&file)
        )];

        dynamic
            .update(Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::clone(&day_col),
                    Operator::GtEq,
                    day(19730),
                )),
                Operator::And,
                Arc::new(BinaryExpr::new(
                    Arc::clone(&day_col),
                    Operator::LtEq,
                    day(19740),
                )),
            )))
            .unwrap();
        decisions.push(format!(
            "after a range covering the file: skip={}",
            pruner.should_skip(&file)
        ));

        // A tighter routing CASE whose only branch holds another day: the file can't match.
        let routing: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                Some(Arc::clone(&day_col)),
                vec![(
                    day(19740),
                    Arc::new(BinaryExpr::new(
                        Arc::clone(&day_col),
                        Operator::Eq,
                        day(19740),
                    )),
                )],
                Some(lit(false)),
            )
            .unwrap(),
        );
        dynamic.update(routing).unwrap();
        let first = pruner.pruning_predicate().unwrap();
        decisions.push(format!(
            "after an update: skip={}",
            pruner.should_skip(&file)
        ));
        // Same generation: the cached predicate is reused, not rebuilt.
        assert!(Arc::ptr_eq(&first, &pruner.pruning_predicate().unwrap()));

        insta::assert_snapshot!(decisions.join("\n"), @"
        before any update: skip=false
        after a range covering the file: skip=false
        after an update: skip=true
        ");
    }

    /// A pruner over the dynamic filters among `filters`, as a scan installs it.
    fn pruner(filters: &[Arc<dyn PhysicalExpr>], schema: &SchemaRef) -> PartitionFilePruner {
        let compacted = CompactedDynamicFilters::try_new(filters, schema, 150).unwrap();
        PartitionFilePruner::new(Arc::new(compacted), schema)
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

    fn task_with_stats(stats: FileScanTaskColumnStats) -> FileScanTask {
        FileScanTask::builder()
            .with_file_size_in_bytes(100)
            .with_start(0)
            .with_length(100)
            .with_record_count(Some(10))
            .with_data_file_path("file:///t/data/f.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(table_schema())
            .with_project_field_ids(vec![1, 2, 3, 4])
            .with_column_stats(Some(stats))
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
