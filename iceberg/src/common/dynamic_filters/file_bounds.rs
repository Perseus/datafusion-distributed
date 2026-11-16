#[allow(
    clippy::disallowed_types,
    reason = "PruningStatistics::contained takes a std HashSet"
)]
use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::Column;
use datafusion::common::pruning::PruningStatistics;
use datafusion::scalar::ScalarValue;
use iceberg::scan::{FileScanTask, FileScanTaskColumnStats};

use crate::table_provider::column_statistics::datum_to_scalar;

/// The column bounds of one data file, as recorded in the table's manifests, seen as DataFusion
/// pruning statistics for a single container.
///
/// Every value in the file lies within its bounds, but a bound need not be a value from the file.
/// Writers may truncate long strings, for example. So the bounds can prove that a file has no
/// matching rows, but they are not exact statistics.
pub(crate) struct FileBounds<'a> {
    /// The scan's output schema, which the filters' columns refer to.
    schema: &'a Schema,
    task: &'a FileScanTask,
    stats: &'a FileScanTaskColumnStats,
}

impl<'a> FileBounds<'a> {
    /// The bounds of `task`'s file, or `None` if the task carries no column statistics.
    pub(crate) fn new(schema: &'a Schema, task: &'a FileScanTask) -> Option<Self> {
        Some(Self {
            schema,
            task,
            stats: task.column_stats()?,
        })
    }

    /// The lower or upper bound of `column`, as a one-row array of the column's type.
    fn bound(&self, column: &Column, side: Side) -> Option<ArrayRef> {
        let data_type = self.schema.field_with_name(&column.name).ok()?.data_type();
        let id = self.field_id(column)?;
        // Iceberg leaves NaN out of float bounds, but DataFusion orders NaN above every number.
        // The bounds then only hold every value when the file has no NaN.
        if matches!(
            data_type,
            DataType::Float16 | DataType::Float32 | DataType::Float64
        ) && self.stats.nan_value_counts().get(&id) != Some(&0)
        {
            return None;
        }
        let bounds = match side {
            Side::Lower => self.stats.lower_bounds(),
            Side::Upper => self.stats.upper_bounds(),
        };
        let bound = datum_to_scalar(bounds.get(&id)?)?;
        cast(bound, data_type)?.to_array().ok()
    }

    fn field_id(&self, column: &Column) -> Option<i32> {
        Some(self.task.schema().field_by_name(&column.name)?.id)
    }
}

impl PruningStatistics for FileBounds<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, Side::Lower)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, Side::Upper)
    }

    fn num_containers(&self) -> usize {
        1
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let count = *self
            .stats
            .null_value_counts()
            .get(&self.field_id(column)?)?;
        Some(Arc::new(UInt64Array::from(vec![count])))
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        Some(Arc::new(UInt64Array::from(vec![self.task.record_count()?])))
    }

    #[allow(
        clippy::disallowed_types,
        reason = "PruningStatistics::contained takes a std HashSet"
    )]
    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

#[derive(Clone, Copy)]
enum Side {
    Lower,
    Upper,
}

/// `value` as `data_type`. Scan columns can use another Arrow type for the same Iceberg type,
/// like a string view for a string.
fn cast(value: ScalarValue, data_type: &DataType) -> Option<ScalarValue> {
    if value.data_type() == *data_type {
        return Some(value);
    }
    value.cast_to(data_type).ok()
}
