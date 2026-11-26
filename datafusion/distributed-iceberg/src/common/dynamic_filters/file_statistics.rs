use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use datafusion::common::Column;
use datafusion::common::pruning::PruningStatistics;
use datafusion::scalar::ScalarValue;
use iceberg::scan::{FileScanTask, FileScanTaskColumnStats};
use iceberg::spec::{Datum, PrimitiveType};

use crate::table_provider::datum_to_scalar;

/// The column stats of one file task, in the form DataFusion's `PruningPredicate` reads. The
/// task holds stats only for the columns the scan asked iceberg-rust to keep.
pub(crate) struct FileStatistics<'a> {
    task: &'a FileScanTask,
    stats: &'a FileScanTaskColumnStats,
}

impl<'a> FileStatistics<'a> {
    /// `None` when the task has no column stats.
    pub(crate) fn try_new(task: &'a FileScanTask) -> Option<Self> {
        Some(Self {
            task,
            stats: task.column_stats()?,
        })
    }

    fn field_id(&self, column: &Column) -> Option<i32> {
        self.task.schema().field_id_by_name(&column.name)
    }

    /// The file's lower or upper bound for `column`, as a one-row array.
    fn bound(&self, column: &Column, side: Side) -> Option<ArrayRef> {
        let id = self.field_id(column)?;
        let bound = match side {
            Side::Lower => self.stats.lower_bounds().get(&id)?,
            Side::Upper => self.stats.upper_bounds().get(&id)?,
        };
        if is_float(bound) {
            // Iceberg bounds leave out NaN, but DataFusion sorts NaN above every number. So a
            // file with a NaN can hold a value above its upper bound, and we only use float
            // bounds when the file has no NaN.
            if self.stats.nan_value_counts().get(&id) != Some(&0) {
                return None;
            }
        }
        let value = widen_zero(datum_to_scalar(bound)?, side);
        value.to_array().ok()
    }
}

#[derive(Clone, Copy)]
enum Side {
    Lower,
    Upper,
}

fn is_float(datum: &Datum) -> bool {
    matches!(
        datum.data_type(),
        PrimitiveType::Float | PrimitiveType::Double
    )
}

/// Writers don't agree on the sign of a zero bound, and DataFusion orders -0.0 below 0.0. So a
/// zero lower bound becomes -0.0 and a zero upper bound becomes 0.0, which covers both zeros.
fn widen_zero(value: ScalarValue, side: Side) -> ScalarValue {
    let zero = match side {
        Side::Lower => -0.0,
        Side::Upper => 0.0,
    };
    match value {
        // A float pattern compares with `==`, so `0.0` also matches -0.0.
        ScalarValue::Float32(Some(0.0)) => ScalarValue::Float32(Some(zero as f32)),
        ScalarValue::Float64(Some(0.0)) => ScalarValue::Float64(Some(zero)),
        value => value,
    }
}

impl PruningStatistics for FileStatistics<'_> {
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
        let count = self
            .stats
            .null_value_counts()
            .get(&self.field_id(column)?)?;
        Some(Arc::new(UInt64Array::from(vec![*count])))
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        let count = self.task.record_count()?;
        Some(Arc::new(UInt64Array::from(vec![count])))
    }

    #[allow(
        clippy::disallowed_types,
        reason = "DataFusion's `PruningStatistics` takes a std `HashSet`"
    )]
    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}
