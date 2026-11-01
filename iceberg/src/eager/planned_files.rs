use std::mem::{size_of, size_of_val};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::common::{Result, Statistics};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use futures::TryStreamExt;
use iceberg::scan::{FileScanTask, FileScanTaskColumnStats, TableScan};
use iceberg::spec::{Datum, PrimitiveLiteral};

use crate::common::df_err;

/// File work retained by a query after manifest planning and before execution.
/// Shared by task-local sources so their descriptors and reservation have one owner.
#[derive(Debug)]
pub(crate) struct PlannedFiles {
    tasks: Vec<FileScanTask>,
    _reservation: MemoryReservation,
}

impl PlannedFiles {
    pub(crate) fn tasks(&self) -> &[FileScanTask] {
        &self.tasks
    }

    /// Summarize selected file work; predicates and deletes can reduce output rows.
    pub(crate) fn statistics(&self, schema: &SchemaRef, may_filter_rows: bool) -> Statistics {
        let bytes = self.tasks.iter().fold(0usize, |bytes, task| {
            bytes.saturating_add(task.length() as usize)
        });
        let rows = self.tasks.iter().try_fold(0usize, |rows, task| {
            rows.checked_add(usize::try_from(task.record_count()?).ok()?)
        });
        let mut statistics = Statistics::new_unknown(schema);
        statistics.num_rows = match rows {
            Some(0) if self.tasks.is_empty() => Precision::Exact(0),
            Some(rows) if !may_filter_rows => Precision::Exact(rows),
            Some(rows) => Precision::Inexact(rows),
            None => Precision::Absent,
        };
        statistics.total_byte_size = if may_filter_rows {
            Precision::Inexact(bytes)
        } else {
            Precision::Exact(bytes)
        };
        statistics
    }

    fn new(context: &TaskContext) -> Self {
        Self {
            tasks: Vec::new(),
            _reservation: MemoryConsumer::new("Iceberg file planning")
                .register(context.memory_pool()),
        }
    }

    fn push(&mut self, task: FileScanTask) -> Result<()> {
        // Include owned paths/delete references and allow room for Vec capacity.
        self._reservation
            .try_grow(retained_size(&task).saturating_mul(2))?;
        self.tasks.push(task);
        Ok(())
    }

    pub(crate) fn from_tasks(tasks: Vec<FileScanTask>, context: &TaskContext) -> Result<Self> {
        let mut planned = Self::new(context);
        for task in tasks {
            planned.push(task)?;
        }
        Ok(planned)
    }

    pub(crate) async fn collect(scan: TableScan, context: &TaskContext) -> Result<Self> {
        let mut planned = Self::new(context);
        let mut stream = scan.plan_files().await.map_err(df_err)?;
        while let Some(task) = stream.try_next().await.map_err(df_err)? {
            planned.push(task)?;
        }
        Ok(planned)
    }
}

fn retained_size(task: &FileScanTask) -> usize {
    let mut bytes = size_of::<FileScanTask>()
        + task.data_file_path().len()
        + size_of_val(task.project_field_ids())
        + size_of_val(task.deletes())
        + task.key_metadata().map_or(0, <[u8]>::len);
    for delete in task.deletes() {
        bytes = bytes
            .saturating_add(delete.file_path().len())
            .saturating_add(delete.referenced_data_file().map_or(0, str::len))
            .saturating_add(delete.equality_ids().map_or(0, size_of_val))
            .saturating_add(delete.key_metadata().map_or(0, <[u8]>::len));
    }
    if let Some(stats) = task.column_stats() {
        bytes = bytes.saturating_add(column_stats_size(stats));
    }
    bytes
}

/// The boxed per-file column statistics: their maps' entries and the bounds' string and binary
/// payloads. Map overhead beyond the entries is covered by the caller's capacity allowance.
fn column_stats_size(stats: &FileScanTaskColumnStats) -> usize {
    let counts = [
        stats.column_sizes(),
        stats.value_counts(),
        stats.null_value_counts(),
        stats.nan_value_counts(),
    ]
    .iter()
    .map(|counts| counts.len() * size_of::<(i32, u64)>())
    .sum::<usize>();
    let bounds = [stats.lower_bounds(), stats.upper_bounds()]
        .iter()
        .flat_map(|bounds| bounds.values())
        .map(|bound| {
            size_of::<(i32, Datum)>()
                + match bound.literal() {
                    PrimitiveLiteral::String(s) => s.len(),
                    PrimitiveLiteral::Binary(b) => b.len(),
                    _ => 0,
                }
        })
        .sum::<usize>();
    size_of::<FileScanTaskColumnStats>() + counts + bounds
}

// Iceberg's column statistics are built from std's `HashMap`.
#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use iceberg::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};

    use super::*;

    /// Planned files are retained until execution, so the column statistics they carry for
    /// skipping files by bounds must count against the memory limit. Under a 64 KiB limit the
    /// file task alone fits, but with 100 KB string bounds it is rejected.
    #[test]
    fn column_stats_count_against_the_memory_limit() -> Result<()> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(64 * 1024)))
            .build_arc()?;
        let context = TaskContext::default().with_runtime(runtime);
        let long = "x".repeat(50_000);
        let stats = FileScanTaskColumnStats::builder()
            .with_lower_bounds(HashMap::from([(1, Datum::string(&long))]))
            .with_upper_bounds(HashMap::from([(1, Datum::string(&long))]))
            .build();

        let without_stats = PlannedFiles::from_tasks(vec![task(None)], &context);
        let with_stats = PlannedFiles::from_tasks(vec![task(Some(stats))], &context);

        assert!(without_stats.is_ok());
        insta::assert_snapshot!(
            with_stats.unwrap_err().to_string().lines().next().unwrap_or_default(),
            @"Resources exhausted: Failed to allocate additional 196.9 KB for Iceberg file planning with 0.0 B already allocated for this reservation - 63.2 KB remain available for the total memory pool: greedy(used: 792.0 B, pool_size: 64.0 KB)"
        );
        Ok(())
    }

    fn task(stats: Option<FileScanTaskColumnStats>) -> FileScanTask {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        FileScanTask::builder()
            .with_file_size_in_bytes(100)
            .with_start(0)
            .with_length(100)
            .with_record_count(Some(10))
            .with_data_file_path("file:///t/data/f.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(Arc::new(schema))
            .with_project_field_ids(vec![1])
            .with_column_stats(stats)
            .with_case_sensitive(true)
            .build()
            .unwrap()
    }
}
