use crate::ExecuteTaskRequest;
use crate::worker::worker_service::Worker;
use datafusion::common::exec_datafusion_err;
use datafusion::common::{Result, exec_err};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{TryStreamExt, stream};
use std::sync::Arc;
use std::time::Duration;

const WAIT_PLAN_TIMEOUT_SECS: u64 = 10;

/// Builds several per-partition streams by retrieving the appropriate entry from [TaskDataEntries]
/// based on the task key extracted from [ExecuteTaskRequest].
///
/// This method is async mainly for the key retrieval operation from [TaskDataEntries], but it does
/// not start polling any stream, it just instantiates them.
impl Worker {
    pub async fn execute_task(
        &self,
        request: ExecuteTaskRequest,
    ) -> Result<(Vec<SendableRecordBatchStream>, Arc<TaskContext>)> {
        let entry = self
            .task_data_entries
            .get_with(request.task_key, async { Default::default() })
            .await;

        // Other request is responsible for writing the plan that belongs to this TaskKey, so
        // we'll resolve immediately if it was already there, or wait until it's ready.
        let task_data = entry
            .read(Duration::from_secs(WAIT_PLAN_TIMEOUT_SECS))
            .await
            .map_err(|e| exec_datafusion_err!("Worker::execute_task timed-out while waiting for the plan to be set by the coordinator. ({e})"))?
            .map_err(DataFusionError::Shared)?;
        task_data.task_data_metrics.mark_execution_started_once();

        let plan = task_data.plan(request.producer_head)?;
        let task_ctx = task_data.task_ctx;
        let dynamic_filter_wait = task_data.dynamic_filter_wait;
        let task_data_metrics = task_data.task_data_metrics;
        let partition_count = plan.properties().partitioning.partition_count();
        let plan_name = plan.name();

        // Execute all the requested partitions at once, and collect all the streams so that they
        // can be merged into a single one at the end of this function.
        let n_streams = request.target_partition_end - request.target_partition_start;
        let mut streams = Vec::with_capacity(n_streams);
        for partition in request.target_partition_start..request.target_partition_end {
            if partition >= partition_count {
                return exec_err!(
                    "partition {partition} not available. The head plan {plan_name} of the stage just has {partition_count} partitions"
                );
            }

            let stream = match &dynamic_filter_wait {
                None => plan.execute(partition, Arc::clone(&task_ctx))?,
                // Execute the partition only after the wait. Some nodes start reading their input
                // as soon as they are executed.
                Some(wait) => {
                    let wait = wait.clone();
                    let deferred_plan = Arc::clone(&plan);
                    let task_ctx = Arc::clone(&task_ctx);
                    let task_data_metrics = Arc::clone(&task_data_metrics);
                    let stream = stream::once(async move {
                        let waited = wait.wait().await;
                        task_data_metrics
                            .dynamic_filter_wait_time
                            .add_duration(waited);
                        deferred_plan.execute(partition, task_ctx)
                    })
                    .try_flatten();
                    Box::pin(RecordBatchStreamAdapter::new(plan.schema(), stream))
                }
            };
            let stream_schema = plan.schema();
            let (dbg_stage, dbg_task) = (request.task_key.stage_id, request.task_key.task_number);
            let dbg_rows = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let dbg_rows_in = Arc::clone(&dbg_rows);
            let stream = futures::StreamExt::chain(
                futures::StreamExt::inspect(stream, move |batch| {
                    if let Ok(batch) = batch {
                        dbg_rows_in.fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);
                    }
                }),
                futures::StreamExt::filter_map(stream::once(async move {
                    eprintln!(
                        "DFDBG {} partition_done stage={} task={} partition={} rows={}",
                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0), dbg_stage, dbg_task, partition,
                        dbg_rows.load(std::sync::atomic::Ordering::Relaxed)
                    );
                }), |_| async { None }),
            );

            streams.push(Box::pin(RecordBatchStreamAdapter::new(stream_schema, stream)) as _);
        }
        Ok((streams, task_ctx))
    }
}
