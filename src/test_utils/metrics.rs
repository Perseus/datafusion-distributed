use crate::coordinator::DistributedExec;
use chrono::{DateTime, Utc};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::{Count, Metric, MetricValue, MetricsSet, Time, Timestamp};
use std::sync::Arc;

/// Waits until all worker tasks have reported their metrics back via the coordinator channel.
pub async fn wait_for_all_metrics(plan: &Arc<dyn ExecutionPlan>) {
    if let Some(dist_exec) = plan.downcast_ref::<DistributedExec>() {
        let _ = dist_exec.wait_for_metrics().await;
    }
}

/// creates a "distinct" set of metrics from the provided seed
pub fn make_test_metrics_set_from_seed(seed: u64, num_metrics: usize) -> MetricsSet {
    const TEST_TIMESTAMP: i64 = 1758200400000000000; // 2025-09-18 13:00:00 UTC

    let mut result = MetricsSet::new();

    for i in 0..num_metrics {
        let value = (seed + i as u64) as usize;
        result.push(Arc::new(Metric::new(
            match i % 4 {
                0 => {
                    let count = Count::new();
                    count.add(value);
                    MetricValue::OutputRows(count)
                }
                1 => {
                    let time = Time::new();
                    time.add_duration(std::time::Duration::from_nanos(value as u64));
                    MetricValue::ElapsedCompute(time)
                }
                2 => MetricValue::StartTimestamp(timestamp_from_nanos(
                    TEST_TIMESTAMP + (value as i64 * 1_000_000_000),
                )),
                3 => MetricValue::EndTimestamp(timestamp_from_nanos(
                    TEST_TIMESTAMP + (value as i64 * 1_000_000_000),
                )),
                _ => unreachable!(),
            },
            None,
        )))
    }
    result
}

fn timestamp_from_nanos(nanos: i64) -> Timestamp {
    let timestamp = Timestamp::new();
    timestamp.set(DateTime::<Utc>::from_timestamp_nanos(nanos));
    timestamp
}

/// Sums the values of the metrics named `metric_name` on the `index`th node of type `T`.
pub fn node_metrics<T: ExecutionPlan + 'static>(
    plan: &Arc<dyn ExecutionPlan>,
    metric_name: &str,
    mut index: usize,
) -> usize {
    use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
    let mut metrics = None;
    Arc::clone(plan)
        .transform_down(|plan| {
            if plan.name() == T::static_name() {
                metrics = plan.metrics();
                if index == 0 {
                    return Ok(Transformed::new(plan, false, TreeNodeRecursion::Stop));
                }
                index -= 1;
            }
            Ok(Transformed::no(plan))
        })
        .unwrap();
    let metrics =
        metrics.unwrap_or_else(|| panic!("Could not find metrics for plan {}", T::static_name()));
    let summed = metrics
        .iter()
        .filter(|v| v.value().name().starts_with(metric_name))
        .map(|v| v.value().as_usize())
        .sum();
    assert!(
        summed > 0,
        "Sum of metric values is 0. Either the metric {metric_name} is not present or the test is too trivial"
    );
    summed
}
