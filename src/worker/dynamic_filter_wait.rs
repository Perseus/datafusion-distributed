use crate::dynamic_filtering::discover_dynamic_filter_consumers;
use datafusion::common::Result;
use datafusion::common::instant::Instant;
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use futures::future::join_all;
use std::sync::Arc;
use std::time::Duration;

/// The dynamic filters a task waits for before it starts: the ones it consumes that a hash join
/// in another task produces. See [crate::DistributedConfig::dynamic_filter_wait_timeout_ms].
#[derive(Clone, Debug)]
pub(super) struct DynamicFilterWait {
    filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
    timeout: Duration,
}

impl DynamicFilterWait {
    /// Returns a wait for the filters in `plan` whose IDs are in `ids`. Returns `None` when there
    /// is nothing to wait for.
    pub(super) fn try_new(
        plan: &Arc<dyn ExecutionPlan>,
        ids: &[u64],
        timeout: Duration,
    ) -> Result<Option<Self>> {
        if ids.is_empty() || timeout.is_zero() {
            return Ok(None);
        }
        let filters: Vec<_> = discover_dynamic_filter_consumers(plan)?
            .consumers
            .into_iter()
            .filter(|consumer| ids.contains(&consumer.id))
            .map(|consumer| consumer.expression)
            .collect();
        Ok((!filters.is_empty()).then_some(Self { filters, timeout }))
    }

    /// Waits until every filter is complete, or until the timeout passes. Returns how long it
    /// waited.
    pub(super) async fn wait(&self) -> Duration {
        let start = Instant::now();
        let all_complete = join_all(self.filters.iter().map(|filter| filter.wait_complete()));
        let _ = tokio::time::timeout(self.timeout, all_complete).await;
        start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::physical_expr::expressions::lit;
    use futures::poll;
    use std::pin::pin;

    #[tokio::test]
    async fn stops_waiting_once_every_filter_is_complete() {
        let (first, second) = (empty_filter(), empty_filter());
        let wait = DynamicFilterWait {
            filters: vec![Arc::clone(&first), Arc::clone(&second)],
            timeout: Duration::from_secs(60),
        };

        let mut waiting = pin!(wait.wait());
        assert!(poll!(&mut waiting).is_pending());
        first.mark_complete();
        assert!(poll!(&mut waiting).is_pending());
        second.mark_complete();

        let waited = waiting.await;
        assert!(waited < Duration::from_secs(60), "waited {waited:?}");
    }

    #[tokio::test]
    async fn stops_waiting_at_the_timeout() {
        let (complete, incomplete) = (empty_filter(), empty_filter());
        complete.mark_complete();
        let wait = DynamicFilterWait {
            filters: vec![complete, incomplete],
            timeout: Duration::from_millis(50),
        };

        let waited = wait.wait().await;

        assert!(waited >= Duration::from_millis(50), "waited {waited:?}");
    }

    fn empty_filter() -> Arc<DynamicFilterPhysicalExpr> {
        Arc::new(DynamicFilterPhysicalExpr::new(vec![], lit(true)))
    }
}
