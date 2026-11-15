#[cfg(test)]
mod tests {
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::common::Result;
    use datafusion::physical_plan::collect;
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::test_utils::metrics::node_metrics;
    use datafusion_distributed::test_utils::parquet::register_parquet_tables;
    use datafusion_distributed::{
        DefaultSessionBuilder, DistributedExt, DistributedLeafExec, DistributedMetricsFormat,
        assert_snapshot, rewrite_distributed_plan_with_metrics,
    };
    use std::sync::Arc;

    /// In a partitioned join, the probe scan runs in a different stage than the join. It waits
    /// for the join's filter before it starts, so it only returns the rows the filter keeps.
    #[tokio::test]
    async fn the_probe_scan_waits_for_the_join_filter() -> Result<()> {
        let (ctx, _guard, _) = start_localhost_context(2, DefaultSessionBuilder).await;
        let ctx = ctx.with_distributed_broadcast_joins(false)?;
        {
            let state = ctx.state_ref();
            let mut state = state.write();
            let options = state.config_mut().options_mut();
            // Force a partitioned join.
            options.optimizer.hash_join_single_partition_threshold = 0;
            options.optimizer.hash_join_single_partition_threshold_rows = 0;
            // Check the filter against each row, so the scan's output rows show whether the scan
            // had the filter.
            options.execution.parquet.pushdown_filters = true;
            options.set("distributed.dynamic_filter_wait_timeout_ms", "60000")?;
        }
        register_parquet_tables(&ctx).await?;

        // One day has more than 30 mm of rain. Its MinTemp appears on 4 of the 366 days.
        let plan = ctx
            .sql(
                r#"
                SELECT probe."MinTemp", COUNT(*) AS days
                FROM (SELECT DISTINCT "MinTemp" AS key FROM weather WHERE "Rainfall" > 30) build
                JOIN weather probe ON build.key = probe."MinTemp"
                GROUP BY probe."MinTemp"
                "#,
            )
            .await?
            .create_physical_plan()
            .await?;
        let results = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        let results = pretty_format_batches(&results)?.to_string();
        let plan =
            rewrite_distributed_plan_with_metrics(plan, DistributedMetricsFormat::Aggregated)
                .await?;

        assert_snapshot!(results, @"
        +---------+------+
        | MinTemp | days |
        +---------+------+
        | 13.3    | 4    |
        +---------+------+
        ");
        // `DistributedLeafExec` 0 is the build scan, and 1 is the probe scan. Without the filter,
        // the probe scan returns all 366 rows.
        assert_eq!(
            node_metrics::<DistributedLeafExec>(&plan, "output_rows", 1),
            4
        );
        Ok(())
    }
}
