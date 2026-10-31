use std::path::Path;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_distributed_benchmarks::backend::BenchmarkBackend;
use datafusion_distributed_iceberg::{IcebergExt, IcebergIntegrationOptions};

use crate::dataset;

/// Iceberg registration and session extensions for local and remote benchmarks.
#[derive(Clone)]
pub struct IcebergBenchmarkBackend;

#[async_trait]
impl BenchmarkBackend for IcebergBenchmarkBackend {
    async fn register_tables(&mut self, ctx: &SessionContext, path: &Path) -> Result<()> {
        dataset::register_tables(ctx, path).await
    }

    fn configure_session(&self, builder: SessionStateBuilder) -> SessionStateBuilder {
        let mut builder = builder
            .with_iceberg_integration(IcebergIntegrationOptions::default())
            .with_iceberg_column_stats_enabled(true);
        // BENCH PATCH: opt-in iceberg.* options via env vars; unset = unchanged.
        let options = builder.config().get_or_insert_default().options_mut();
        for (env, key) in [
            (
                "DFBENCH_ICEBERG_DYNAMIC_FILTER_PRUNING",
                "iceberg.dynamic_filter_pruning_enabled",
            ),
            (
                "DFBENCH_ICEBERG_ROW_SELECTION",
                "iceberg.row_selection_enabled",
            ),
        ] {
            if let Ok(value) = std::env::var(env) {
                options
                    .set(key, &value)
                    .expect("invalid iceberg option value");
            }
        }
        builder
    }
}
