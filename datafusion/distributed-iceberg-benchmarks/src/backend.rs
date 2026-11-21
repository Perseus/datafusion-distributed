use std::path::Path;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_distributed::{DistributedExt, WorkerQueryContext};
use datafusion_distributed_benchmarks::backend::BenchmarkBackend;
use datafusion_distributed_iceberg::{IcebergConfig, IcebergExt, IcebergIntegrationOptions};

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
        builder
            // Workers read data files with the `iceberg.*` options of their own session. We
            // register the options here, so that the coordinator sends its values with every
            // query.
            .with_distributed_option_extension(IcebergConfig::default())
            .with_iceberg_integration(IcebergIntegrationOptions::default())
            .with_iceberg_column_stats_enabled(true)
    }

    fn configure_worker_session(&self, ctx: WorkerQueryContext) -> Result<SessionStateBuilder> {
        Ok(ctx
            .builder
            .with_distributed_option_extension_from_headers::<IcebergConfig>(&ctx.headers)?
            .with_iceberg_integration(IcebergIntegrationOptions::default()))
    }
}
