use std::path::Path;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;

use datafusion_datasets::register_tables;
use datafusion_distributed::WorkerQueryContext;

#[async_trait]
pub trait BenchmarkBackend: Send + Sync + 'static {
    async fn register_tables(&mut self, ctx: &SessionContext, path: &Path) -> Result<()>;

    fn configure_session(&self, builder: SessionStateBuilder) -> SessionStateBuilder {
        builder
    }

    /// Extends the session that a local worker builds for each query. `ctx.headers` holds the
    /// config options that the coordinator sends with the query. By default, we extend the
    /// worker's session the same way as the coordinator's.
    fn configure_worker_session(&self, ctx: WorkerQueryContext) -> Result<SessionStateBuilder> {
        Ok(self.configure_session(ctx.builder))
    }
}

#[derive(Clone)]
pub struct ParquetBenchmarkBackend;

#[async_trait]
impl BenchmarkBackend for ParquetBenchmarkBackend {
    async fn register_tables(&mut self, ctx: &SessionContext, path: &Path) -> Result<()> {
        register_tables(ctx, path).await
    }
}
