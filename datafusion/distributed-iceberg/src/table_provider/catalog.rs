use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{ScanArgs, ScanResult, Session};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::{Catalog, NamespaceIdent, TableIdent};

use super::requested_stats_columns;
use super::scan::{IcebergDataSourceOptions, plan_scan};
use crate::common::df_err;

/// Catalog-backed, read-only table provider with automatic metadata refresh.
///
/// The provider loads fresh table metadata from the catalog on every scan. For
/// a fixed snapshot, use IcebergStaticTableProvider instead.
#[derive(Debug, Clone)]
pub struct IcebergCatalogTableProvider {
    catalog: Arc<dyn Catalog>,
    table_ident: TableIdent,
    schema: SchemaRef,
    iceberg_runtime: iceberg::Runtime,
}

impl IcebergCatalogTableProvider {
    /// Creates a read-only catalog-backed provider.
    pub async fn try_new(
        catalog: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
        name: impl Into<String>,
        iceberg_runtime: iceberg::Runtime,
    ) -> Result<Self> {
        let table_ident = TableIdent::new(namespace, name.into());
        let table = catalog.load_table(&table_ident).await.map_err(df_err)?;
        let schema = schema_to_arrow_schema(table.metadata().current_schema()).map_err(df_err)?;

        Ok(Self {
            catalog,
            table_ident,
            schema: Arc::new(schema),
            iceberg_runtime,
        })
    }
}

#[async_trait]
impl TableProvider for IcebergCatalogTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let args = ScanArgs::default()
            .with_projection(projection)
            .with_filters(Some(filters))
            .with_limit(limit);
        Ok(self.scan_with_args(state, args).await?.into_inner())
    }

    async fn scan_with_args<'a>(
        &self,
        state: &dyn Session,
        args: ScanArgs<'a>,
    ) -> Result<ScanResult> {
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(df_err)?;
        let plan = plan_scan(
            table.clone(),
            self.schema.clone(),
            IcebergDataSourceOptions {
                snapshot_id: None,
                projection: args.projection(),
                filters: args.filters().unwrap_or_default(),
                fetch: args.limit(),
                stats_columns: requested_stats_columns(args.statistics_requests()),
                iceberg_runtime: Some(self.iceberg_runtime.clone()),
            },
            state.task_ctx(),
        )
        .await?;
        Ok(plan.into())
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}
