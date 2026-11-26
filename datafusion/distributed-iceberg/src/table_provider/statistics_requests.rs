use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::common::{Column, HashSet, Result};
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::statistics::StatisticsRequest;
use datafusion::logical_expr::{Expr, LogicalPlan, TableScan};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::{IcebergCatalogTableProvider, IcebergConfig, IcebergStaticTableProvider};

/// Asks Iceberg scans for the column stats that their dynamic filters can use.
///
/// DataFusion pushes dynamic filters into a scan from three places: a hash join, a sort with a
/// limit, and an aggregate made only of min() and max() with no GROUP BY. We collect the columns
/// those operators filter on, and add Min, Max and NullCount requests for them to each Iceberg
/// `TableScan`. The table provider then asks iceberg-rust to keep those columns' stats on each
/// file task, so workers can skip files whose stats rule out the filter.
///
/// We match columns by name, since a logical plan doesn't say which scan a join key comes from.
/// A name that is also used by an unrelated column only makes us keep a few extra stats.
#[derive(Debug, Default)]
pub(crate) struct RequestDynamicFilterStatistics;

impl RequestDynamicFilterStatistics {
    pub(crate) const NAME: &str = "request_iceberg_dynamic_filter_statistics";
}

impl OptimizerRule for RequestDynamicFilterStatistics {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !IcebergConfig::from_config_options(&config.options()).dynamic_filtering_enabled {
            return Ok(Transformed::no(plan));
        }
        let columns = dynamic_filter_columns(&plan)?;
        if columns.is_empty() {
            return Ok(Transformed::no(plan));
        }
        plan.transform_up_with_subqueries(|node| match node {
            LogicalPlan::TableScan(scan) if is_iceberg_scan(&scan) => {
                Ok(request_statistics(scan, &columns))
            }
            node => Ok(Transformed::no(node)),
        })
    }
}

/// The column names that a dynamic filter in `plan` can filter on.
fn dynamic_filter_columns(plan: &LogicalPlan) -> Result<HashSet<String>> {
    let mut columns = HashSet::new();
    plan.apply_with_subqueries(|node| {
        match node {
            LogicalPlan::Join(join) => {
                for (left, right) in &join.on {
                    add_columns(left, &mut columns);
                    add_columns(right, &mut columns);
                }
            }
            LogicalPlan::Sort(sort) if sort.fetch.is_some() => {
                for sort_expr in &sort.expr {
                    add_columns(&sort_expr.expr, &mut columns);
                }
            }
            LogicalPlan::Aggregate(aggregate) if aggregate.group_expr.is_empty() => {
                // DataFusion only builds this filter when every aggregate is min() or max() of
                // one column, and it checks the function by name too.
                let arguments: Option<Vec<&Column>> =
                    aggregate.aggr_expr.iter().map(min_max_column).collect();
                for column in arguments.into_iter().flatten() {
                    columns.insert(column.name.clone());
                }
            }
            _ => {}
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(columns)
}

fn add_columns(expr: &Expr, columns: &mut HashSet<String>) {
    columns.extend(expr.column_refs().into_iter().map(|c| c.name.clone()));
}

/// The column inside `min(column)` or `max(column)`.
fn min_max_column(expr: &Expr) -> Option<&Column> {
    let Expr::AggregateFunction(AggregateFunction { func, params }) = unalias(expr) else {
        return None;
    };
    let name = func.name();
    if !name.eq_ignore_ascii_case("min") && !name.eq_ignore_ascii_case("max") {
        return None;
    }
    match params.args.as_slice() {
        [Expr::Column(column)] => Some(column),
        _ => None,
    }
}

fn unalias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => unalias(&alias.expr),
        expr => expr,
    }
}

fn is_iceberg_scan(scan: &TableScan) -> bool {
    source_as_provider(&scan.source).is_ok_and(|provider| {
        provider.is::<IcebergStaticTableProvider>() || provider.is::<IcebergCatalogTableProvider>()
    })
}

/// Adds Min, Max and NullCount requests for each column of `scan` in `columns`.
fn request_statistics(mut scan: TableScan, columns: &HashSet<String>) -> Transformed<LogicalPlan> {
    let mut changed = false;
    for field in scan.projected_schema.fields() {
        if !columns.contains(field.name()) {
            continue;
        }
        let column = Arc::new(Column::from_name(field.name()));
        for request in [
            StatisticsRequest::Min(Arc::clone(&column)),
            StatisticsRequest::Max(Arc::clone(&column)),
            StatisticsRequest::NullCount(Arc::clone(&column)),
        ] {
            changed |= scan.statistics_requests.insert(request);
        }
    }
    Transformed::new_transformed(LogicalPlan::TableScan(scan), changed)
}

/// The columns with a Min or Max request, in the order of `requests`.
pub(crate) fn requested_stats_columns(requests: &[StatisticsRequest]) -> Vec<String> {
    let mut columns: Vec<String> = Vec::new();
    for request in requests {
        let (StatisticsRequest::Min(column) | StatisticsRequest::Max(column)) = request else {
            continue;
        };
        if !columns.contains(&column.name) {
            columns.push(column.name.clone());
        }
    }
    columns
}
