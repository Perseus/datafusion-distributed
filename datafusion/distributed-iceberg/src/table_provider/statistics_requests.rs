use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, HashSet, Result};
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::statistics::StatisticsRequest;
use datafusion::logical_expr::{Expr, LogicalPlan, TableScan};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::{IcebergCatalogTableProvider, IcebergConfig, IcebergStaticTableProvider};

/// Asks eagerly planned Iceberg scans for the column stats that their dynamic filters can use.
///
/// Eager planning lists a scan's files inside `TableProvider::scan`, before DataFusion pushes
/// dynamic filters into the scan, so the scan has to learn the filter columns from the logical
/// plan. Lazy planning lists files when the query runs, and reads the columns from the dynamic
/// filters it was given, so it doesn't need this rule.
///
/// DataFusion pushes dynamic filters into a scan from three places: a hash join, a sort with a
/// limit, and an aggregate made only of min() and max() with no GROUP BY. We walk down from the
/// root, follow the columns those operators filter on to the scans they come from, and add Min,
/// Max and NullCount requests for them to those scans. The table provider then asks the
/// `iceberg` crate to keep those columns' stats on each file task.
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
        let iceberg = IcebergConfig::from_config_options(&config.options());
        if !iceberg.eager_file_planning || !iceberg.dynamic_filtering_enabled {
            return Ok(Transformed::no(plan));
        }
        request_statistics(plan, &HashSet::new())
    }
}

/// Adds statistics requests to the Iceberg scans in `plan`. `wanted` holds the columns of
/// `plan`'s output that a dynamic filter above it uses.
fn request_statistics(
    plan: LogicalPlan,
    wanted: &HashSet<Column>,
) -> Result<Transformed<LogicalPlan>> {
    if let LogicalPlan::TableScan(scan) = plan {
        return Ok(request_scan_statistics(scan, wanted));
    }
    let mut wanted_by_inputs = wanted_by_inputs(&plan, wanted).into_iter();
    plan.map_children(|input| {
        let wanted = wanted_by_inputs.next().unwrap_or_default();
        request_statistics(input, &wanted)
    })?
    // A subquery's dynamic filters stay inside it, so it starts with no wanted columns.
    .transform_data(|plan| {
        plan.map_subqueries(|subquery| request_statistics(subquery, &HashSet::new()))
    })
}

/// The columns of each of `plan`'s inputs that a dynamic filter uses: the `wanted` columns of
/// `plan`'s output, traced back to its inputs, and the columns of `plan`'s own dynamic filter.
fn wanted_by_inputs(plan: &LogicalPlan, wanted: &HashSet<Column>) -> Vec<HashSet<Column>> {
    match plan {
        // These nodes rename columns, so we map each output column to its input by position.
        LogicalPlan::Projection(projection) => {
            let inputs = wanted
                .iter()
                .filter_map(|column| projection.schema.maybe_index_of_column(column))
                .flat_map(|index| projection.expr[index].column_refs())
                .cloned()
                .collect();
            vec![inputs]
        }
        LogicalPlan::SubqueryAlias(alias) => {
            vec![input_columns(&alias.schema, alias.input.schema(), wanted)]
        }
        LogicalPlan::Union(union) => union
            .inputs
            .iter()
            .map(|input| input_columns(&union.schema, input.schema(), wanted))
            .collect(),
        // Other nodes keep their input columns' names, so a column belongs to the input whose
        // schema has it.
        _ => {
            let mut wanted = wanted.clone();
            wanted.extend(own_dynamic_filter_columns(plan));
            plan.inputs()
                .into_iter()
                .map(|input| {
                    wanted
                        .iter()
                        .filter(|column| input.schema().has_column(column))
                        .cloned()
                        .collect()
                })
                .collect()
        }
    }
}

/// The columns of `input` at the positions of the `wanted` columns in `output`.
fn input_columns(output: &DFSchema, input: &DFSchema, wanted: &HashSet<Column>) -> HashSet<Column> {
    wanted
        .iter()
        .filter_map(|column| output.maybe_index_of_column(column))
        .map(|index| {
            let (qualifier, field) = input.qualified_field(index);
            Column::new(qualifier.cloned(), field.name())
        })
        .collect()
}

/// The input columns that `plan`'s own dynamic filter uses, if `plan` builds one.
fn own_dynamic_filter_columns(plan: &LogicalPlan) -> Vec<Column> {
    match plan {
        LogicalPlan::Join(join) => join
            .on
            .iter()
            .flat_map(|(left, right)| left.column_refs().into_iter().chain(right.column_refs()))
            .cloned()
            .collect(),
        LogicalPlan::Sort(sort) if sort.fetch.is_some() => sort
            .expr
            .iter()
            .flat_map(|sort_expr| sort_expr.expr.column_refs())
            .cloned()
            .collect(),
        // DataFusion only builds this filter when every aggregate is min() or max() of one
        // column, and it checks the function by name too.
        LogicalPlan::Aggregate(aggregate) if aggregate.group_expr.is_empty() => aggregate
            .aggr_expr
            .iter()
            .map(min_max_column)
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// The column inside `min(column)` or `max(column)`.
fn min_max_column(expr: &Expr) -> Option<Column> {
    let Expr::AggregateFunction(AggregateFunction { func, params }) = unalias(expr) else {
        return None;
    };
    let name = func.name();
    if !name.eq_ignore_ascii_case("min") && !name.eq_ignore_ascii_case("max") {
        return None;
    }
    match params.args.as_slice() {
        [Expr::Column(column)] => Some(column.clone()),
        _ => None,
    }
}

fn unalias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => unalias(&alias.expr),
        expr => expr,
    }
}

/// Adds Min, Max and NullCount requests for each `wanted` column of an Iceberg `scan`.
fn request_scan_statistics(
    mut scan: TableScan,
    wanted: &HashSet<Column>,
) -> Transformed<LogicalPlan> {
    if wanted.is_empty() || !is_iceberg_scan(&scan) {
        return Transformed::no(LogicalPlan::TableScan(scan));
    }
    let mut changed = false;
    for (qualifier, field) in scan.projected_schema.iter() {
        let column = Column::new(qualifier.cloned(), field.name());
        if !wanted.contains(&column) && !wanted.contains(&Column::new_unqualified(field.name())) {
            continue;
        }
        let column = Arc::new(column);
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

fn is_iceberg_scan(scan: &TableScan) -> bool {
    source_as_provider(&scan.source).is_ok_and(|provider| {
        provider.is::<IcebergStaticTableProvider>() || provider.is::<IcebergCatalogTableProvider>()
    })
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
