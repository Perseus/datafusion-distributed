use crate::IcebergConfig;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_expr::{PhysicalExpr, conjunction};
use std::fmt::Formatter;
use std::sync::Arc;

/// Visits the filters kept by an Iceberg data source, stopping when `f` asks to.
pub(crate) fn apply_filters(
    filters: &[Arc<dyn PhysicalExpr>],
    f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    for filter in filters {
        if f(filter)? == TreeNodeRecursion::Stop {
            return Ok(TreeNodeRecursion::Stop);
        }
    }
    Ok(TreeNodeRecursion::Continue)
}

/// Writes `, <key>=<filters ANDed together>`, or nothing when there are no filters.
pub(crate) fn fmt_filters(
    filters: &[Arc<dyn PhysicalExpr>],
    key: &str,
    f: &mut Formatter,
) -> std::fmt::Result {
    if filters.is_empty() {
        return Ok(());
    }
    write!(f, ", {key}={}", conjunction(filters.iter().cloned()))
}

/// Whether Iceberg sources keep the filters pushed to them, read from the planning-time
/// options passed to `try_pushdown_filters`.
pub(crate) fn keeps_filters(config: &ConfigOptions) -> bool {
    IcebergConfig::from_config_options(config).dynamic_filter_pruning_enabled
}
