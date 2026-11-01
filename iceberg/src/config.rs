use std::sync::Arc;

use datafusion::common::extensions_options;
use datafusion::config::{ConfigExtension, ConfigOptions};
use datafusion::execution::TaskContext;
use datafusion::prelude::SessionConfig;

extensions_options! {
    /// Configuration for Iceberg table reads.
    pub struct IcebergConfig {
        /// Maximum number of data files to read concurrently. Must be greater than zero.
        pub data_file_concurrency_limit: usize, default = 8
        /// Whether to prune Parquet row groups using their statistics.
        pub row_group_filtering_enabled: bool, default = true
        /// Whether to apply row-level selections while reading Parquet files.
        pub row_selection_enabled: bool, default = false
        /// Whether to include column statistics read during planning: table-level statistics
        /// reported to the planner. Unrelated to the per-file manifest column statistics that
        /// `dynamic_filter_pruning_enabled` requests to skip files by their column bounds.
        pub column_stats_enabled: bool, default = false
        /// Discover and retain file tasks during physical planning instead of execution.
        pub eager_file_planning: bool, default = false
        /// Whether Iceberg scans keep the filters parent operators push into them, including
        /// dynamic filters such as a hash join's build-side bounds or a TopK's threshold, and
        /// send them to workers with the scan. Every filter is still reported as not pushed
        /// down. Kept dynamic filters skip data files whose identity partition values or
        /// manifest column bounds (min/max, null and NaN counts) can't match them, checked with
        /// the filters' value when the reader takes each file and counted in the scan's
        /// `files_skipped` metric, with `files_skipped_by_bounds` counting those only the
        /// bounds ruled out. To that end each file task carries the manifest statistics of the
        /// scan's projected columns, requested when the scan is planned; this is separate from
        /// `column_stats_enabled`. When the reader opens a file, they
        /// also prune its row groups, and its pages when `row_selection_enabled` is set, with
        /// their value at that time. They are evaluated row by row inside the reader only when
        /// `datafusion.execution.parquet.pushdown_filters` is set, as DataFusion's Parquet scan
        /// does; otherwise rows in the surviving row groups and pages are returned unfiltered.
        /// Static filters stay enforced by the `FilterExec` above the scan; dynamic filters
        /// have no `FilterExec` and are enforced by the operator that created them (the hash
        /// join probes its hash table, TopK and aggregates re-check their threshold), so
        /// pruning only changes what is read, never the result.
        pub dynamic_filter_pruning_enabled: bool, default = false
    }
}

impl ConfigExtension for IcebergConfig {
    const PREFIX: &'static str = "iceberg";
}

impl IcebergConfig {
    /// Returns the registered Iceberg configuration, or its defaults when it
    /// has not been registered.
    ///
    /// Register [`IcebergConfig::default`] through
    /// [`SessionConfig::with_option_extension`] to configure these settings
    /// with `iceberg.*` DataFusion options.
    pub fn from_config_options(cfg: &ConfigOptions) -> Self {
        cfg.extensions.get::<Self>().cloned().unwrap_or_default()
    }

    /// Returns the registered Iceberg configuration, or its defaults.
    pub fn from_session_config(session_cfg: &SessionConfig) -> Self {
        Self::from_config_options(session_cfg.options())
    }

    /// Returns the registered Iceberg configuration, or its defaults.
    pub fn from_task_context(ctx: &Arc<TaskContext>) -> Self {
        Self::from_session_config(ctx.session_config())
    }
}

#[cfg(test)]
mod tests {
    use super::IcebergConfig;
    use datafusion::prelude::SessionConfig;

    #[test]
    fn config_options_can_be_set_through_session_config() {
        let config = SessionConfig::new()
            .with_option_extension(IcebergConfig::default())
            .set_usize("iceberg.data_file_concurrency_limit", 8)
            .set_bool("iceberg.row_group_filtering_enabled", false)
            .set_bool("iceberg.row_selection_enabled", true);

        let iceberg_config = IcebergConfig::from_session_config(&config);
        assert_eq!(iceberg_config.data_file_concurrency_limit, 8);
        assert!(!iceberg_config.row_group_filtering_enabled);
        assert!(iceberg_config.row_selection_enabled);
    }

    #[test]
    fn unregistered_config_uses_defaults() {
        assert_eq!(
            IcebergConfig::from_session_config(&SessionConfig::new()).data_file_concurrency_limit,
            IcebergConfig::default().data_file_concurrency_limit,
        );
    }
}
