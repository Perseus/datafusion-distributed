#![deny(clippy::all)]

mod codec;
mod common;
mod config_extension_ext;
mod coordinator;
mod distributed_ext;
mod distributed_planner;
mod dynamic_filtering;
mod execution_plans;
mod explain_analyze;
mod metrics;
mod passthrough_headers;
mod protocol;
mod stage;
mod work_unit_feed;
mod worker;
mod worker_resolver;

#[cfg(feature = "grpc")]
pub use arrow_ipc::CompressionType;
pub use coordinator::DistributedExec;
pub use distributed_ext::{DistributedExt, DistributedGetterExt};
pub use distributed_planner::{
    ColumnByteStatisticsProvider, DistributedConfig, NetworkBoundary, NetworkBoundaryExt,
    ProducerHead, SessionStateBuilderExt, create_distributed_plan,
};
pub use dynamic_filtering::rewrite_distributed_plan_with_dynamic_filters;
pub use events::{
    CoordinatorToWorkerDialer, DesiredTaskCountEvent, DesiredTaskCountEventResponse,
    DesiredTaskCountHandler, RouteTaskEvent, RouteTaskEventResponse, RouteTaskHandler,
    ScaleUpLeafNodeEvent, ScaleUpLeafNodeEventResponse, ScaleUpLeafNodeHandler,
    TaskCountAnnotation, TaskFinishedEvent, TaskFinishedEventResponse, TaskFinishedHandler,
    TaskOutcome, WorkerPlanRewriteEvent, WorkerPlanRewriteEventResponse, WorkerPlanRewriteHandler,
};
pub use execution_plans::{
    BroadcastExec, DistributedLeafExec, NetworkBroadcastExec, NetworkCoalesceExec,
    NetworkShuffleExec,
};
pub use metrics::{
    AvgLatencyMetric, BytesCounterMetric, BytesMetricExt, DISTRIBUTED_DATAFUSION_TASK_ID_LABEL,
    DistributedMetricsFormat, FirstLatencyMetric, GaugeMetricExt, LatencyMetricExt, MaxGaugeMetric,
    MaxLatencyMetric, MinLatencyMetric, P50LatencyMetric, P75LatencyMetric, P95LatencyMetric,
    P99LatencyMetric, rewrite_distributed_plan_with_metrics,
};
pub use protocol::LocalWorkerContext;

mod events;
#[cfg(any(feature = "integration", test))]
pub mod test_utils;

#[cfg(feature = "grpc")]
pub use protocol::grpc;

pub use codec::DistributedCodec;
pub use common::MaybeEncoded;
pub use worker_resolver::{WorkerResolver, get_distributed_worker_resolver};

pub use protocol::{
    ApplyDynamicFilter, ChannelResolver, CoordinatorToWorkerMsg, ExecuteTaskRequest,
    GetWorkerInfoRequest, GetWorkerInfoResponse, LoadInfo, ProducedDynamicFilter, SetPlanRequest,
    TaskCompletedDynamicFilters, TaskDynamicFilter, TaskKey, TaskMetrics, WorkUnitBatch,
    WorkUnitFeedDeclaration, WorkUnitMsg, WorkerChannel, WorkerToCoordinatorMsg,
    get_distributed_channel_resolver,
};
pub use stage::{
    DistributedTaskContext, Stage, display_plan_ascii, display_plan_graphviz, explain_analyze,
};
pub use work_unit_feed::{
    DistributedWorkUnitFeedContext, WorkUnit, WorkUnitFeed, WorkUnitFeedProto, WorkUnitFeedProvider,
};
pub use worker::{
    CoordinatorChannelResult, DefaultSessionBuilder, MappedWorkerSessionBuilder,
    MappedWorkerSessionBuilderExt, TaskData, Worker, WorkerQueryContext, WorkerSessionBuilder,
};

#[cfg(any(feature = "integration", test))]
pub use execution_plans::benchmarks::{
    LocalRepartitionBench, LocalRepartitionFixture, LocalRepartitionMode, ShuffleBench,
    ShuffleFixture, TransportBench, TransportBenchMode, TransportFixture,
};

/// Temporary, for measuring where query time goes: prints a timestamped line to stderr.
#[doc(hidden)]
pub fn timing(event: &str, details: std::fmt::Arguments<'_>) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let thread = std::thread::current();
    eprintln!(
        "DFD_TIMING t={}.{:09} ev={event} thread={}/{:?} {details}",
        now.as_secs(),
        now.subsec_nanos(),
        thread.name().unwrap_or("-"),
        thread.id(),
    );
}
