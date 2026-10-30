#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use datafusion::error::Result;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::{ExecutionPlan, displayable};
    use datafusion_distributed::DistributedExt;
    use datafusion_distributed_iceberg::{
        EagerIcebergCodec, IcebergExt, IcebergIntegrationOptions, LazyIcebergCodec,
    };
    use datafusion_distributed_iceberg_test_tools::{IcebergTestHarness, create_taxi_table};
    use datafusion_proto::physical_plan::{
        ComposedPhysicalExtensionCodec, PhysicalExtensionCodec, PhysicalProtoConverterExtension,
    };

    /// The worker-side lazy scan receives the hash join's dynamic filter, with the build
    /// side's bounds and values, instead of the initial empty placeholder.
    #[tokio::test]
    async fn lazy_worker_scan_sees_updated_dynamic_filter() -> Result<()> {
        let decoded = DecodedScans::default();
        let recorder = decoded.clone();
        let harness = IcebergTestHarness::builder()
            .with_session_extension(move |state| {
                state
                    .with_distributed_user_codec(recorder.codec())
                    .with_iceberg_integration(Default::default())
            })
            .with_table(create_taxi_table().await?)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1_000_000)
            })?
            .build()
            .await?;

        let (_, batches) = harness.query(JOIN_QUERY).await?;

        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        insta::assert_snapshot!(decoded.worker_probe_scans(&harness), @"
        DataSourceExec: format=iceberg, physical_predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        DataSourceExec: format=iceberg, physical_predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        ");
        Ok(())
    }

    /// The worker-side eager scan receives the hash join's dynamic filter, with the build
    /// side's bounds and values, instead of the initial empty placeholder.
    #[tokio::test]
    async fn eager_worker_scan_sees_updated_dynamic_filter() -> Result<()> {
        let decoded = DecodedScans::default();
        let recorder = decoded.clone();
        let harness = IcebergTestHarness::builder()
            .with_session_extension(move |state| {
                state
                    .with_distributed_user_codec(recorder.codec())
                    .with_iceberg_integration(Default::default())
            })
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .with_workers(2)
            .configure_session(|state| {
                state.with_distributed_file_scan_config_bytes_per_partition(1_000_000)
            })?
            .build()
            .await?;

        let (_, batches) = harness.query(JOIN_QUERY).await?;

        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        insta::assert_snapshot!(decoded.worker_probe_scans(&harness), @"
        DataSourceExec: format=iceberg, planned_files=3, predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        DataSourceExec: format=iceberg, planned_files=4, predicate=DynamicFilter [ pickup_date@0 >= 2024-01-10 AND pickup_date@0 <= 2024-01-10 AND pickup_date@0 IN (SET) ([2024-01-10]) ]
        ");
        Ok(())
    }

    const JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM (SELECT DISTINCT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10') days
        JOIN taxi ON taxi.pickup_date = days.pickup_date";

    /// Iceberg scans decoded from the wire, i.e. as workers hold them.
    #[derive(Clone, Default)]
    struct DecodedScans(Arc<Mutex<Vec<DecodedScan>>>);

    /// The id of the session that decoded a scan, and the scan.
    type DecodedScan = (String, Arc<dyn ExecutionPlan>);

    impl DecodedScans {
        /// A codec for Iceberg scans that records every scan it decodes. Registered before
        /// the Iceberg integration, it takes precedence over the integration's own codecs.
        fn codec(&self) -> RecordingCodec {
            let options = IcebergIntegrationOptions::default();
            RecordingCodec {
                inner: ComposedPhysicalExtensionCodec::new(vec![
                    Arc::new(LazyIcebergCodec::new(
                        Arc::clone(&options.storage_factory),
                        options.iceberg_runtime.clone(),
                    )),
                    Arc::new(EagerIcebergCodec::new(
                        options.storage_factory,
                        options.iceberg_runtime,
                    )),
                ]),
                decoded: self.clone(),
            }
        }

        /// Displays the scans workers decoded that hold a dynamic filter, one per line, with
        /// the filter's current value. Scans the coordinator decodes for itself are skipped.
        fn worker_probe_scans(&self, harness: &IcebergTestHarness) -> String {
            let coordinator = harness.state().session_id().to_string();
            let mut lines: Vec<String> = self
                .0
                .lock()
                .unwrap()
                .iter()
                .filter(|(session, _)| *session != coordinator)
                .map(|(_, plan)| displayable(plan.as_ref()).one_line().to_string())
                .filter(|line| line.contains("DynamicFilter"))
                .collect();
            lines.sort();
            lines.join("")
        }
    }

    #[derive(Debug)]
    struct RecordingCodec {
        inner: ComposedPhysicalExtensionCodec,
        decoded: DecodedScans,
    }

    impl std::fmt::Debug for DecodedScans {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("DecodedScans")
        }
    }

    impl PhysicalExtensionCodec for RecordingCodec {
        fn try_decode(
            &self,
            buf: &[u8],
            inputs: &[Arc<dyn ExecutionPlan>],
            ctx: &TaskContext,
            proto_converter: &dyn PhysicalProtoConverterExtension,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            let plan = self.inner.try_decode(buf, inputs, ctx, proto_converter)?;
            self.decoded
                .0
                .lock()
                .unwrap()
                .push((ctx.session_id(), Arc::clone(&plan)));
            Ok(plan)
        }

        fn try_encode(
            &self,
            node: Arc<dyn ExecutionPlan>,
            buf: &mut Vec<u8>,
            proto_converter: &dyn PhysicalProtoConverterExtension,
        ) -> Result<()> {
            self.inner.try_encode(node, buf, proto_converter)
        }
    }
}
