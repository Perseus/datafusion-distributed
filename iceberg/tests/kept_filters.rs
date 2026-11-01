mod tests {
    use datafusion::error::Result;
    use datafusion_distributed_iceberg::IcebergExt;
    use datafusion_distributed_iceberg_test_tools::IcebergTestHarness;
    use datafusion_distributed_iceberg_test_tools::create_taxi_table;

    /// The static filter is kept and still enforced by the `FilterExec` above the scan.
    #[tokio::test]
    async fn lazy_scan_keeps_static_filter() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query("SELECT COUNT(*) AS trips FROM taxi WHERE trip_distance >= 2.0")
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: trip_distance@0 >= 2, projection=[]
                  DataSourceExec: format=iceberg, table=default.taxi, projection=[trip_distance], predicate=trip_distance >= 2, physical_predicate=trip_distance@0 >= 2
        ");
        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 82899 |
        +-------+
        ");
        Ok(())
    }

    #[tokio::test]
    async fn eager_scan_keeps_static_filter() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .with_config_option("iceberg.eager_file_planning", true)
            .query("SELECT COUNT(*) AS trips FROM taxi WHERE trip_distance >= 2.0")
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                FilterExec: trip_distance@0 >= 2, projection=[]
                  DataSourceExec: format=iceberg, planned_files=7, predicate=trip_distance@0 >= 2
        ");
        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 82899 |
        +-------+
        ");
        Ok(())
    }

    /// The dynamic filter is kept on the probe scan with no `FilterExec` above it: the hash
    /// join enforces it.
    #[tokio::test]
    async fn lazy_scan_keeps_hash_join_dynamic_filter() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .query(JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                HashJoinExec: mode=CollectLeft, join_type=RightSemi, on=[(pickup_date@0, pickup_date@0)], projection=[]
                  CoalescePartitionsExec
                    AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                      RepartitionExec: partitioning=Hash([pickup_date@0], 4), input_partitions=4
                        AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                          FilterExec: pickup_date@0 = 2024-01-10
                            DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], predicate=pickup_date = 2024-01-10, physical_predicate=pickup_date@0 = 2024-01-10
                  DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], physical_predicate=DynamicFilter [ empty ]
        ");
        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    #[tokio::test]
    async fn eager_scan_keeps_hash_join_dynamic_filter() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.dynamic_filter_pruning_enabled", true)
            .with_config_option("iceberg.eager_file_planning", true)
            .query(JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                HashJoinExec: mode=CollectLeft, join_type=RightSemi, on=[(pickup_date@0, pickup_date@0)], projection=[]
                  CoalescePartitionsExec
                    AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                      RepartitionExec: partitioning=Hash([pickup_date@0], 4), input_partitions=4
                        AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                          FilterExec: pickup_date@0 = 2024-01-10
                            DataSourceExec: format=iceberg, planned_files=1, predicate=pickup_date@0 = 2024-01-10
                  DataSourceExec: format=iceberg, planned_files=7, predicate=DynamicFilter [ empty ]
        ");
        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// With `iceberg.dynamic_filter_pruning_enabled` off the scan keeps nothing, so the join
    /// creates no dynamic filter and the plan is the one this query had before filters were kept.
    #[tokio::test]
    async fn lazy_scan_keeps_no_filters_when_disabled() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .query(JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                HashJoinExec: mode=CollectLeft, join_type=RightSemi, on=[(pickup_date@0, pickup_date@0)], projection=[]
                  CoalescePartitionsExec
                    AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                      RepartitionExec: partitioning=Hash([pickup_date@0], 4), input_partitions=4
                        AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                          FilterExec: pickup_date@0 = 2024-01-10
                            DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date], predicate=pickup_date = 2024-01-10
                  DataSourceExec: format=iceberg, table=default.taxi, projection=[pickup_date]
        ");
        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// With `iceberg.dynamic_filter_pruning_enabled` off the scan keeps nothing, so the join
    /// creates no dynamic filter and the plan is the one this query had before filters were kept.
    #[tokio::test]
    async fn eager_scan_keeps_no_filters_when_disabled() -> Result<()> {
        let (plan, batches) = IcebergTestHarness::builder()
            .with_session_extension(|state| state.with_iceberg_integration(Default::default()))
            .with_table(create_taxi_table().await?)
            .with_config_option("iceberg.eager_file_planning", true)
            .query(JOIN_QUERY)
            .await?;

        insta::assert_snapshot!(plan, @"
        ProjectionExec: expr=[count(Int64(1))@0 as trips]
          AggregateExec: mode=Final, gby=[], aggr=[count(Int64(1))]
            CoalescePartitionsExec
              AggregateExec: mode=Partial, gby=[], aggr=[count(Int64(1))]
                HashJoinExec: mode=CollectLeft, join_type=RightSemi, on=[(pickup_date@0, pickup_date@0)], projection=[]
                  CoalescePartitionsExec
                    AggregateExec: mode=FinalPartitioned, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                      RepartitionExec: partitioning=Hash([pickup_date@0], 4), input_partitions=4
                        AggregateExec: mode=Partial, gby=[pickup_date@0 as pickup_date], aggr=[], ordering_mode=Sorted
                          FilterExec: pickup_date@0 = 2024-01-10
                            DataSourceExec: format=iceberg, planned_files=1
                  DataSourceExec: format=iceberg, planned_files=7
        ");
        insta::assert_snapshot!(batches, @"
        +-------+
        | trips |
        +-------+
        | 25000 |
        +-------+
        ");
        Ok(())
    }

    /// The build side is one pickup date, so the join pushes a dynamic filter on
    /// `pickup_date` into the probe-side scan of the whole table.
    const JOIN_QUERY: &str = r"SELECT COUNT(*) AS trips
        FROM (SELECT DISTINCT pickup_date FROM taxi WHERE pickup_date = DATE '2024-01-10') days
        JOIN taxi ON taxi.pickup_date = days.pickup_date";
}
