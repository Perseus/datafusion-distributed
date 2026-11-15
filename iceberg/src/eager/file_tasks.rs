//! Compact wire form for the files assigned to an Iceberg scan.
//!
//! A [`FileScanTask`] carries values that are the same for every file of a table, like the table
//! schema and its column name mapping. Its serde form writes them again for every file. For wide
//! tables they make up most of the encoded plan, and decoding them again for every file is slow.
//!
//! [`FileTaskEncoder`] writes each of these values once per scan, and each file refers to them by
//! position. [`FileTaskDecoder`] reads them once and gives every file the same shared value back.

use std::sync::Arc;

use datafusion::common::{HashMap, Result, exec_datafusion_err};
use iceberg::expr::BoundPredicate;
use iceberg::scan::{FileScanTask, FileScanTaskColumnStats, FileScanTaskDeleteFile};
use iceberg::spec::{
    DataFileFormat, Literal, NameMapping, PartitionSpec, Schema, SortOrder, StructType, Type,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Collects the shared values of a scan while its files are encoded.
#[derive(Default)]
pub(crate) struct FileTaskEncoder<'a> {
    schemas: Pool<'a, Schema>,
    partition_specs: Pool<'a, PartitionSpec>,
    name_mappings: Pool<'a, NameMapping>,
    partition_types: Pool<'a, StructType>,
    sort_orders: Pool<'a, SortOrder>,
    predicates: Vec<&'a BoundPredicate>,
    file_partition_types: HashMap<(usize, Option<usize>), Type>,
}

impl<'a> FileTaskEncoder<'a> {
    /// Encodes one file. Its shared values are kept until [`Self::finish`].
    pub(crate) fn encode(&mut self, file: &'a FileScanTask) -> Result<Vec<u8>> {
        let schema = self.schemas.position(file.schema());
        let partition_spec = file
            .partition_spec()
            .map(|spec| self.partition_specs.position(&**spec));
        let partition = file
            .partition()
            .map(|partition| {
                let partition_type = cached_partition_type(
                    &mut self.file_partition_types,
                    (schema, partition_spec),
                    file.schema(),
                    file.partition_spec().map(|spec| &**spec),
                )?;
                Literal::Struct(partition.clone())
                    .try_into_json(partition_type)
                    .map_err(|error| {
                        exec_datafusion_err!("Cannot encode Iceberg partition value: {error}")
                    })
            })
            .transpose()?;
        let wire = FileRef {
            file_size_in_bytes: file.file_size_in_bytes(),
            start: file.start(),
            length: file.length(),
            record_count: file.record_count(),
            first_row_id: file.first_row_id(),
            data_sequence_number: file.data_sequence_number(),
            data_file_path: file.data_file_path(),
            data_file_format: file.data_file_format(),
            schema,
            project_field_ids: file.project_field_ids(),
            predicate: file.predicate().map(|predicate| self.predicate(predicate)),
            deletes: file.deletes(),
            partition,
            partition_spec,
            name_mapping: file
                .name_mapping()
                .map(|mapping| self.name_mappings.position(&**mapping)),
            unified_partition_type: file
                .unified_partition_type()
                .map(|partition_type| self.partition_types.position(&**partition_type)),
            sort_order_id: file.sort_order_id(),
            sort_order: file
                .sort_order()
                .map(|order| self.sort_orders.position(&**order)),
            column_stats: file.column_stats(),
            case_sensitive: file.case_sensitive(),
            key_metadata: file.key_metadata(),
        };
        rmp_serde::to_vec_named(&wire)
            .map_err(|error| exec_datafusion_err!("Cannot encode assigned Iceberg file: {error}"))
    }

    /// Encodes the shared values of every file passed to [`Self::encode`].
    pub(crate) fn finish(self) -> Result<Vec<u8>> {
        let shared = SharedRef {
            schemas: self.schemas.values,
            partition_specs: self.partition_specs.values,
            name_mappings: self.name_mappings.values,
            partition_types: self.partition_types.values,
            sort_orders: self.sort_orders.values,
            predicates: self.predicates,
        };
        rmp_serde::to_vec_named(&shared).map_err(|error| {
            exec_datafusion_err!("Cannot encode shared Iceberg file values: {error}")
        })
    }

    /// Files of one scan usually share one predicate, so a file reuses the last one if equal.
    fn predicate(&mut self, predicate: &'a BoundPredicate) -> usize {
        if self.predicates.last() != Some(&predicate) {
            self.predicates.push(predicate);
        }
        self.predicates.len() - 1
    }
}

/// Rebuilds the files of a scan from the shared values written by [`FileTaskEncoder::finish`].
pub(crate) struct FileTaskDecoder {
    schemas: Vec<Arc<Schema>>,
    partition_specs: Vec<Arc<PartitionSpec>>,
    name_mappings: Vec<Arc<NameMapping>>,
    partition_types: Vec<Arc<StructType>>,
    sort_orders: Vec<Arc<SortOrder>>,
    predicates: Vec<BoundPredicate>,
    file_partition_types: HashMap<(usize, Option<usize>), Type>,
}

impl FileTaskDecoder {
    pub(crate) fn new(shared: &[u8]) -> Result<Self> {
        let shared: Shared = rmp_serde::from_slice(shared)
            .map_err(|error| exec_datafusion_err!("Invalid shared Iceberg file values: {error}"))?;
        Ok(Self {
            schemas: arcs(shared.schemas),
            partition_specs: arcs(shared.partition_specs),
            name_mappings: arcs(shared.name_mappings),
            partition_types: arcs(shared.partition_types),
            sort_orders: arcs(shared.sort_orders),
            predicates: shared.predicates,
            file_partition_types: HashMap::default(),
        })
    }

    pub(crate) fn decode(&mut self, bytes: &[u8]) -> Result<FileScanTask> {
        let wire: FileOwned = rmp_serde::from_slice(bytes)
            .map_err(|error| exec_datafusion_err!("Invalid assigned Iceberg file: {error}"))?;
        let schema = Arc::clone(get(&self.schemas, wire.schema, "schema")?);
        let partition_spec = wire
            .partition_spec
            .map(|i| get(&self.partition_specs, i, "partition spec").cloned())
            .transpose()?;
        let partition = wire
            .partition
            .map(|json| {
                let partition_type = cached_partition_type(
                    &mut self.file_partition_types,
                    (wire.schema, wire.partition_spec),
                    &schema,
                    partition_spec.as_deref(),
                )?;
                match Literal::try_from_json(json, partition_type) {
                    Ok(Some(Literal::Struct(partition))) => Ok(partition),
                    Ok(_) => Err(exec_datafusion_err!(
                        "Iceberg partition value must be a struct"
                    )),
                    Err(error) => Err(exec_datafusion_err!(
                        "Invalid Iceberg partition value: {error}"
                    )),
                }
            })
            .transpose()?;
        FileScanTask::builder()
            .with_file_size_in_bytes(wire.file_size_in_bytes)
            .with_start(wire.start)
            .with_length(wire.length)
            .with_record_count(wire.record_count)
            .with_first_row_id(wire.first_row_id)
            .with_data_sequence_number(wire.data_sequence_number)
            .with_data_file_path(wire.data_file_path)
            .with_data_file_format(wire.data_file_format)
            .with_schema(schema)
            .with_project_field_ids(wire.project_field_ids)
            .with_predicate(
                wire.predicate
                    .map(|i| get(&self.predicates, i, "predicate").cloned())
                    .transpose()?,
            )
            .with_deletes(wire.deletes)
            .with_partition(partition)
            .with_partition_spec(partition_spec)
            .with_name_mapping(shared(
                &self.name_mappings,
                wire.name_mapping,
                "name mapping",
            )?)
            .with_unified_partition_type(shared(
                &self.partition_types,
                wire.unified_partition_type,
                "partition type",
            )?)
            .with_sort_order_id(wire.sort_order_id)
            .with_sort_order(shared(&self.sort_orders, wire.sort_order, "sort order")?)
            .with_column_stats(wire.column_stats)
            .with_case_sensitive(wire.case_sensitive)
            .with_key_metadata(wire.key_metadata.map(Vec::into_boxed_slice))
            .build()
            .map_err(|error| exec_datafusion_err!("Invalid assigned Iceberg file: {error}"))
    }
}

/// The type of a file's partition value, computed as Iceberg's own serde form does, once for
/// each pair of schema and partition spec.
fn cached_partition_type<'c>(
    cache: &'c mut HashMap<(usize, Option<usize>), Type>,
    key: (usize, Option<usize>),
    schema: &Schema,
    partition_spec: Option<&PartitionSpec>,
) -> Result<&'c Type> {
    if !cache.contains_key(&key) {
        let partition_type = match partition_spec {
            Some(spec) => spec.partition_type(schema),
            None => PartitionSpec::unpartition_spec().partition_type(schema),
        };
        let partition_type = partition_type
            .map_err(|error| exec_datafusion_err!("Invalid Iceberg partition spec: {error}"))?;
        cache.insert(key, Type::Struct(partition_type));
    }
    Ok(&cache[&key])
}

fn arcs<T>(values: Vec<T>) -> Vec<Arc<T>> {
    values.into_iter().map(Arc::new).collect()
}

fn shared<T>(values: &[Arc<T>], i: Option<usize>, what: &str) -> Result<Option<Arc<T>>> {
    i.map(|i| get(values, i, what).cloned()).transpose()
}

fn get<'v, T>(values: &'v [T], i: usize, what: &str) -> Result<&'v T> {
    values
        .get(i)
        .ok_or_else(|| exec_datafusion_err!("Iceberg file refers to a missing shared {what}"))
}

/// Gives each distinct shared value a position. Values are the same when they are the same
/// allocation, which is how the files planned from one table share them.
struct Pool<'a, T> {
    positions: HashMap<*const T, usize>,
    values: Vec<&'a T>,
}

impl<T> Default for Pool<'_, T> {
    fn default() -> Self {
        Self {
            positions: HashMap::default(),
            values: Vec::new(),
        }
    }
}

impl<'a, T> Pool<'a, T> {
    fn position(&mut self, value: &'a T) -> usize {
        let Self { positions, values } = self;
        *positions
            .entry(std::ptr::from_ref(value))
            .or_insert_with(|| {
                values.push(value);
                values.len() - 1
            })
    }
}

#[derive(Serialize)]
struct SharedRef<'a> {
    schemas: Vec<&'a Schema>,
    partition_specs: Vec<&'a PartitionSpec>,
    name_mappings: Vec<&'a NameMapping>,
    partition_types: Vec<&'a StructType>,
    sort_orders: Vec<&'a SortOrder>,
    predicates: Vec<&'a BoundPredicate>,
}

#[derive(Deserialize)]
struct Shared {
    schemas: Vec<Schema>,
    partition_specs: Vec<PartitionSpec>,
    name_mappings: Vec<NameMapping>,
    partition_types: Vec<StructType>,
    sort_orders: Vec<SortOrder>,
    predicates: Vec<BoundPredicate>,
}

/// A file with its shared values replaced by their positions in [`Shared`].
#[derive(Serialize)]
struct FileRef<'a> {
    file_size_in_bytes: u64,
    start: u64,
    length: u64,
    record_count: Option<u64>,
    first_row_id: Option<i64>,
    data_sequence_number: Option<i64>,
    data_file_path: &'a str,
    data_file_format: DataFileFormat,
    schema: usize,
    project_field_ids: &'a [i32],
    predicate: Option<usize>,
    deletes: &'a [FileScanTaskDeleteFile],
    partition: Option<JsonValue>,
    partition_spec: Option<usize>,
    name_mapping: Option<usize>,
    unified_partition_type: Option<usize>,
    sort_order_id: Option<i32>,
    sort_order: Option<usize>,
    column_stats: Option<&'a FileScanTaskColumnStats>,
    case_sensitive: bool,
    key_metadata: Option<&'a [u8]>,
}

#[derive(Deserialize)]
struct FileOwned {
    file_size_in_bytes: u64,
    start: u64,
    length: u64,
    record_count: Option<u64>,
    first_row_id: Option<i64>,
    data_sequence_number: Option<i64>,
    data_file_path: String,
    data_file_format: DataFileFormat,
    schema: usize,
    project_field_ids: Vec<i32>,
    predicate: Option<usize>,
    deletes: Vec<FileScanTaskDeleteFile>,
    partition: Option<JsonValue>,
    partition_spec: Option<usize>,
    name_mapping: Option<usize>,
    unified_partition_type: Option<usize>,
    sort_order_id: Option<i32>,
    sort_order: Option<usize>,
    column_stats: Option<FileScanTaskColumnStats>,
    case_sensitive: bool,
    key_metadata: Option<Vec<u8>>,
}

#[cfg(test)]
mod tests {
    use iceberg::expr::{Bind, Reference};
    use iceberg::spec::{
        DataContentType, Datum, MappedField, NestedField, NullOrder, PrimitiveType, SortDirection,
        SortField, Struct, Transform,
    };

    use super::*;

    fn roundtrip(files: &[FileScanTask]) -> Result<Vec<FileScanTask>> {
        let mut encoder = FileTaskEncoder::default();
        let encoded = files
            .iter()
            .map(|file| encoder.encode(file))
            .collect::<Result<Vec<_>>>()?;
        let mut decoder = FileTaskDecoder::new(&encoder.finish()?)?;
        encoded.iter().map(|bytes| decoder.decode(bytes)).collect()
    }

    #[test]
    fn rejects_missing_shared_values() -> Result<()> {
        let empty = FileTaskEncoder::default().finish()?;
        let wire = FileRef {
            file_size_in_bytes: 1,
            start: 0,
            length: 1,
            record_count: None,
            first_row_id: None,
            data_sequence_number: None,
            data_file_path: "file.parquet",
            data_file_format: DataFileFormat::Parquet,
            schema: 0,
            project_field_ids: &[],
            predicate: None,
            deletes: &[],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            unified_partition_type: None,
            sort_order_id: None,
            sort_order: None,
            column_stats: None,
            case_sensitive: true,
            key_metadata: None,
        };
        let bytes = rmp_serde::to_vec_named(&wire).unwrap();
        let error = FileTaskDecoder::new(&empty)?.decode(&bytes).unwrap_err();
        assert!(
            error.to_string().contains("missing shared schema"),
            "{error}"
        );
        Ok(())
    }

    /// Two files of one table with every field set, sharing the table's values.
    fn files_with_every_field() -> Vec<FileScanTask> {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Long),
                    )),
                    Arc::new(NestedField::optional(
                        2,
                        "day",
                        Type::Primitive(PrimitiveType::Date),
                    )),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .add_partition_field("day", "day", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );
        let partition_type = Arc::new(partition_spec.partition_type(&schema).unwrap());
        let name_mapping = Arc::new(NameMapping::new(vec![
            MappedField::new(Some(1), vec!["id".to_string()], vec![]),
            MappedField::new(Some(2), vec!["day".to_string()], vec![]),
        ]));
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_order_id(1)
                .with_sort_field(
                    SortField::builder()
                        .source_id(1)
                        .transform(Transform::Identity)
                        .direction(SortDirection::Ascending)
                        .null_order(NullOrder::First)
                        .build(),
                )
                .build(&schema)
                .unwrap(),
        );
        let predicate = Reference::new("id")
            .greater_than(Datum::long(5))
            .bind(schema.clone(), true)
            .unwrap();
        let deletes = vec![
            FileScanTaskDeleteFile::builder()
                .with_file_path("deletes.parquet".to_string())
                .with_file_size_in_bytes(10)
                .with_file_type(DataContentType::EqualityDeletes)
                .with_file_format(DataFileFormat::Parquet)
                .with_partition_spec_id(0)
                .with_equality_ids(Some(vec![1]))
                .with_record_count(Some(2))
                .build()
                .unwrap(),
        ];
        let column_stats = FileScanTaskColumnStats::builder()
            .with_column_sizes([(1, 64)].into_iter().collect())
            .with_value_counts([(1, 8)].into_iter().collect())
            .with_null_value_counts([(2, 1)].into_iter().collect())
            .with_lower_bounds([(1, Datum::long(1))].into_iter().collect())
            .with_upper_bounds([(1, Datum::long(9))].into_iter().collect())
            .build();
        (0..2)
            .map(|i| {
                FileScanTask::builder()
                    .with_file_size_in_bytes(1000 + i)
                    .with_start(i)
                    .with_length(900)
                    .with_record_count(Some(8))
                    .with_first_row_id(Some(100))
                    .with_data_sequence_number(Some(3))
                    .with_data_file_path(format!("data-{i}.parquet"))
                    .with_data_file_format(DataFileFormat::Parquet)
                    .with_schema(schema.clone())
                    .with_project_field_ids(vec![1, 2])
                    .with_predicate(Some(predicate.clone()))
                    .with_deletes(deletes.clone())
                    .with_partition(Some(Struct::from_iter([Some(Literal::date(
                        19_000 + i as i32,
                    ))])))
                    .with_partition_spec(Some(partition_spec.clone()))
                    .with_name_mapping(Some(name_mapping.clone()))
                    .with_unified_partition_type(Some(partition_type.clone()))
                    .with_sort_order_id(Some(1))
                    .with_sort_order(Some(sort_order.clone()))
                    .with_column_stats(Some(column_stats.clone()))
                    .with_case_sensitive(true)
                    .with_key_metadata(Some(vec![7, 8].into_boxed_slice()))
                    .build()
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn roundtrips_every_field() -> Result<()> {
        let files = files_with_every_field();
        let decoded = roundtrip(&files)?;
        assert_eq!(decoded, files);
        // The decoded files share the table's values again.
        assert!(Arc::ptr_eq(
            &decoded[0].schema_ref(),
            &decoded[1].schema_ref()
        ));
        assert!(Arc::ptr_eq(
            decoded[0].name_mapping().unwrap(),
            decoded[1].name_mapping().unwrap()
        ));
        Ok(())
    }

    #[test]
    fn writes_shared_values_once() -> Result<()> {
        let files = files_with_every_field();
        let mut encoder = FileTaskEncoder::default();
        for file in &files {
            encoder.encode(file)?;
        }
        let shared: Shared = rmp_serde::from_slice(&encoder.finish()?).unwrap();
        assert_eq!(shared.schemas.len(), 1);
        assert_eq!(shared.partition_specs.len(), 1);
        assert_eq!(shared.name_mappings.len(), 1);
        assert_eq!(shared.partition_types.len(), 1);
        assert_eq!(shared.sort_orders.len(), 1);
        assert_eq!(shared.predicates.len(), 1);
        Ok(())
    }
}
