use crate::common::df_err;
use datafusion::common::runtime::JoinSet;
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, HashMap, HashSet, Result};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, Operator};
use datafusion::scalar::ScalarValue;
use iceberg::puffin::APACHE_DATASKETCHES_THETA_V1;
use iceberg::scan::FileScanTask;
use iceberg::spec::{
    DataFile, Datum, ManifestContentType, ManifestList, NestedFieldRef, PrimitiveLiteral,
    PrimitiveType, SnapshotRef, Type,
};
use iceberg::table::Table;
use std::sync::Arc;

/// Inputs for computing Iceberg column statistics for a planned scan.
pub(crate) struct ColumnStatsRequest<'a> {
    pub(crate) fields: Vec<NestedFieldRef>,
    pub(crate) snapshot: Option<SnapshotRef>,
    /// Data files selected by eager file planning, if eager planning ran.
    pub(crate) selected_files: Option<&'a [FileScanTask]>,
    /// Filter expressions pushed to the scan.
    pub(crate) filters: &'a [Expr],
    pub(crate) may_filter_rows: bool,
}

/// Getting number of distinct values for columns in a table
///
/// Note: https://iceberg.apache.org/puffin-spec/#apache-datasketches-theta-v1-blob-type
fn ndv_from_metadata(table: &Table, snapshot_id: i64) -> HashMap<i32, usize> {
    let mut ndvs = HashMap::new();
    let Some(stats) = table.metadata().statistics_for_snapshot(snapshot_id) else {
        return ndvs;
    };
    for blob in &stats.blob_metadata {
        // Getting sketch estimates of ndv in every column
        if blob.r#type != APACHE_DATASKETCHES_THETA_V1 {
            continue;
        }

        let &[field_id] = &blob.fields[..] else {
            continue;
        };
        if let Some(ndv) = blob.properties.get("ndv").and_then(|v| v.parse().ok()) {
            ndvs.insert(field_id, ndv);
        }
    }
    ndvs
}

/// Reading table column statistics from data-files concurrently, optionally scoped to the
/// files selected by eager file planning.
///
/// Table-wide Puffin NDV sketches are omitted whenever file pruning dropped files, because
/// applying a snapshot-wide `1 / NDV` selectivity to an already-pruned file subset
/// double-counts pruning and severely underestimates skewed values. Similarly, non-singleton
/// `min_value`/`max_value` bounds are cleared on equality-filtered numeric/temporal columns
/// with unknown NDV so DataFusion's `FilterExec` does not fall back to `1 / (max - min + 1)`
/// over sparse ID ranges.
pub(crate) async fn compute_column_stats(
    table: Table,
    request: ColumnStatsRequest<'_>,
) -> Result<Vec<ColumnStatistics>> {
    let ColumnStatsRequest {
        fields,
        snapshot,
        selected_files,
        filters,
        may_filter_rows,
    } = request;

    if fields.is_empty() {
        return Ok(Vec::new());
    }

    if let Some(tasks) = selected_files {
        if tasks.is_empty() {
            return Ok(vec![
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    distinct_count: Precision::Exact(0),
                    byte_size: Precision::Exact(0),
                    ..ColumnStatistics::new_unknown()
                };
                fields.len()
            ]);
        }
        if tasks
            .iter()
            .any(|task| task.start() != 0 || task.length() != task.file_size_in_bytes())
        {
            // Full-file manifest metrics cannot describe a byte-range split task.
            return Ok(vec![ColumnStatistics::new_unknown(); fields.len()]);
        }
    }

    let Some(actual_snapshot) = snapshot else {
        return Ok(vec![ColumnStatistics::new_unknown(); fields.len()]);
    };

    let metadata = table.metadata();
    let ml_bytes = table
        .file_io()
        .new_input(actual_snapshot.manifest_list())
        .map_err(df_err)?
        .read()
        .await
        .map_err(df_err)?;
    let manifest_list = Arc::new(
        ManifestList::parse_with_version(&ml_bytes, metadata.format_version()).map_err(df_err)?,
    );

    // Account for merge-on-read delete files later by marking the counts as inexact.
    let has_deletes = manifest_list
        .entries()
        .iter()
        .any(|f| f.content == ManifestContentType::Deletes);
    // Collect all needed manifest entries before spawning.
    let manifests: Vec<_> = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Data)
        .cloned()
        .collect();

    let field_specs: Arc<Vec<(i32, bool, Vec<i32>)>> = Arc::new(
        fields
            .iter()
            .map(|field| {
                let is_primitive = matches!(field.field_type.as_ref(), Type::Primitive(_));
                let mut ids = Vec::new();
                leaf_ids(field, &mut ids);
                (field.id, is_primitive, ids)
            })
            .collect(),
    );
    let selected_paths: Option<Arc<HashSet<String>>> = selected_files.map(|tasks| {
        Arc::new(
            tasks
                .iter()
                .map(|task| task.data_file_path().to_owned())
                .collect(),
        )
    });

    let mut join_set = JoinSet::new();

    for manifest_file in manifests {
        let table = table.clone();
        let field_specs = Arc::clone(&field_specs);
        let selected_paths = selected_paths.clone();

        join_set.spawn(async move {
            let manifest = table.manifest_reader().read(&manifest_file).await?;

            let mut col_stats: Vec<Option<ColumnStatistics>> = vec![None; field_specs.len()];
            let mut total_alive_files = 0usize;
            let mut matched_files = 0usize;

            for entry in manifest.entries().iter().filter(|e| e.is_alive()) {
                total_alive_files += 1;
                let df = entry.data_file();
                if let Some(paths) = &selected_paths
                    && !paths.contains(df.file_path())
                {
                    continue;
                }
                matched_files += 1;

                for (i, (id, is_primitive, leaf_ids)) in field_specs.iter().enumerate() {
                    let next = data_file_col_stats(df, *id, *is_primitive, leaf_ids);
                    col_stats[i] = Some(merge_col_stats(col_stats[i].take(), next));
                }
            }

            Ok::<_, iceberg::Error>((col_stats, total_alive_files, matched_files))
        });
    }

    let mut merged: Vec<Option<ColumnStatistics>> = vec![None; fields.len()];
    let mut total_alive_files = 0usize;
    let mut matched_files = 0usize;
    while let Some(result) = join_set.join_next().await {
        let (manifest_stats, manifest_alive, manifest_matched) = result
            .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))?
            .map_err(df_err)?;
        total_alive_files += manifest_alive;
        matched_files += manifest_matched;
        for (acc, next) in merged.iter_mut().zip(manifest_stats) {
            if let Some(next) = next {
                *acc = Some(merge_col_stats(acc.take(), next));
            }
        }
    }

    let mut merged: Vec<ColumnStatistics> = merged
        .into_iter()
        .map(|cs| cs.unwrap_or_else(ColumnStatistics::new_unknown))
        .collect();

    let files_pruned = selected_files.is_some() && matched_files < total_alive_files;
    if !files_pruned {
        let ndvs = ndv_from_metadata(&table, actual_snapshot.snapshot_id());
        for (cs, field) in merged.iter_mut().zip(&fields) {
            if let Some(&ndv) = ndvs.get(&field.id) {
                cs.distinct_count = Precision::Inexact(ndv);
            }
        }
    }

    let eq_columns = equality_filter_columns(filters);
    if !eq_columns.is_empty() {
        for (cs, field) in merged.iter_mut().zip(&fields) {
            if eq_columns.contains(field.name.as_str())
                && cs.distinct_count == Precision::Absent
                && uses_interval_cardinality_ratio(field.field_type.as_ref())
                && !is_singleton_bounds(&cs.min_value, &cs.max_value)
            {
                cs.min_value = Precision::Absent;
                cs.max_value = Precision::Absent;
            }
        }
    }

    // TODO: probably we can do something about the delete files?
    // However it wouldn't be free of cost in terms of performance
    if has_deletes || may_filter_rows {
        for cs in &mut merged {
            cs.null_count = cs.null_count.to_inexact();
        }
    }

    Ok(merged)
}

fn leaf_ids(field: &NestedFieldRef, ids: &mut Vec<i32>) {
    match field.field_type.as_ref() {
        Type::Primitive(_) | Type::Variant(_) => ids.push(field.id),
        Type::Struct(fields) => {
            for child in fields.fields() {
                leaf_ids(child, ids);
            }
        }
        Type::List(list) => leaf_ids(&list.element_field, ids),
        Type::Map(map) => {
            leaf_ids(&map.key_field, ids);
            leaf_ids(&map.value_field, ids);
        }
    }
}

/// Collects column names constrained by point-equality (`col = lit` or `col IN (lit, ...)`)
/// conjuncts in `filters`. Range predicates (`<`, `<=`, `>`, `>=`, `BETWEEN`) are excluded so
/// their `min_value`/`max_value` bounds remain available for range selectivity estimation.
fn equality_filter_columns(filters: &[Expr]) -> HashSet<&str> {
    let mut columns = HashSet::new();
    for filter in filters {
        collect_equality_columns(filter, &mut columns);
    }
    columns
}

fn collect_equality_columns<'a>(expr: &'a Expr, out: &mut HashSet<&'a str>) {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::And => {
            collect_equality_columns(&binary.left, out);
            collect_equality_columns(&binary.right, out);
        }
        Expr::BinaryExpr(binary) if binary.op == Operator::Eq => {
            if let Some(col) = column_name(&binary.left)
                && is_literal(&binary.right)
            {
                out.insert(col);
            } else if let Some(col) = column_name(&binary.right)
                && is_literal(&binary.left)
            {
                out.insert(col);
            }
        }
        Expr::InList(in_list) if !in_list.negated => {
            if let Some(col) = column_name(&in_list.expr)
                && !in_list.list.is_empty()
                && in_list.list.iter().all(is_literal)
            {
                out.insert(col);
            }
        }
        _ => {}
    }
}

fn column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(col) => Some(col.name.as_str()),
        Expr::Alias(alias) => column_name(&alias.expr),
        Expr::Cast(cast) => column_name(&cast.expr),
        Expr::TryCast(cast) => column_name(&cast.expr),
        _ => None,
    }
}

fn is_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_, _) => true,
        Expr::Cast(cast) => is_literal(&cast.expr),
        Expr::TryCast(cast) => is_literal(&cast.expr),
        _ => false,
    }
}

/// Returns `true` for Iceberg primitive types whose DataFusion `Interval::cardinality()` computes
/// a numeric/temporal range distance (`upper - lower`), causing `FilterExec` to estimate
/// `1 / (upper - lower + 1)` on point-equality predicates when `distinct_count` is `Absent`.
fn uses_interval_cardinality_ratio(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Primitive(
            PrimitiveType::Int
                | PrimitiveType::Long
                | PrimitiveType::Float
                | PrimitiveType::Double
                | PrimitiveType::Date
                | PrimitiveType::Timestamp
                | PrimitiveType::Timestamptz
                | PrimitiveType::Decimal { .. }
        )
    )
}

fn is_singleton_bounds(min: &Precision<ScalarValue>, max: &Precision<ScalarValue>) -> bool {
    match (min.get_value(), max.get_value()) {
        (Some(lo), Some(hi)) => !lo.is_null() && lo == hi,
        _ => false,
    }
}

/// Merging table's column statistics incrementally
fn merge_col_stats(acc: Option<ColumnStatistics>, next: ColumnStatistics) -> ColumnStatistics {
    match acc {
        None => next,
        Some(acc) => ColumnStatistics {
            null_count: acc.null_count.add(&next.null_count),
            min_value: acc.min_value.min(&next.min_value),
            max_value: acc.max_value.max(&next.max_value),
            byte_size: acc.byte_size.add(&next.byte_size),
            sum_value: Precision::Absent,
            distinct_count: Precision::Absent,
        },
    }
}

fn data_file_col_stats(
    df: &DataFile,
    id: i32,
    is_primitive: bool,
    leaf_ids: &[i32],
) -> ColumnStatistics {
    let byte_size = leaf_ids
        .iter()
        .try_fold(0_usize, |total, leaf_id| {
            let size = usize::try_from(*df.column_sizes().get(leaf_id)?).ok()?;
            total.checked_add(size)
        })
        .map(Precision::Inexact)
        .unwrap_or(Precision::Absent);

    if !is_primitive {
        return ColumnStatistics {
            byte_size,
            ..ColumnStatistics::new_unknown()
        };
    }

    ColumnStatistics {
        null_count: df
            .null_value_counts()
            .get(&id)
            .map(|n| Precision::Exact(*n as usize))
            .unwrap_or(Precision::Absent),
        min_value: df
            .lower_bounds()
            .get(&id)
            .and_then(datum_to_scalar)
            .map(Precision::Inexact)
            .unwrap_or(Precision::Absent),
        max_value: df
            .upper_bounds()
            .get(&id)
            .and_then(datum_to_scalar)
            .map(Precision::Inexact)
            .unwrap_or(Precision::Absent),
        byte_size,
        sum_value: Precision::Absent,
        distinct_count: Precision::Absent,
    }
}

/// Conversion function of iceberg's Datum
fn datum_to_scalar(d: &Datum) -> Option<ScalarValue> {
    match (d.data_type(), d.literal()) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(v)) => {
            Some(ScalarValue::Boolean(Some(*v)))
        }
        (PrimitiveType::Int, PrimitiveLiteral::Int(v)) => Some(ScalarValue::Int32(Some(*v))),
        (PrimitiveType::Long, PrimitiveLiteral::Long(v)) => Some(ScalarValue::Int64(Some(*v))),
        (PrimitiveType::Float, PrimitiveLiteral::Float(v)) => {
            Some(ScalarValue::Float32(Some(v.into_inner())))
        }
        (PrimitiveType::Double, PrimitiveLiteral::Double(v)) => {
            Some(ScalarValue::Float64(Some(v.into_inner())))
        }
        (PrimitiveType::String, PrimitiveLiteral::String(s)) => {
            Some(ScalarValue::Utf8(Some(s.clone())))
        }
        (PrimitiveType::Date, PrimitiveLiteral::Int(v)) => Some(ScalarValue::Date32(Some(*v))),
        (PrimitiveType::Timestamp, PrimitiveLiteral::Long(v)) => {
            Some(ScalarValue::TimestampMicrosecond(Some(*v), None))
        }
        (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(v)) => Some(
            ScalarValue::TimestampMicrosecond(Some(*v), Some("UTC".into())),
        ),
        (PrimitiveType::Decimal { precision, scale }, PrimitiveLiteral::Int128(v)) => Some(
            ScalarValue::Decimal128(Some(*v), *precision as u8, *scale as i8),
        ),
        _ => None,
    }
}
