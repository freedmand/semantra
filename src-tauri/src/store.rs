//! Persistent, file-aware chunk store backed by LanceDB.
//!
//! Holds one `chunks` table (shared with the [`Catalog`](crate::catalog) tables
//! in the same connection / on-disk directory). Each row is one embeddable
//! window plus the metadata needed to (a) attribute it to a source file
//! (`sha512`), (b) map it back to an exact source span for highlighting
//! (`char_start/char_end` in the flat text, `page` + `page_char_start` in PDF
//! page space), and (c) know which pipeline produced it (`pipeline_version`).
//!
//! Vectors are the model's unit-norm embeddings, so cosine distance is the right
//! metric and `score = 1 - distance` is a true cosine similarity in `[0, 1]`.
//! Two retrieval paths are exposed: dense vector search (exact brute-force or the
//! approximate IVF-HNSW-SQ index) and BM25 **full-text search** over an inverted
//! index on `text`. Both can be restricted to a set of files (a project's
//! members). Higher-level query semantics (semantic + quoted-literal keyword
//! filtering) are composed from these primitives in `lib.rs`.

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType, Table};

/// Below this row count an ANN index is pointless — brute-force over a few
/// hundred vectors is already sub-millisecond. We (re)build past this and fall
/// back to exact search until then.
const MIN_ANN_ROWS: usize = 256;

const TABLE_NAME: &str = "chunks";

/// How a dense query should be answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    /// Exhaustive brute-force scan — exact nearest neighbours, no index.
    Exact,
    /// Approximate search via the IVF-HNSW-SQ index (falls back to a full scan
    /// automatically when no index has been built yet).
    Ann,
}

impl SearchMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "exact" => Ok(SearchMode::Exact),
            "ann" => Ok(SearchMode::Ann),
            other => Err(format!("unknown search mode {other:?} (want \"exact\" or \"ann\")")),
        }
    }
}

/// A chunk to insert, with its embedding. The store assigns the row `id`.
#[derive(Clone, Debug)]
pub struct ChunkRow {
    pub sha512: String,
    pub text: String,
    pub char_start: i64,
    pub char_end: i64,
    pub page: Option<i64>,
    pub page_char_start: i64,
    pub pipeline_version: String,
    pub vector: Vec<f32>,
}

/// A single search hit, carrying enough metadata for the UI to group by file and
/// navigate into the document. Serialized camelCase for the frontend.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hit {
    pub id: i64,
    pub sha512: String,
    pub text: String,
    /// Cosine distance for dense hits; BM25-derived pseudo-distance for FTS hits.
    pub distance: f32,
    pub score: f32,
    pub char_start: i64,
    pub char_end: i64,
    pub page: Option<i64>,
    pub page_char_start: i64,
}

/// Persistent LanceDB-backed chunk store.
pub struct VectorStore {
    conn: Connection,
    /// `None` until the first insert creates the table.
    table: Option<Table>,
    dim: i32,
    /// Monotonic row id (never reused, even after deletes).
    next_id: i64,
    ann_built: bool,
    rows_at_last_index: usize,
    fts_built: bool,
}

impl VectorStore {
    /// Open (or re-open) the chunk store on an existing connection, sized to the
    /// model's embedding `dim`. Reuses the `chunks` table if it already exists,
    /// resuming the id counter past the highest stored id.
    pub async fn open(conn: Connection, dim: i32) -> Result<Self, String> {
        let names = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| format!("list tables: {e}"))?;

        let (table, next_id) = if names.iter().any(|n| n == TABLE_NAME) {
            let table = conn
                .open_table(TABLE_NAME)
                .execute()
                .await
                .map_err(|e| format!("open chunks: {e}"))?;
            let next_id = max_id(&table).await? + 1;
            (Some(table), next_id)
        } else {
            (None, 0)
        };

        Ok(Self {
            conn,
            table,
            dim,
            next_id,
            ann_built: false,
            rows_at_last_index: 0,
            fts_built: names.iter().any(|n| n == TABLE_NAME),
        })
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sha512", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("char_start", DataType::Int64, false),
            Field::new("char_end", DataType::Int64, false),
            Field::new("page", DataType::Int64, true),
            Field::new("page_char_start", DataType::Int64, false),
            Field::new("pipeline_version", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim,
                ),
                true,
            ),
        ]))
    }

    /// Highest id ever assigned + 1 (also the count of inserts, ignoring deletes).
    pub fn next_id(&self) -> i64 {
        self.next_id
    }

    fn build_batch(&self, start_id: i64, rows: &[ChunkRow]) -> Result<RecordBatch, String> {
        let ids = Int64Array::from_iter_values(start_id..start_id + rows.len() as i64);
        let sha = StringArray::from(rows.iter().map(|r| r.sha512.as_str()).collect::<Vec<_>>());
        let text = StringArray::from(rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>());
        let cstart = Int64Array::from(rows.iter().map(|r| r.char_start).collect::<Vec<_>>());
        let cend = Int64Array::from(rows.iter().map(|r| r.char_end).collect::<Vec<_>>());
        let page = Int64Array::from(rows.iter().map(|r| r.page).collect::<Vec<_>>());
        let pstart = Int64Array::from(rows.iter().map(|r| r.page_char_start).collect::<Vec<_>>());
        let pv = StringArray::from(
            rows.iter().map(|r| r.pipeline_version.as_str()).collect::<Vec<_>>(),
        );
        let vec_arr = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            rows.iter()
                .map(|r| Some(r.vector.iter().map(|&x| Some(x)).collect::<Vec<_>>())),
            self.dim,
        );
        RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(ids),
                Arc::new(sha),
                Arc::new(text),
                Arc::new(cstart),
                Arc::new(cend),
                Arc::new(page),
                Arc::new(pstart),
                Arc::new(pv),
                Arc::new(vec_arr),
            ],
        )
        .map_err(|e| format!("build record batch: {e}"))
    }

    /// Insert a batch of chunks, creating the table on first call.
    pub async fn insert(&mut self, rows: &[ChunkRow]) -> Result<(), String> {
        if rows.is_empty() {
            return Ok(());
        }
        let batch = self.build_batch(self.next_id, rows)?;
        let schema = batch.schema();
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

        match &self.table {
            None => {
                let table = self
                    .conn
                    .create_table(TABLE_NAME, reader)
                    .execute()
                    .await
                    .map_err(|e| format!("create table: {e}"))?;
                self.table = Some(table);
            }
            Some(table) => {
                table
                    .add(reader)
                    .execute()
                    .await
                    .map_err(|e| format!("append rows: {e}"))?;
            }
        }
        self.next_id += rows.len() as i64;
        Ok(())
    }

    /// (Re)build the ANN index when the table is large enough and has grown
    /// meaningfully since the last build.
    pub async fn maybe_build_ann(&mut self) -> Result<(), String> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        let rows = table
            .count_rows(None)
            .await
            .map_err(|e| format!("count rows: {e}"))?;
        if rows < MIN_ANN_ROWS {
            return Ok(());
        }
        if self.ann_built && rows < self.rows_at_last_index * 5 / 4 {
            return Ok(());
        }
        let num_partitions = (rows as f64).sqrt().round().max(1.0) as u32;
        let builder = IvfHnswSqIndexBuilder::default()
            .distance_type(DistanceType::Cosine)
            .num_partitions(num_partitions);
        table
            .create_index(&["vector"], Index::IvfHnswSq(builder))
            .execute()
            .await
            .map_err(|e| format!("build ANN index: {e}"))?;
        self.ann_built = true;
        self.rows_at_last_index = rows;
        Ok(())
    }

    /// Ensure a BM25 full-text index exists on `text`. Safe to call repeatedly;
    /// rebuilding refreshes it to include newly inserted rows.
    pub async fn ensure_fts_index(&mut self) -> Result<(), String> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
            .map_err(|e| format!("build FTS index: {e}"))?;
        self.fts_built = true;
        Ok(())
    }

    /// Dense nearest-neighbour search restricted to `sha_filter` files.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        mode: SearchMode,
        sha_filter: &[String],
    ) -> Result<Vec<Hit>, String> {
        let Some(table) = &self.table else {
            return Ok(Vec::new());
        };
        if sha_filter.is_empty() {
            return Ok(Vec::new());
        }
        let mut q = table
            .query()
            .only_if(sha_in(sha_filter))
            .nearest_to(query)
            .map_err(|e| format!("build vector query: {e}"))?
            .distance_type(DistanceType::Cosine)
            .limit(k);
        match mode {
            SearchMode::Exact => q = q.bypass_vector_index(),
            SearchMode::Ann => q = q.nprobes(20).refine_factor(10),
        }
        let batches: Vec<RecordBatch> = q
            .execute()
            .await
            .map_err(|e| format!("execute query: {e}"))?
            .try_collect()
            .await
            .map_err(|e| format!("collect results: {e}"))?;

        let mut hits = rows_to_hits(&batches, "_distance", true)?;
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits.truncate(k);
        Ok(hits)
    }

    /// BM25 full-text search for `term`, restricted to `sha_filter` files.
    pub async fn fts_search(
        &self,
        term: &str,
        k: usize,
        sha_filter: &[String],
    ) -> Result<Vec<Hit>, String> {
        let Some(table) = &self.table else {
            return Ok(Vec::new());
        };
        if sha_filter.is_empty() {
            return Ok(Vec::new());
        }
        let batches: Vec<RecordBatch> = table
            .query()
            .only_if(sha_in(sha_filter))
            .full_text_search(FullTextSearchQuery::new(term.to_string()))
            .limit(k)
            .execute()
            .await
            .map_err(|e| format!("execute fts query: {e}"))?
            .try_collect()
            .await
            .map_err(|e| format!("collect fts results: {e}"))?;
        // FTS returns a BM25 `_score` (higher = better); map to a pseudo-distance
        // so callers can treat all hits uniformly.
        let mut hits = rows_to_hits(&batches, "_score", false)?;
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(k);
        Ok(hits)
    }

    /// Delete every chunk belonging to a file (used when a file is GC'd).
    pub async fn delete_file(&mut self, sha512: &str) -> Result<(), String> {
        if let Some(table) = &self.table {
            table
                .delete(&format!("sha512 = '{sha512}'"))
                .await
                .map_err(|e| format!("delete file chunks: {e}"))?;
        }
        Ok(())
    }

    /// Number of chunks belonging to the given files.
    pub async fn count_for(&self, sha_filter: &[String]) -> Result<usize, String> {
        let Some(table) = &self.table else {
            return Ok(0);
        };
        if sha_filter.is_empty() {
            return Ok(0);
        }
        table
            .count_rows(Some(sha_in(sha_filter)))
            .await
            .map_err(|e| format!("count rows: {e}"))
    }
}

/// Build a `sha512 IN ('a','b',...)` predicate. SHA-512 hex is a safe charset.
fn sha_in(shas: &[String]) -> String {
    let list = shas
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("sha512 IN ({list})")
}

/// Highest `id` currently stored, or -1 if empty.
async fn max_id(table: &Table) -> Result<i64, String> {
    let batches: Vec<RecordBatch> = table
        .query()
        .execute()
        .await
        .map_err(|e| format!("scan ids: {e}"))?
        .try_collect()
        .await
        .map_err(|e| format!("collect ids: {e}"))?;
    let mut max = -1i64;
    for b in &batches {
        if let Some(ids) = b.column_by_name("id").and_then(|c| c.as_any().downcast_ref::<Int64Array>()) {
            for i in 0..ids.len() {
                max = max.max(ids.value(i));
            }
        }
    }
    Ok(max)
}

/// Convert result batches into [`Hit`]s. `metric_col` is `_distance` (cosine) or
/// `_score` (BM25); `distance_is_cosine` controls how `score`/`distance` are
/// derived so both paths populate the same struct.
fn rows_to_hits(batches: &[RecordBatch], metric_col: &str, distance_is_cosine: bool) -> Result<Vec<Hit>, String> {
    let mut hits = Vec::new();
    for b in batches {
        let s = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let n = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let f = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        let (Some(ids), Some(sha), Some(text), Some(cstart), Some(cend), Some(pstart)) = (
            n("id"),
            s("sha512"),
            s("text"),
            n("char_start"),
            n("char_end"),
            n("page_char_start"),
        ) else {
            continue;
        };
        let page = n("page");
        let metric = f(metric_col);
        for i in 0..b.num_rows() {
            let m = metric.map(|a| a.value(i)).unwrap_or(0.0);
            let (distance, score) = if distance_is_cosine {
                (m, (1.0 - m).clamp(0.0, 1.0))
            } else {
                // BM25 score: keep it as the ranking score; distance is unused.
                (0.0, m)
            };
            hits.push(Hit {
                id: ids.value(i),
                sha512: sha.value(i).to_string(),
                text: text.value(i).to_string(),
                distance,
                score,
                char_start: cstart.value(i),
                char_end: cend.value(i),
                page: page.and_then(|p| if p.is_null(i) { None } else { Some(p.value(i)) }),
                page_char_start: pstart.value(i),
            });
        }
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store(dim: i32) -> (VectorStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let conn = lancedb::connect(tmp.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();
        (VectorStore::open(conn, dim).await.unwrap(), tmp)
    }

    fn row(sha: &str, text: &str, vector: Vec<f32>) -> ChunkRow {
        ChunkRow {
            sha512: sha.into(),
            text: text.into(),
            char_start: 0,
            char_end: text.chars().count() as i64,
            page: None,
            page_char_start: 0,
            pipeline_version: "v1".into(),
            vector,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn search_is_scoped_to_files_and_carries_metadata() {
        let (mut store, _t) = temp_store(2).await;
        store
            .insert(&[
                row("fileA", "alpha", vec![1.0, 0.0]),
                row("fileB", "beta", vec![0.0, 1.0]),
            ])
            .await
            .unwrap();

        // Restrict to fileB: even a query pointing at fileA's vector only returns B.
        let hits = store
            .search(&[1.0, 0.0], 5, SearchMode::Exact, &["fileB".into()])
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].sha512, "fileB");

        // Empty filter → nothing.
        assert!(store
            .search(&[1.0, 0.0], 5, SearchMode::Exact, &[])
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_file_removes_its_chunks() {
        let (mut store, _t) = temp_store(2).await;
        store
            .insert(&[row("a", "x", vec![1.0, 0.0]), row("b", "y", vec![0.0, 1.0])])
            .await
            .unwrap();
        store.delete_file("a").await.unwrap();
        assert_eq!(store.count_for(&["a".into()]).await.unwrap(), 0);
        assert_eq!(store.count_for(&["b".into()]).await.unwrap(), 1);
    }
}
