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
use lancedb::index::scalar::{
    BooleanQuery, FtsIndexBuilder, FtsQuery, FullTextSearchQuery, MatchQuery, Occur, Operator,
};
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::optimize::Duration;
use lancedb::table::{CompactionOptions, OptimizeAction};
use lancedb::{Connection, DistanceType, Table};

/// Below this row count an ANN index is pointless — brute-force over a few
/// hundred vectors is already sub-millisecond. We (re)build past this and fall
/// back to exact search until then.
const MIN_ANN_ROWS: usize = 256;

const TABLE_NAME: &str = "chunks";

/// N-gram length for the substring (keyword-prefix) FTS index. The `text` column
/// is tokenized into trigrams so `full_text_search` can match *within* words
/// (`green` inside `greenhouse`) via the inverted index instead of a linear
/// `contains` scan — the difference that lets keyword filtering scale to large
/// corpora. Trigrams (3) are the standard selectivity sweet spot: bigrams are
/// far less selective (huge postings) and longer grams miss short queries.
/// Quoted-keyword tokens shorter than this can't be tokenized, so they're
/// dropped upstream (see `query::MIN_PREFIX_LEN`, kept equal to this).
const NGRAM_LEN: u32 = 3;

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
    /// Row count at the last `maintain()` (index rebuild + compaction). Drives
    /// the worker's "rows added since the indexes were last refreshed" trigger.
    rows_at_last_maintenance: usize,
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

        // Detect which indexes actually exist (rather than assuming they do
        // whenever the table exists): the FTS/ANN builds are deferred off the
        // per-file path now, so a table can exist with no indexes yet — e.g. an
        // import that inserted rows but was killed before the post-drain rebuild.
        //
        // Seed `rows_at_last_maintenance` to the number of rows the FTS index
        // *actually covers* (0 if there's no index). That way reopening a fully
        // indexed store reports 0 rows pending (no spurious rebuild), while any
        // unindexed tail left by a crash reports as pending and the worker
        // rebuilds it on its first drain — keyword search self-heals on restart.
        let (ann_built, fts_built, rows, indexed_rows) = match &table {
            Some(t) => {
                let indices = t
                    .list_indices()
                    .await
                    .map_err(|e| format!("list indices: {e}"))?;
                let on_col =
                    |col: &str| indices.iter().find(|i| i.columns.iter().any(|c| c == col));
                let rows = t
                    .count_rows(None)
                    .await
                    .map_err(|e| format!("count rows: {e}"))?;
                let indexed_rows = match on_col("text") {
                    Some(ix) => t
                        .index_stats(&ix.name)
                        .await
                        .map_err(|e| format!("fts index stats: {e}"))?
                        .map(|s| s.num_indexed_rows)
                        .unwrap_or(0),
                    None => 0,
                };
                (on_col("vector").is_some(), on_col("text").is_some(), rows, indexed_rows)
            }
            None => (false, false, 0, 0),
        };

        Ok(Self {
            conn,
            table,
            dim,
            next_id,
            ann_built,
            rows_at_last_index: rows,
            fts_built,
            rows_at_last_maintenance: indexed_rows,
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

    /// Ensure the trigram full-text index exists on `text`. Safe to call
    /// repeatedly; `create_index` replaces, so rebuilding both refreshes the
    /// index with newly inserted rows and (re)applies the tokenizer config.
    ///
    /// The index uses an **n-gram tokenizer** (not the default word tokenizer)
    /// so `full_text_search` does substring matching — see [`NGRAM_LEN`]. This
    /// trades word-level BM25 ranking for substring capability + scale; keyword
    /// results rank by trigram-overlap BM25 instead. `lower_case` makes matching
    /// case-insensitive (the same analyzer lowercases the query side too).
    pub async fn ensure_fts_index(&mut self) -> Result<(), String> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        let params = FtsIndexBuilder::default()
            .base_tokenizer("ngram".to_string())
            .ngram_min_length(NGRAM_LEN)
            .ngram_max_length(NGRAM_LEN)
            .lower_case(true);
        table
            .create_index(&["text"], Index::FTS(params))
            .execute()
            .await
            .map_err(|e| format!("build n-gram FTS index: {e}"))?;
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

    /// Substring candidate search backing the quoted-keyword *prefix* filter.
    /// Returns rows whose `text` contains every `token` as a substring (case-
    /// insensitive), restricted to `sha_filter`, scored by trigram-overlap BM25
    /// (higher = better). This is a coarse superset — *substring*, not prefix —
    /// which the handler in `lib.rs` narrows to word-prefix matches.
    ///
    /// Accelerated by the n-gram FTS index ([`ensure_fts_index`]): each token is
    /// tokenized into trigrams that must **all** match (`Operator::And`), so the
    /// candidate set is tight (≈ docs that actually contain the substring, modulo
    /// rare trigram coincidences the prefix recheck drops). Multiple tokens (a
    /// multi-word literal) are ANDed as separate clauses so they may match
    /// anywhere/independently — matching the handler's per-token prefix recheck,
    /// not a contiguous phrase. Tokens shorter than [`NGRAM_LEN`] can't be
    /// tokenized and are skipped (the caller already drops them upstream).
    ///
    /// [`ensure_fts_index`]: Self::ensure_fts_index
    pub async fn substring_search(
        &self,
        tokens: &[String],
        k: usize,
        sha_filter: &[String],
    ) -> Result<Vec<Hit>, String> {
        let Some(table) = &self.table else {
            return Ok(Vec::new());
        };
        // `full_text_search` is only valid once the FTS index exists, and it has
        // no scan fallback (unlike vector search). The index build is deferred to
        // `maintain()`, so before the first refresh there's simply nothing to
        // match against — yield no candidates rather than erroring. Keyword
        // results become available after the next maintenance pass.
        if !self.fts_built {
            return Ok(Vec::new());
        }
        if sha_filter.is_empty() {
            return Ok(Vec::new());
        }
        let clauses: Vec<FtsQuery> = tokens
            .iter()
            .filter(|t| t.chars().count() >= NGRAM_LEN as usize)
            .map(|t| {
                FtsQuery::Match(
                    MatchQuery::new(t.clone())
                        .with_column(Some("text".to_string()))
                        // All trigrams of the token must be present…
                        .with_operator(Operator::And)
                        // …and matched exactly — no fuzzy trigram expansion.
                        .with_fuzziness(Some(0)),
                )
            })
            .collect();
        if clauses.is_empty() {
            return Ok(Vec::new());
        }
        // One token → a plain Match; several → all required (Boolean MUST).
        let query = if clauses.len() == 1 {
            clauses.into_iter().next().unwrap()
        } else {
            FtsQuery::Boolean(BooleanQuery::new(
                clauses.into_iter().map(|c| (Occur::Must, c)),
            ))
        };
        let batches: Vec<RecordBatch> = table
            .query()
            .only_if(sha_in(sha_filter))
            .full_text_search(FullTextSearchQuery::new_query(query))
            .limit(k)
            .execute()
            .await
            .map_err(|e| format!("execute substring query: {e}"))?
            .try_collect()
            .await
            .map_err(|e| format!("collect substring results: {e}"))?;
        let mut hits = rows_to_hits(&batches, "_score", false)?;
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(k);
        Ok(hits)
    }

    /// Reclaim disk by compacting fragments and pruning stale dataset/index
    /// versions. Every append creates new fragments and every `create_index`
    /// (re)build is additive — old versions linger on disk, so repeated
    /// re-indexing bloats the store (observed ~250 MB of mostly stale index
    /// versions). Compaction merges small fragments; the prune drops every
    /// version but the current one to free that space.
    ///
    /// `delete_unverified` + `older_than = 0` deletes all non-current versions,
    /// which would corrupt an in-flight reader on a different snapshot. This is
    /// safe **only** because all store access is serialized behind the async
    /// mutex in `lib.rs`: the post-index call site holds the guard exclusively,
    /// so no reader is mid-scan on an older version.
    pub async fn compact(&mut self) -> Result<(), String> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        table
            .optimize(OptimizeAction::Compact {
                options: CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .map_err(|e| format!("compact files: {e}"))?;
        table
            .optimize(OptimizeAction::Prune {
                older_than: Some(Duration::seconds(0)),
                delete_unverified: Some(true),
                error_if_tagged_old_versions: Some(false),
            })
            .await
            .map_err(|e| format!("prune old versions: {e}"))?;
        Ok(())
    }

    /// Current committed row count (0 before the table exists).
    async fn current_rows(&self) -> Result<usize, String> {
        match &self.table {
            Some(table) => table
                .count_rows(None)
                .await
                .map_err(|e| format!("count rows: {e}")),
            None => Ok(0),
        }
    }

    /// Rows inserted since the last [`maintain`](Self::maintain). Lets the
    /// indexing worker decide whether enough new chunks have accumulated to
    /// justify a mid-run index refresh.
    pub async fn rows_since_maintenance(&self) -> Result<usize, String> {
        Ok(self
            .current_rows()
            .await?
            .saturating_sub(self.rows_at_last_maintenance))
    }

    /// Refresh the search indexes and reclaim disk in one exclusive pass: build
    /// the ANN index if it has grown enough, (re)build the n-gram FTS index, and
    /// compact + prune. Deliberately kept *off* the per-file indexing path — each
    /// of these rebuilds over the whole table, so running them per file makes a
    /// multi-file import O(files²). The worker calls this once when the queue
    /// drains and occasionally mid-run (see `MAINTENANCE_ROW_THRESHOLD`).
    pub async fn maintain(&mut self) -> Result<(), String> {
        self.maybe_build_ann().await?;
        self.ensure_fts_index().await?;
        self.compact().await?;
        self.rows_at_last_maintenance = self.current_rows().await?;
        Ok(())
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

/// Drop the `chunks` table entirely. No-op if it doesn't exist yet.
///
/// Used when the active embedding model changes: every stored vector is then from
/// a different model (and possibly a different dimension), so the table is dropped
/// and recreated at the new model's dim on the next [`VectorStore::insert`]. Run
/// this on the shared connection *before* opening the store, so it reopens clean.
pub async fn drop_chunks(conn: &Connection) -> Result<(), String> {
    let names = conn
        .table_names()
        .execute()
        .await
        .map_err(|e| format!("list tables: {e}"))?;
    if names.iter().any(|n| n == TABLE_NAME) {
        conn.drop_table(TABLE_NAME, &[])
            .await
            .map_err(|e| format!("drop chunks table: {e}"))?;
    }
    Ok(())
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
    async fn substring_search_is_case_insensitive_and_substring() {
        let (mut store, _t) = temp_store(2).await;
        store
            .insert(&[
                row("f", "The Greenhouse effect", vec![1.0, 0.0]),
                row("f", "evergreen forest", vec![0.0, 1.0]),
                row("f", "a house of cards", vec![1.0, 1.0]),
            ])
            .await
            .unwrap();
        store.ensure_fts_index().await.unwrap();

        // `green` is a substring of both "Greenhouse" (case-insensitive, via the
        // lower-casing trigram index) and "evergreen" — substring_search is the
        // coarse superset, so it returns both. (The handler's prefix recheck
        // later drops "evergreen".) "house" lacks the `gre` trigram → excluded.
        let hits = store
            .substring_search(&["green".into()], 10, &["f".into()])
            .await
            .unwrap();
        let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(hits.len(), 2, "got {texts:?}");
        assert!(texts.contains(&"The Greenhouse effect"));
        assert!(texts.contains(&"evergreen forest"));

        // No trigram match → empty; empty filter / no usable tokens → empty.
        assert!(store
            .substring_search(&["zzz".into()], 10, &["f".into()])
            .await
            .unwrap()
            .is_empty());
        assert!(store
            .substring_search(&["green".into()], 10, &[])
            .await
            .unwrap()
            .is_empty());
        // Token shorter than the trigram length is skipped → no clauses → empty.
        assert!(store
            .substring_search(&["gr".into()], 10, &["f".into()])
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn substring_search_multi_token_requires_all() {
        let (mut store, _t) = temp_store(2).await;
        store
            .insert(&[
                row("f", "new yorker magazine", vec![1.0, 0.0]),
                row("f", "new jersey transit", vec![0.0, 1.0]),
            ])
            .await
            .unwrap();
        store.ensure_fts_index().await.unwrap();

        // Both tokens must be substrings (independently). "new yorker" has both
        // `new` and `york`; "new jersey" has `new` but not `york`.
        let hits = store
            .substring_search(&["new".into(), "york".into()], 10, &["f".into()])
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "new yorker magazine");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compact_runs_clean_on_a_populated_table() {
        let (mut store, _t) = temp_store(2).await;
        store
            .insert(&[row("a", "x", vec![1.0, 0.0]), row("b", "y", vec![0.0, 1.0])])
            .await
            .unwrap();
        // Compaction + prune should succeed and leave the data intact.
        store.compact().await.unwrap();
        assert_eq!(store.count_for(&["a".into(), "b".into()]).await.unwrap(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn substring_search_yields_nothing_until_index_built() {
        let (mut store, _t) = temp_store(2).await;
        store
            .insert(&[row("f", "The Greenhouse effect", vec![1.0, 0.0])])
            .await
            .unwrap();

        // No FTS index yet (the build is deferred to `maintain`): keyword search
        // returns nothing rather than erroring on the missing index.
        assert_eq!(store.rows_since_maintenance().await.unwrap(), 1);
        assert!(store
            .substring_search(&["green".into()], 10, &["f".into()])
            .await
            .unwrap()
            .is_empty());

        // After a maintenance pass the FTS index exists and the same query hits.
        store.maintain().await.unwrap();
        assert_eq!(store.rows_since_maintenance().await.unwrap(), 0);
        let hits = store
            .substring_search(&["green".into()], 10, &["f".into()])
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "The Greenhouse effect");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reopen_reports_unindexed_rows_as_pending_maintenance() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let open = || async {
            let conn = lancedb::connect(uri).execute().await.unwrap();
            VectorStore::open(conn, 2).await.unwrap()
        };

        // Insert rows but never maintain — simulates an import killed before the
        // post-drain index rebuild.
        {
            let mut store = open().await;
            store
                .insert(&[row("f", "The Greenhouse effect", vec![1.0, 0.0])])
                .await
                .unwrap();
        }

        // Reopen: the FTS index doesn't exist, so all rows read as pending and a
        // maintenance pass is owed (the worker runs one on its first drain).
        let mut store = open().await;
        assert_eq!(store.rows_since_maintenance().await.unwrap(), 1);
        store.maintain().await.unwrap();

        // Reopen again after a clean maintenance: nothing pending, no rebuild owed.
        drop(store);
        let store = open().await;
        assert_eq!(store.rows_since_maintenance().await.unwrap(), 0);
        assert!(!store
            .substring_search(&["green".into()], 10, &["f".into()])
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
