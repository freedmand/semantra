//! Indexing phase-breakdown benchmark — answers "where does indexing time go?"
//! by running the REAL pipeline pieces the worker uses (`embed_sentences` on the
//! bundled model, then `VectorStore::insert` + `maintain` on a temp LanceDB) and
//! timing each phase separately.
//!
//! It deliberately isolates the two big costs so we can see LanceDB's share:
//!   • EMBED   — candle/Metal inference, batched 32 like `embed::BATCH_SIZE`
//!   • INSERT  — LanceDB appends, batched 1024 like `lib::INSERT_BATCH`
//!   • MAINTAIN— LanceDB ANN (IVF-HNSW-SQ) + FTS index build + compaction
//!
//! Skipped unless `BENCH=1` (it needs the model and takes seconds). Knobs:
//!   BENCH=1                 enable the bench
//!   BENCH_N=<n>             corpus size in chunks (default 5000)
//!   BENCH_SKIP_STORE=1      embed only — no insert/maintain (clean embed timing)
//!   LEAF_IR_DEVICE=cpu|metal  force the embed device (prints which it used)
//!
//! Run (dev profile — deps already opt-3 via [profile.dev.package."*"]):
//!   BENCH=1 cargo test --test index_bench -- --nocapture
//! Run (release profile):
//!   BENCH=1 cargo test --release --test index_bench -- --nocapture

use std::path::PathBuf;
use std::time::Instant;

use leaf_ir_candle_test::{embed_sentences, setup_model, warmup};
use semantra_lib::store::{ChunkRow, VectorStore};

// Mirror the production constants (the modules that own them aren't public).
// BATCH_SIZE is overridable via BENCH_BATCH to sweep the Metal-dispatch tradeoff.
const BATCH_SIZE: usize = 32; // embed::BATCH_SIZE (production default)
const INSERT_BATCH: usize = 1024; // lib::INSERT_BATCH
const CHUNK_WORDS: usize = 50; // lib::DEFAULT_CHUNK_SIZE

/// A small, varied vocabulary so tokenization isn't degenerate (all-identical
/// tokens would mis-estimate real throughput). ~80 common English words.
const VOCAB: &[&str] = &[
    "the", "model", "search", "vector", "index", "document", "embedding", "query",
    "result", "score", "page", "chunk", "text", "data", "system", "memory", "file",
    "process", "value", "table", "field", "store", "build", "time", "speed", "size",
    "device", "metal", "kernel", "matrix", "batch", "token", "word", "language",
    "machine", "learning", "neural", "network", "feature", "vectorize", "distance",
    "cosine", "similar", "nearest", "neighbor", "cluster", "partition", "graph",
    "node", "edge", "weight", "bias", "layer", "dense", "pool", "mean", "norm",
    "unit", "scale", "compact", "merge", "insert", "append", "delete", "filter",
    "scan", "row", "column", "schema", "type", "string", "number", "float", "byte",
    "hash", "content", "address", "project", "import", "export", "pipeline", "stage",
];

/// Tiny deterministic PRNG (no `rand` dep), matching the style in vector_store.rs.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
}

/// Build `n` distinct chunks of ~CHUNK_WORDS words, deterministically.
fn synth_corpus(n: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (i as u64).wrapping_mul(2654435761));
        let mut words = Vec::with_capacity(CHUNK_WORDS);
        for _ in 0..CHUNK_WORDS {
            words.push(VOCAB[(rng.next_u64() as usize) % VOCAB.len()]);
        }
        out.push(words.join(" "));
    }
    out
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn pct(part: f64, whole: f64) -> f64 {
    if whole <= 0.0 { 0.0 } else { part / whole * 100.0 }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn index_phase_breakdown() {
    if std::env::var("BENCH").is_err() {
        eprintln!("skipping index_bench (set BENCH=1 to run)");
        return;
    }
    let n = env_usize("BENCH_N", 5000);
    let batch_size = env_usize("BENCH_BATCH", BATCH_SIZE);
    let skip_store = std::env::var("BENCH_SKIP_STORE").is_ok();

    // Load the active model the same way the app does in dev.
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mdbr-leaf-mt");
    let ctx = setup_model(&model_dir).expect("load bundled mdbr-leaf-mt model");
    let dim = ctx.embedding_dim();
    warmup(&ctx).expect("warmup"); // compiles Metal pipelines off the clock

    let corpus = synth_corpus(n);
    let total_words = n * CHUNK_WORDS;
    eprintln!(
        "\n=== index bench: {n} chunks (~{CHUNK_WORDS} words each, {total_words} words), dim {dim}, skip_store={skip_store} ===",
    );

    // ---- Phase EMBED: batched inference, collecting vectors -----------------
    let t_embed = Instant::now();
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
    for batch in corpus.chunks(batch_size) {
        let emb = embed_sentences(&ctx, batch).expect("embed batch");
        vectors.extend(emb.rows);
    }
    let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(vectors.len(), n);

    if skip_store {
        eprintln!(
            "EMBED only (batch {batch_size}): {embed_ms:.0} ms  ({:.0} chunks/s, {:.1} ms/batch)\n",
            n as f64 / (embed_ms / 1000.0),
            embed_ms / (n as f64 / batch_size as f64),
        );
        return;
    }

    // Materialize ChunkRows (cheap; excluded from phase timers below).
    let rows: Vec<ChunkRow> = corpus
        .iter()
        .zip(vectors.into_iter())
        .enumerate()
        .map(|(i, (text, vector))| ChunkRow {
            sha512: "bench".into(),
            text: text.clone(),
            char_start: i as i64,
            char_end: (i + text.len()) as i64,
            page: None,
            page_char_start: 0,
            pipeline_version: "bench".into(),
            vector,
        })
        .collect();

    // Fresh temp LanceDB store sized to the model dim.
    let tmp = tempfile::tempdir().unwrap();
    let conn = lancedb::connect(tmp.path().to_str().unwrap())
        .execute()
        .await
        .unwrap();
    let mut store = VectorStore::open(conn, dim as i32).await.unwrap();

    // ---- Phase INSERT: LanceDB appends, batched like production -------------
    let t_insert = Instant::now();
    for batch in rows.chunks(INSERT_BATCH) {
        store.insert(batch).await.expect("insert batch");
    }
    let insert_ms = t_insert.elapsed().as_secs_f64() * 1000.0;

    // ---- Phase MAINTAIN: ANN + FTS index build + compaction -----------------
    let t_maint = Instant::now();
    store.maintain().await.expect("maintain");
    let maint_ms = t_maint.elapsed().as_secs_f64() * 1000.0;

    let lancedb_ms = insert_ms + maint_ms;
    let total_ms = embed_ms + lancedb_ms;

    eprintln!("\n--- phase breakdown ({n} chunks) ---");
    eprintln!("  EMBED    {embed_ms:>9.0} ms   {:>5.1}%   ({:.0} chunks/s)", pct(embed_ms, total_ms), n as f64 / (embed_ms / 1000.0));
    eprintln!("  INSERT   {insert_ms:>9.0} ms   {:>5.1}%", pct(insert_ms, total_ms));
    eprintln!("  MAINTAIN {maint_ms:>9.0} ms   {:>5.1}%", pct(maint_ms, total_ms));
    eprintln!("  ──────────────────────────────");
    eprintln!("  LanceDB  {lancedb_ms:>9.0} ms   {:>5.1}%   (insert + maintain)", pct(lancedb_ms, total_ms));
    eprintln!("  TOTAL    {total_ms:>9.0} ms  100.0%\n");
    eprintln!(
        "  >> LanceDB is {:.1}% of indexing time. opt-level only moves this slice.\n",
        pct(lancedb_ms, total_ms),
    );
}
