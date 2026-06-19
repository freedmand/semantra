//! Reusable, Tauri-agnostic embedding pipeline.
//!
//! This module owns the "embed many chunks in batches, report progress" logic
//! so it can be shared by the current similarity demo and the upcoming
//! file-upload feature. It deliberately does NOT depend on Tauri: progress is
//! delivered through a plain `FnMut(EmbedProgress)` callback, so callers can
//! forward events to a Tauri `Channel`, log them, collect them in tests, etc.

use std::time::Instant;

use leaf_ir_candle_test::QUERY_PREFIX;

/// Default number of chunks handed to the model per inference call. Batching
/// does not change results (pooling/normalization are per-row, padding is
/// masked) — it is purely a memory/throughput guard. Mirrored on the frontend.
pub const BATCH_SIZE: usize = 32;

/// A single progress update streamed while embedding a list of chunks.
///
/// Serialized as a tagged union (`{ "kind": "batch", ... }`) so the frontend
/// can discriminate on `kind`. All timings are milliseconds (`f64`).
#[derive(Clone, serde::Serialize)]
// `rename_all` renames the variant tags (Started -> "started"); `rename_all_fields`
// renames the fields inside each variant (total_chunks -> "totalChunks"). Both are
// needed so the JSON matches the camelCase TS `EmbedProgress` union on the frontend.
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum EmbedProgress {
    /// Emitted once before any work begins, carrying the totals so the UI can
    /// size a progress bar up front.
    Started {
        total_chunks: usize,
        total_batches: usize,
        batch_size: usize,
    },
    /// Emitted after each batch finishes. `batch_ms` is that batch alone;
    /// `elapsed_ms` is cumulative wall-clock since work began.
    Batch {
        batch_index: usize,
        total_batches: usize,
        chunks_done: usize,
        total_chunks: usize,
        batch_ms: f64,
        elapsed_ms: f64,
    },
    /// Emitted once after the last batch, carrying summary timing stats.
    Finished {
        total_chunks: usize,
        total_batches: usize,
        total_ms: f64,
        chunks_per_sec: f64,
        avg_batch_ms: f64,
        min_batch_ms: f64,
        max_batch_ms: f64,
    },
}

/// Embed every chunk in `chunks` in batches of `batch_size`, **keeping** each
/// batch's vectors: `on_batch(start_index, chunk_texts, vectors)` fires once per
/// batch with the rows so a caller can insert them into a vector store as they
/// are produced (pipelining embedding with persistence). `on_progress` fires
/// before the run, after each batch, and once at the end, so the UI can size and
/// advance a progress bar.
///
/// `embed_batch` does the actual inference for one batch of (already
/// prefix-applied) inputs and returns one vector per input. Taking it as a
/// callback — rather than a borrowed `ModelCtx` — lets the caller control lock
/// scope: the background worker re-locks the shared model **per batch** so a
/// concurrent search can grab it between batches instead of waiting for the
/// whole file.
///
/// When `is_query` is true each chunk is prefixed with the asymmetric-retrieval
/// query prefix; documents are embedded raw. `on_batch` borrows the (original,
/// unprefixed) texts and owns the vectors. Synchronous and CPU-bound; run it on
/// a blocking thread when an event loop must stay responsive.
pub fn embed_chunks_with_vectors(
    mut embed_batch: impl FnMut(&[String]) -> Result<Vec<Vec<f32>>, String>,
    chunks: Vec<String>,
    is_query: bool,
    batch_size: usize,
    mut on_batch: impl FnMut(usize, &[String], Vec<Vec<f32>>) -> Result<(), String>,
    mut on_progress: impl FnMut(EmbedProgress),
) -> Result<(), String> {
    let total_chunks = chunks.len();
    let batch_size = batch_size.max(1);
    let total_batches = total_chunks.div_ceil(batch_size);

    on_progress(EmbedProgress::Started {
        total_chunks,
        total_batches,
        batch_size,
    });

    let run_start = Instant::now();
    let mut batch_times_ms: Vec<f64> = Vec::with_capacity(total_batches);

    for (batch_index, batch) in chunks.chunks(batch_size).enumerate() {
        // Apply the query prefix per batch so we never materialize a second
        // full-length Vec of inputs. Documents are embedded as-is.
        let inputs: Vec<String> = if is_query {
            batch
                .iter()
                .map(|t| format!("{QUERY_PREFIX}{t}"))
                .collect()
        } else {
            batch.to_vec()
        };

        let batch_start = Instant::now();
        let rows = embed_batch(&inputs)?;
        let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        batch_times_ms.push(batch_ms);

        // Hand the freshly-embedded rows to the caller. We pass the *original*
        // (unprefixed) chunk texts so they can be stored verbatim.
        on_batch(batch_index * batch_size, batch, rows)?;

        on_progress(EmbedProgress::Batch {
            batch_index,
            total_batches,
            chunks_done: (batch_index * batch_size + batch.len()).min(total_chunks),
            total_chunks,
            batch_ms,
            elapsed_ms: run_start.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let total_ms = run_start.elapsed().as_secs_f64() * 1000.0;
    let chunks_per_sec = if total_ms > 0.0 {
        total_chunks as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    let (avg_batch_ms, min_batch_ms, max_batch_ms) = if batch_times_ms.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let sum: f64 = batch_times_ms.iter().sum();
        (
            sum / batch_times_ms.len() as f64,
            batch_times_ms.iter().copied().fold(f64::INFINITY, f64::min),
            batch_times_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };

    on_progress(EmbedProgress::Finished {
        total_chunks,
        total_batches,
        total_ms,
        chunks_per_sec,
        avg_batch_ms,
        min_batch_ms,
        max_batch_ms,
    });

    Ok(())
}
