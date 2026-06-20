// Benchmark: embed a synthetic 207-chunk corpus (~128 words/chunk) in batches of
// 32, mirroring the Tauri app's file-embed workload, and report timing. Compare
// devices with:  LEAF_IR_DEVICE=metal cargo run --release --example bench
//            vs:  LEAF_IR_DEVICE=cpu   cargo run --release --example bench
// A warm-up batch excludes one-time init (Metal kernel compilation) from the timing.
use std::time::Instant;

use leaf_ir_candle_test::{embed_sentences, setup_model};

fn main() -> anyhow::Result<()> {
    const CHUNKS: usize = 207;
    const WORDS: usize = 128;
    const BATCH: usize = 32;

    // Build varied 128-word chunks from a small vocabulary so tokenization is
    // realistic (not a single repeated token).
    let vocab = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "machine",
        "learning", "neural", "network", "embedding", "vector", "similarity",
        "language", "model", "transformer", "attention", "gradient",
    ];
    let chunks: Vec<String> = (0..CHUNKS)
        .map(|c| {
            (0..WORDS)
                .map(|w| vocab[(c * 7 + w * 3) % vocab.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();

    let ctx = setup_model("models")?;

    // Warm-up batch (kernel compilation / lazy init shouldn't count).
    let _ = embed_sentences(&ctx, &chunks[..BATCH.min(CHUNKS)].to_vec())?;

    let start = Instant::now();
    for batch in chunks.chunks(BATCH) {
        let _ = embed_sentences(&ctx, &batch.to_vec())?;
    }
    let total = start.elapsed().as_secs_f64();
    println!(
        "embedded {CHUNKS} chunks in {:.2} s  ({:.1} chunks/s)",
        total,
        CHUNKS as f64 / total
    );
    Ok(())
}
