// Sanity check for the bundled EmbeddingGemma 2 files, loaded the same way the
// app resolves them in dev (relative to CARGO_MANIFEST_DIR): 768-d unit vectors,
// query/document prompts applied, and an on-topic passage outranking an
// off-topic one. Numerical parity with the FP32 PyTorch reference lives in
// semantra-embed's golden tests.
//
// Skips when the (gitignored, ~1.5 GB) weights haven't been fetched.
// Run with: cargo test --release --test embedding

use std::path::PathBuf;

use semantra_embed::{document_prompt, Model, QUERY_PREFIX};

#[test]
fn bundled_model_is_768d_and_ranks_sanely() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/embeddinggemma-2");
    if !model_dir.join("model.safetensors").exists() {
        eprintln!("models/embeddinggemma-2 not fetched; skipping (run ./fetch-model.sh)");
        return;
    }
    let model = Model::load(&model_dir, 768).expect("load bundled model files");
    let doc = document_prompt(None);

    let q = model
        .embed_texts(&[format!("{QUERY_PREFIX}What is machine learning?")])
        .expect("embed query");
    let d = model
        .embed_texts(&[
            format!("{doc}Machine learning is a subset of artificial intelligence that learns from data."),
            format!("{doc}The central bank raised interest rates to curb rising inflation."),
        ])
        .expect("embed docs");

    assert_eq!(q.dim, 768);
    let norm: f32 = q.rows[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "unit norm, got {norm}");

    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let on = cos(&q.rows[0], &d.rows[0]);
    let off = cos(&q.rows[0], &d.rows[1]);
    assert!(on > off + 0.05, "on-topic {on:.4} should clearly outrank off-topic {off:.4}");
}
