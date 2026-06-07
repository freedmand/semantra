// Oracle check for the model files bundled under src-tauri/models, loaded the
// same way the app resolves them in dev (relative to CARGO_MANIFEST_DIR). This
// guards against a corrupted copy and against the move breaking the pipeline:
// if the published model-card similarity drifts from 0.6857, something is wrong.
//
// Run with: cargo test --release

use std::path::PathBuf;

use leaf_ir_candle_test::{embed_sentences, setup_model, QUERY_PREFIX};

#[test]
fn bundled_model_reproduces_oracle() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let ctx = setup_model(&model_dir).expect("load bundled model files");

    let query = format!("{QUERY_PREFIX}What is machine learning?");
    let doc = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.".to_string();

    let q = embed_sentences(&ctx, &[query]).expect("embed query");
    let d = embed_sentences(&ctx, &[doc]).expect("embed doc");

    assert_eq!(q.dim, 768, "embedding must be 768-d");

    // rows are unit-norm, so dot product == cosine similarity
    let sim: f32 = q.rows[0].iter().zip(&d.rows[0]).map(|(a, b)| a * b).sum();
    assert!(
        (sim - 0.6857).abs() < 0.01,
        "oracle drifted: got {sim:.4}, want 0.6857"
    );
}
