// Oracle check for the bundled model files, loaded the same way the app resolves
// them in dev (relative to CARGO_MANIFEST_DIR). This guards against a corrupted
// copy and against the move breaking the pipeline: if the published model-card
// similarity drifts from 0.6857, something is wrong.
//
// The oracle value is specific to mdbr-leaf-ir, so this test names that model
// directly rather than following the app's `MODEL_NAME` switch.
//
// Run with: cargo test --release

use std::path::PathBuf;

use leaf_ir_candle_test::{embed_sentences, setup_model, QUERY_PREFIX};

#[test]
fn bundled_model_reproduces_oracle() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mdbr-leaf-ir");
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

// Smoke test for the multi-task model we also bundle (`mdbr-leaf-mt`, the current
// active model): it loads, emits 1024-d unit-norm vectors, and ranks an on-topic
// doc above an off-topic one for the same query. The model card doesn't pin a
// similarity value for this pair, so we assert shape + ordering rather than an
// oracle constant.
#[test]
fn bundled_mt_model_is_1024d_and_ranks_sanely() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/mdbr-leaf-mt");
    let ctx = setup_model(&model_dir).expect("load bundled mt model files");

    let query = format!("{QUERY_PREFIX}What is machine learning?");
    let on_topic =
        "Machine learning is a subset of artificial intelligence that learns from data."
            .to_string();
    let off_topic = "The central bank raised interest rates to curb rising inflation.".to_string();

    let q = embed_sentences(&ctx, &[query]).expect("embed query");
    let d = embed_sentences(&ctx, &[on_topic, off_topic]).expect("embed docs");

    assert_eq!(q.dim, 1024, "mt embedding must be 1024-d");

    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let on = cos(&q.rows[0], &d.rows[0]);
    let off = cos(&q.rows[0], &d.rows[1]);
    assert!(on > off, "on-topic {on:.4} should outrank off-topic {off:.4}");

    let norm: f32 = q.rows[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "query vector not unit-norm: {norm}");
}
