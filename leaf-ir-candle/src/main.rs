// Demo: run MongoDB/mdbr-leaf-ir embeddings with Candle, loading from ./models.
// Shared pipeline lives in lib.rs; this is just a small driver + the oracle check.

use anyhow::Result;
use leaf_ir_candle_test::{embed_sentences, setup_model, QUERY_PREFIX};

fn main() -> Result<()> {
    let ctx = setup_model("models")?;

    let queries = [
        "What is machine learning?",
        "How does neural network training work?",
    ];
    let documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Neural networks are trained through backpropagation, adjusting weights to minimize prediction errors.",
    ];

    let q_inputs: Vec<String> = queries
        .iter()
        .map(|q| format!("{QUERY_PREFIX}{q}"))
        .collect();
    let d_inputs: Vec<String> = documents.iter().map(|d| d.to_string()).collect();

    let q = embed_sentences(&ctx, &q_inputs)?;
    let d = embed_sentences(&ctx, &d_inputs)?;

    // cosine = dot (rows are unit-norm)
    for (i, query) in queries.iter().enumerate() {
        println!("Query: {query}");
        for (j, doc) in documents.iter().enumerate() {
            let sim: f32 = q.rows[i].iter().zip(&d.rows[j]).map(|(a, b)| a * b).sum();
            let preview: String = doc.chars().take(60).collect();
            println!("  Similarity: {sim:.4} | Document {j}: {preview}...");
        }
        println!();
    }

    let got: f32 = q.rows[0].iter().zip(&d.rows[0]).map(|(a, b)| a * b).sum();
    if (got - 0.6857).abs() < 0.01 {
        println!("PASS: scores[0][0] = {got:.4} matches expected ~0.6857");
    } else {
        println!("WARNING: scores[0][0] = {got:.4}, expected ~0.6857");
    }
    Ok(())
}
