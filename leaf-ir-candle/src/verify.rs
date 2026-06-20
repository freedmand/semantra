// Self-contained correctness verification for the mdbr-leaf-ir Candle pipeline.
//
// No reference model needed. Three layers of checks, strongest first:
//
//   1. ORACLE (exact): model-card published similarities. If these match, the
//      whole pipeline (BERT -> mean-pool -> Dense 384->768 -> L2-norm + query
//      prefix) is provably correct.
//        "What is machine learning?"        vs ML doc           = 0.6857
//        "How does neural network training" vs neural-net doc   = 0.5723
//
//   2. INVARIANTS (must hold for ANY correct embedding model):
//        - embedding dimension is 768
//        - every embedding is unit L2 norm
//        - self-similarity == 1.0
//        - similarity matrix is symmetric
//        - all cosine sims in [-1, 1]
//
//   3. SEMANTIC RANKING (20 sentences): each probe query has an UNAMBIGUOUS
//      best match. We assert the right sentence ranks #1. These pairings are
//      chosen to be robust to small numerical differences — the correct answer
//      should clearly beat unrelated content, not win by a hair.
//
// Build: this expects the embedding fns from your main module. Easiest path is
// to paste your `embed`, `load_dense`, model/tokenizer loading into a small
// `lib`-style module, OR copy the loading preamble from main.rs into fn setup().
// Run with: cargo run --release --bin verify   (after wiring Cargo.toml bins)

use anyhow::Result;
// Reuse the exact same loading + embedding logic as main.rs via the lib crate,
// so the test exercises the real pipeline, not a reimplementation.
// Replace `leaf_ir_candle_test` if you named the package differently.
use leaf_ir_candle_test as pipeline;
use leaf_ir_candle_test::{Embeddings, QUERY_PREFIX};

const ORACLE_TOL: f32 = 0.01; // model card prints 4 decimals; 0.01 is safe
const NORM_TOL: f32 = 1e-3;

fn main() -> Result<()> {
    let mut failures = 0;
    let ctx = pipeline::setup_model("models")?; // loads tokenizer + BERT + dense once

    // ===================== LAYER 1: ORACLE =====================
    println!("== Layer 1: model-card oracle ==");
    {
        let queries = [
            format!("{QUERY_PREFIX}What is machine learning?"),
            format!("{QUERY_PREFIX}How does neural network training work?"),
        ];
        let docs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.".to_string(),
            "Neural networks are trained through backpropagation, adjusting weights to minimize prediction errors.".to_string(),
        ];
        let q = pipeline::embed_sentences(&ctx, &queries)?;
        let d = pipeline::embed_sentences(&ctx, &docs)?;
        let sims = cosine_matrix(&q, &d);

        failures += check_close("q0·d0 (ML/ML)", sims[0][0], 0.6857, ORACLE_TOL);
        failures += check_close("q1·d1 (NN/NN)", sims[1][1], 0.5723, ORACLE_TOL);
        // Cross pairs should be LOWER than the matching pairs (card shows this).
        failures += check_true("q0 matches d0 better than d1", sims[0][0] > sims[0][1]);
        failures += check_true("q1 matches d1 better than d0", sims[1][1] > sims[1][0]);
    }

    // ===================== LAYER 2: INVARIANTS =====================
    println!("\n== Layer 2: structural invariants ==");
    {
        let sentences: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        let emb = pipeline::embed_sentences(&ctx, &sentences)?;

        // dimension
        failures += check_true(
            &format!("embedding dim == 768 (got {})", emb.dim),
            emb.dim == 768,
        );

        // unit norm for every row
        let mut bad_norm = 0;
        for (i, v) in emb.rows.iter().enumerate() {
            let n = l2(v);
            if (n - 1.0).abs() > NORM_TOL {
                bad_norm += 1;
                if bad_norm <= 3 {
                    println!("    row {i} norm = {n:.5} (expected 1.0)");
                }
            }
        }
        failures += check_true("all embeddings unit-norm", bad_norm == 0);

        let sims = cosine_matrix(&emb, &emb);

        // self-similarity == 1
        let mut bad_self = 0;
        for i in 0..emb.rows.len() {
            if (sims[i][i] - 1.0).abs() > 1e-3 {
                bad_self += 1;
            }
        }
        failures += check_true("self-similarity == 1.0", bad_self == 0);

        // symmetry
        let mut asym = 0;
        for i in 0..sims.len() {
            for j in 0..sims.len() {
                if (sims[i][j] - sims[j][i]).abs() > 1e-4 {
                    asym += 1;
                }
            }
        }
        failures += check_true("similarity matrix symmetric", asym == 0);

        // range
        let mut oob = 0;
        for row in &sims {
            for &s in row {
                if s < -1.0001 || s > 1.0001 {
                    oob += 1;
                }
            }
        }
        failures += check_true("all sims in [-1, 1]", oob == 0);
    }

    // ===================== LAYER 3: SEMANTIC RANKING =====================
    // Each probe is a query; its expected best match is an index into CORPUS.
    // Pairings chosen so the correct answer is unambiguous.
    println!("\n== Layer 3: semantic ranking ==");
    {
        let docs: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        let doc_emb = pipeline::embed_sentences(&ctx, &docs)?;

        for probe in PROBES {
            let q = format!("{QUERY_PREFIX}{}", probe.query);
            let qe = pipeline::embed_sentences(&ctx, &[q])?;
            let sims = cosine_matrix(&qe, &doc_emb);
            let (best_idx, best_score) = argmax(&sims[0]);

            let ok = best_idx == probe.expect_idx;
            if !ok {
                failures += 1;
            }
            println!(
                "  [{}] \"{}\"\n      top: ({best_score:.3}) \"{}\"\n      want: \"{}\"",
                if ok { "PASS" } else { "FAIL" },
                probe.query,
                CORPUS[best_idx],
                CORPUS[probe.expect_idx],
            );
        }
    }

    println!("\n=========================================");
    if failures == 0 {
        println!("ALL CHECKS PASSED");
        Ok(())
    } else {
        println!("{failures} CHECK(S) FAILED");
        std::process::exit(1);
    }
}

// ---------- 20-sentence corpus across distinct semantic clusters -------------
// Clusters: cooking, weather, finance, space, sports, programming, health, music.
// Distinct enough that a correct model ranks within-cluster matches far above
// cross-cluster ones.
const CORPUS: [&str; 20] = [
    // cooking (0-2)
    "Knead the dough for ten minutes until it becomes smooth and elastic.",
    "Simmer the tomato sauce on low heat so the flavors deepen.",
    "Preheat the oven to 220 degrees before baking the bread.",
    // weather (3-5)
    "A cold front will bring heavy rain and strong winds tomorrow afternoon.",
    "The forecast predicts clear skies and sunshine throughout the weekend.",
    "Meteorologists issued a warning about the approaching tropical storm.",
    // finance (6-8)
    "The central bank raised interest rates to curb rising inflation.",
    "Quarterly earnings exceeded analyst expectations, lifting the stock price.",
    "Diversifying your portfolio reduces exposure to any single market risk.",
    // space (9-11)
    "The spacecraft entered orbit around Mars after a seven-month journey.",
    "Astronomers detected a distant galaxy using the new infrared telescope.",
    "The rocket's booster separated cleanly during the second stage of launch.",
    // sports (12-14)
    "The striker scored a last-minute goal to win the championship match.",
    "She broke the national record in the four-hundred-meter sprint.",
    "The team's defense held firm through overtime to secure the title.",
    // programming (15-17)
    "The function returns a null pointer when the input array is empty.",
    "Refactor the loop into a recursive call to simplify the algorithm.",
    "Memory leaks occur when allocated objects are never freed.",
    // health (18-19)
    "Regular aerobic exercise lowers blood pressure and strengthens the heart.",
    "A balanced diet rich in fiber supports healthy digestion.",
];

struct Probe {
    query: &'static str,
    expect_idx: usize,
}

// Each query is paraphrased so it does NOT lexically copy the target — this
// tests semantic matching, not keyword overlap. Expected match is unambiguous.
const PROBES: [Probe; 8] = [
    Probe {
        query: "How long should I work the bread dough?",
        expect_idx: 0,
    },
    Probe {
        query: "Will it rain heavily tomorrow?",
        expect_idx: 3,
    },
    Probe {
        query: "Why did the monetary authority increase rates?",
        expect_idx: 6,
    },
    Probe {
        query: "When did the probe reach the red planet?",
        expect_idx: 9,
    },
    Probe {
        query: "Who scored the goal that made all the difference?",
        expect_idx: 12,
    },
    Probe {
        query: "What causes a program to keep consuming memory?",
        expect_idx: 17,
    },
    Probe {
        query: "Does cardio help your cardiovascular system?",
        expect_idx: 18,
    },
    Probe {
        query: "How can I spread investment risk across assets?",
        expect_idx: 8,
    },
];

// --------------------------- math helpers ------------------------------------
fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// Embeddings are pre-normalized, so dot product == cosine similarity.
fn cosine_matrix(a: &Embeddings, b: &Embeddings) -> Vec<Vec<f32>> {
    a.rows
        .iter()
        .map(|ra| b.rows.iter().map(|rb| dot(ra, rb)).collect())
        .collect()
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

fn check_close(name: &str, got: f32, want: f32, tol: f32) -> i32 {
    let ok = (got - want).abs() <= tol;
    println!(
        "  [{}] {name}: got {got:.4}, want {want:.4} (±{tol})",
        if ok { "PASS" } else { "FAIL" }
    );
    if ok {
        0
    } else {
        1
    }
}

fn check_true(name: &str, cond: bool) -> i32 {
    println!("  [{}] {name}", if cond { "PASS" } else { "FAIL" });
    if cond {
        0
    } else {
        1
    }
}
