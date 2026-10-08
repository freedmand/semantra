//! Cross-backend parity through the public `Model` API: the same inputs on the
//! MLX build (writes a reference file) and the ONNX build (`--features onnx`;
//! compares against it). Covers every method the app calls.
//!
//!   cargo run --release --example backend_parity -- <mlx_model_dir> ref.json <image> <image> <audio> <video>
//!   cargo run --release --features onnx --example backend_parity -- <onnx_model_dir> ref.json <same media…>
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use semantra_embed::media::{av, image, mel};
use semantra_embed::{document_prompt, Model, QUERY_PREFIX};
use serde_json::{json, Value};

const BACKEND: &str = if cfg!(backend_mlx) { "mlx" } else { "onnx" };

fn docs() -> Vec<String> {
    [
        "The committee approved a new tariff schedule on imported steel, citing national security and the need to protect domestic jobs.",
        "Waves crashed against the sandy beach while children built castles near the tide line.",
        "To cook the noodles, boil them in salted water for eight minutes, then rinse under cold water.",
        "The stock market fell sharply after the central bank signaled further interest rate increases.",
        "A frog sat on the lily pad, croaking loudly as the sun set over the pond.",
        "Researchers trained a neural network to classify satellite images of farmland by crop type.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn audio_clip(path: &Path, seconds: f64) -> Result<Vec<f32>> {
    let max = (seconds * mel::SAMPLE_RATE as f64) as usize;
    let mut clip = Vec::new();
    av::stream_audio(path, |chunk| {
        clip.extend_from_slice(&chunk[..chunk.len().min(max.saturating_sub(clip.len()))]);
        Ok(())
    })?;
    Ok(clip)
}

fn timed<T>(label: &str, timings: &mut Vec<Value>, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let t = Instant::now();
    let out = f()?;
    timings.push(json!({"what": label, "ms": t.elapsed().as_secs_f64() * 1e3}));
    Ok(out)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (model_dir, ref_path) = (&a[1], Path::new(&a[2]));
    let media: Vec<&Path> = a[3..].iter().map(Path::new).collect();
    let (img_a, img_b, audio_path, video_path) = (media[0], media[1], media[2], media[3]);
    if let Some(lib) = std::env::var_os("ORT_DYLIB_PATH") {
        semantra_embed::set_onnxruntime_path(Path::new(&lib))?;
    }
    let t = Instant::now();
    let model = Model::load(model_dir, 768)?;
    model.warmup()?;
    println!("[{BACKEND}] loaded + warmed up in {:.1}s", t.elapsed().as_secs_f64());
    let mut out = serde_json::Map::new();
    let mut timings = Vec::new();

    // Text: documents (batched, ragged lengths -> padding) and a query.
    let prompt = document_prompt(None);
    let prompted: Vec<String> = docs().iter().map(|d| format!("{prompt}{d}")).collect();
    let doc_vecs = timed("embed_texts x6", &mut timings, || model.embed_texts(&prompted))?.rows;
    let query = "economic policy and trade";
    let q = model.embed_texts(&[format!("{QUERY_PREFIX}{query}")])?.rows.remove(0);
    out.insert("texts".into(), json!(doc_vecs));
    out.insert("query".into(), json!(q));

    // Explain: per-token scores for each doc against the query.
    let pairs: Vec<(String, String)> = docs().into_iter().map(|d| (prompt.clone(), d)).collect();
    let ex = timed("explain x6", &mut timings, || model.explain_similarity_batch(&q, &pairs))?;
    out.insert(
        "explain".into(),
        json!(ex.iter().map(|e| e.tokens.iter().map(|t| t.score).collect::<Vec<_>>()).collect::<Vec<_>>()),
    );
    out.insert("explain_totals".into(), json!(ex.iter().map(|e| e.total).collect::<Vec<_>>()));

    // Images: one at a time, and two same-grid images in one batch.
    let (pa, pb) = (image::prepare(&image::decode(img_a)?, 280)?, image::prepare(&image::decode(img_b)?, 280)?);
    let single = timed("embed_visual image", &mut timings, || model.embed_visual(&image::stack(&[&pa])?, 1))?.rows;
    out.insert("image".into(), json!(single[0]));
    if pa.grid() == pb.grid() {
        let both = timed("embed_visual x2 batch", &mut timings, || model.embed_visual(&image::stack(&[&pa, &pb])?, 1))?.rows;
        out.insert("image_batch".into(), json!(both));
    }

    // Audio: a 10 s clip, and a batch of two equal-length clips.
    let clip = audio_clip(audio_path, 10.0)?;
    let half = &clip[..clip.len() / 2];
    out.insert("audio".into(), json!(timed("embed_audio 10s", &mut timings, || model.embed_audio(&[&clip]))?.rows[0]));
    let shifted = &clip[clip.len() / 2..clip.len() / 2 + half.len()];
    out.insert("audio_batch".into(), json!(model.embed_audio(&[half, shifted])?.rows));

    // Video: two 8-frame blocks -> one 16-frame window (the app's reuse path).
    let times: Vec<f64> = (0..16).map(|i| i as f64 + 0.5).collect();
    let frames = av::frames_at(video_path, &times, 1344, 0.5)?;
    let prepared: Vec<image::Prepared> = frames.iter().map(|f| image::prepare(f, 140)).collect::<Result<_>>()?;
    let window = timed("video window 16f", &mut timings, || {
        let refs: Vec<&image::Prepared> = prepared.iter().collect();
        let a = model.visual_soft_tokens(&image::stack(&refs[..8])?)?;
        let b = model.visual_soft_tokens(&image::stack(&refs[8..])?)?;
        model.embed_frame_window(&[&a, &b])
    })?;
    out.insert("video".into(), json!(window));

    // Mixed query: text + image + audio in one sequence.
    let mixed = timed("embed_query_mixed", &mut timings, || model.embed_query_mixed("but at night", &[pa], &[half]))?;
    out.insert("mixed".into(), json!(mixed));

    // Throughput: 64 ~128-token chunks.
    let long: Vec<String> = (0..64).map(|i| format!("{prompt}{}", docs().join(" ").repeat(1 + i % 2))).collect();
    let t = Instant::now();
    model.embed_texts(&long)?;
    let tokens: usize = long.iter().map(|s| model.count_tokens(s).unwrap_or(0)).sum();
    println!("[{BACKEND}] text throughput: {:.0} tok/s", tokens as f64 / t.elapsed().as_secs_f64());
    for t in &timings {
        println!("  {:24} {:7.1} ms", t["what"].as_str().unwrap(), t["ms"].as_f64().unwrap());
    }

    if cfg!(backend_mlx) {
        std::fs::write(ref_path, serde_json::to_string(&Value::Object(out))?)?;
        println!("wrote reference {}", ref_path.display());
        return Ok(());
    }
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(ref_path)?)?;
    let vecs = |v: &Value| -> Vec<Vec<f32>> {
        match v.as_array().and_then(|a| a.first()).map(|x| x.is_array()) {
            Some(true) => serde_json::from_value(v.clone()).unwrap(),
            _ => vec![serde_json::from_value(v.clone()).unwrap()],
        }
    };
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut worst = 1f32;
    for key in ["texts", "query", "image", "image_batch", "audio", "audio_batch", "video", "mixed"] {
        let (Some(r), Some(o)) = (reference.get(key), out.get(key)) else { continue };
        let cos: Vec<f32> = vecs(r).iter().zip(vecs(o).iter()).map(|(x, y)| dot(x, y)).collect();
        let min = cos.iter().cloned().fold(1.0, f32::min);
        worst = worst.min(min);
        println!("  {key:12} cos vs MLX: min {min:.4} ({} vectors)", cos.len());
    }
    let r_ex: Vec<Vec<f32>> = serde_json::from_value(reference["explain"].clone())?;
    for (i, (r, o)) in r_ex.iter().zip(out["explain"].as_array().ok_or_else(|| anyhow!("no explain"))?).enumerate() {
        let o: Vec<f32> = serde_json::from_value(o.clone())?;
        let n = r.len() as f32;
        let (mr, mo) = (r.iter().sum::<f32>() / n, o.iter().sum::<f32>() / n);
        let cov: f32 = r.iter().zip(&o).map(|(x, y)| (x - mr) * (y - mo)).sum();
        let vr: f32 = r.iter().map(|x| (x - mr).powi(2)).sum();
        let vo: f32 = o.iter().map(|y| (y - mo).powi(2)).sum();
        println!("  explain[{i}]   tokens {}/{} corr {:.4}", o.len(), r.len(), cov / (vr * vo).sqrt());
    }
    println!("worst cosine: {worst:.4}");
    Ok(())
}
