//! The app's indexing + search pipeline end to end, without the UI: extract ->
//! sentence-snapped chunks -> EmbedService -> LanceDB (text + PDF page renders,
//! images, audio, video windows) -> exact search -> explain. Run on the MLX
//! build to write a reference, then on the ONNX build to compare rankings.
//!
//!   SEMANTRA_MODEL_DIR=models/embeddinggemma-2 \
//!     cargo run --release --example headless_pipeline -- ref.json <files…>
//!   SEMANTRA_MODEL_DIR=models/embeddinggemma-2-onnx ORT_DYLIB_PATH=… \
//!     cargo run --release --features onnx --example headless_pipeline -- ref.json <files…>
use std::path::{Path, PathBuf};
use std::time::Instant;

use semantra_embed::{document_prompt, EmbedService, Lane, QUERY_PREFIX};
use semantra_lib::store::{ChunkRow, Modality, SearchMode, VectorStore};
use semantra_lib::{chunk_segments, extract, media, pdf, pipeline_version, text_pipeline_version};
use serde_json::{json, Value};

const QUERIES: &[&str] = &[
    "a model behaving unexpectedly during an incident",
    "timeline of events",
    "economic policy and trade",
    "a photo of the ocean",
    "someone talking about cooking pasta",
    "a frog croaking",
    "a person speaking on stage",
    "a chart or table of numbers",
];

fn key(h: &semantra_lib::store::Hit, names: &[(String, String)]) -> String {
    let name = names.iter().find(|(s, _)| *s == h.sha512).map_or("?", |(_, n)| n.as_str());
    format!("{name}|{}|p{:?}|c{}|t{:?}", h.modality.as_str(), h.page, h.char_start, h.time_start_ms)
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ref_path = PathBuf::from(&a[1]);
    let files: Vec<PathBuf> = a[2..].iter().map(PathBuf::from).collect();
    let model_dir = PathBuf::from(std::env::var("SEMANTRA_MODEL_DIR").map_err(|_| "set SEMANTRA_MODEL_DIR")?);
    if let Some(lib) = std::env::var_os("ORT_DYLIB_PATH") {
        semantra_embed::set_onnxruntime_path(Path::new(&lib)).map_err(|e| e.to_string())?;
    }
    let backend = semantra_embed::BACKEND;
    let t0 = Instant::now();
    let embedder = EmbedService::spawn(model_dir, 768).map_err(|e| e.to_string())?;
    let pdfium = std::sync::Arc::new(pdf::load_library(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("libpdfium"))?);
    let tmp = std::env::temp_dir().join(format!("semantra-headless-{backend}-{}", std::process::id()));
    let conn = lancedb::connect(tmp.to_str().unwrap()).execute().await.map_err(|e| e.to_string())?;
    let mut store = VectorStore::open(conn, 768).await?;
    println!("[{backend}] model + store ready in {:.1}s", t0.elapsed().as_secs_f64());

    let mut names = Vec::new();
    let mut per_file = Vec::new();
    for (i, path) in files.iter().enumerate() {
        let sha = format!("file{i}");
        let name = path.file_name().unwrap().to_string_lossy().chars().take(24).collect::<String>();
        names.push((sha.clone(), name.clone()));
        let t = Instant::now();
        let p = path.to_str().unwrap().to_string();
        let extracted = extract::extract(&pdfium, &p)?;
        let filetype = extracted.filetype;
        let chunks = chunk_segments(&embedder, filetype, &extracted.segments);
        let prompt = document_prompt(None);
        let mut rows = Vec::new();
        for batch in chunks.chunks(32) {
            let inputs: Vec<String> = batch.iter().map(|c| format!("{prompt}{}", c.text)).collect();
            let vecs = embedder.run(Lane::Background, move |m| Ok(m.embed_texts(&inputs)?.rows)).await.map_err(|e| e.to_string())?;
            for (c, vector) in batch.iter().zip(vecs) {
                rows.push(ChunkRow {
                    sha512: sha.clone(),
                    modality: Modality::Text,
                    text: c.text.clone(),
                    char_start: c.char_start as i64,
                    char_end: c.char_end as i64,
                    page: c.page.map(|p| p as i64),
                    page_char_start: c.page_char_start as i64,
                    time_start_ms: None,
                    time_end_ms: None,
                    pipeline_version: text_pipeline_version(),
                    vector,
                });
            }
        }
        let n_text = rows.len();
        let plan = media::plan(filetype, &p, &extracted)?;
        let (pdfium2, embedder2, sha2, p2) = (pdfium.clone(), embedder.clone(), sha.clone(), p.clone());
        let media_rows = tokio::task::spawn_blocking(move || -> Result<Vec<ChunkRow>, String> {
            let mut out = Vec::new();
            media::run(plan, &p2, &pdfium2, &embedder2, |rs, _| {
                out.extend(rs.into_iter().map(|r| ChunkRow {
                    sha512: sha2.clone(),
                    modality: r.modality,
                    text: String::new(),
                    char_start: 0,
                    char_end: 0,
                    page: r.page,
                    page_char_start: 0,
                    time_start_ms: r.time_ms.map(|t| t.0),
                    time_end_ms: r.time_ms.map(|t| t.1),
                    pipeline_version: pipeline_version(),
                    vector: r.vector,
                }));
                Ok(())
            })?;
            Ok(out)
        })
        .await
        .map_err(|e| e.to_string())??;
        let n_media = media_rows.len();
        rows.extend(media_rows);
        store.insert(&rows).await?;
        let secs = t.elapsed().as_secs_f64();
        println!("  {name:24} {:?}: {n_text} text + {n_media} media rows in {secs:.1}s", filetype);
        per_file.push(json!({"name": name, "text": n_text, "media": n_media, "secs": secs}));
    }

    let shas: Vec<String> = names.iter().map(|(s, _)| s.clone()).collect();
    let mut results = Vec::new();
    for q in QUERIES {
        let prompted = format!("{QUERY_PREFIX}{q}");
        let qv = embedder.run(Lane::Priority, move |m| Ok(m.embed_texts(&[prompted])?.rows.remove(0))).await.map_err(|e| e.to_string())?;
        let hits = store.search(&qv, 10, SearchMode::Exact, &shas, &[]).await?;
        let docs: Vec<(String, String)> =
            hits.iter().filter(|h| h.modality == Modality::Text).take(3).map(|h| (document_prompt(None), h.text.clone())).collect();
        let qv2 = qv.clone();
        let ex = embedder.run(Lane::Priority, move |m| m.explain_similarity_batch(&qv2, &docs)).await.map_err(|e| e.to_string())?;
        results.push(json!({
            "query": q,
            "hits": hits.iter().map(|h| json!({"key": key(h, &names), "score": h.score})).collect::<Vec<_>>(),
            "explain": ex.iter().map(|e| e.tokens.iter().map(|t| t.score).collect::<Vec<_>>()).collect::<Vec<_>>(),
        }));
    }
    let out = json!({"backend": backend, "files": per_file, "queries": results});
    let _ = std::fs::remove_dir_all(&tmp);

    if backend == "mlx" {
        std::fs::write(&ref_path, serde_json::to_string_pretty(&out).unwrap()).map_err(|e| e.to_string())?;
        println!("wrote reference {}", ref_path.display());
        return Ok(());
    }
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(&ref_path).map_err(|e| e.to_string())?).unwrap();
    for (r, o) in reference["queries"].as_array().unwrap().iter().zip(out["queries"].as_array().unwrap()) {
        let keys = |v: &Value| v["hits"].as_array().unwrap().iter().map(|h| h["key"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        let (rk, ok) = (keys(r), keys(o));
        let overlap = ok.iter().filter(|k| rk.contains(k)).count();
        let top_same = rk.first() == ok.first();
        let score_diff = r["hits"][0]["score"].as_f64().unwrap_or(0.0) - o["hits"][0]["score"].as_f64().unwrap_or(0.0);
        println!(
            "  {:52} top-10 overlap {overlap}/10, top-1 same: {top_same}, top-1 score Δ {score_diff:+.4}",
            r["query"].as_str().unwrap()
        );
    }
    Ok(())
}
