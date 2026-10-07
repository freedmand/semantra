//! End-to-end media indexing: real `media::plan`/`media::run` over image,
//! audio and video fixtures (plus text chunks) into a temp LanceDB, then
//! cross-modal search. Needs the bundled model and the lab media fixtures
//! (`EG2_MEDIA_DIR`); skips otherwise.
//!
//!   EG2_MEDIA_DIR=~/scraps/eg2-lab/media cargo test --test multimodal_index -- --nocapture

use std::path::PathBuf;

use semantra_embed::{document_prompt, EmbedService, Lane, QUERY_PREFIX};
use semantra_lib::extract::{self, FileType};
use semantra_lib::store::{ChunkRow, Modality, SearchMode, VectorStore};
use semantra_lib::{media, pdf};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_modal_search_over_indexed_media() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/embeddinggemma-2");
    let Ok(media_dir) = std::env::var("EG2_MEDIA_DIR").map(PathBuf::from) else {
        eprintln!("EG2_MEDIA_DIR unset; skipping");
        return;
    };
    if !model_dir.join("model.safetensors").exists() {
        return;
    }
    let embedder = EmbedService::spawn(model_dir, 768).unwrap();
    let pdfium = pdf::load_library(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("libpdfium")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let conn = lancedb::connect(tmp.path().to_str().unwrap()).execute().await.unwrap();
    let mut store = VectorStore::open(conn, 768).await.unwrap();

    let mut rows: Vec<ChunkRow> = Vec::new();
    let row = |sha: &str, modality, text: &str, time: Option<(i64, i64)>, vector| ChunkRow {
        sha512: sha.into(),
        modality,
        text: text.into(),
        char_start: 0,
        char_end: 0,
        page: None,
        page_char_start: 0,
        time_start_ms: time.map(|t| t.0),
        time_end_ms: time.map(|t| t.1),
        pipeline_version: "test".into(),
        vector,
    };
    for file in ["the_beach.jpg", "tree.jpg", "pasta.wav", "clip.mp4"] {
        let path = media_dir.join(file);
        let p = path.to_str().unwrap().to_string();
        let ft = extract::media_type(&p).unwrap();
        let ex = extract::extract(&pdfium, &p).unwrap();
        let plan = media::plan(ft, &p, &ex).unwrap();
        let units = plan.units();
        let t = std::time::Instant::now();
        let mut got = Vec::new();
        let emb = embedder.clone();
        let pdfium_ref = &pdfium;
        media::run(plan, &p, pdfium_ref, &emb, |batch, _| {
            got.extend(batch);
            Ok(())
        })
        .unwrap();
        eprintln!("{file}: {:?} planned {units} units, produced {} rows in {:.0} ms", ft, got.len(), t.elapsed().as_secs_f64() * 1e3);
        assert!(!got.is_empty());
        for r in got {
            rows.push(row(file, r.modality, "", r.time_ms, r.vector));
        }
        let _ = FileType::Text;
    }
    let texts = [
        "Quarterly earnings beat expectations as revenue grew twelve percent.",
        "The committee will vote on the zoning amendment next Tuesday.",
    ];
    let prompt = document_prompt(None);
    let inputs: Vec<String> = texts.iter().map(|t| format!("{prompt}{t}")).collect();
    let vecs = embedder.run(Lane::Priority, move |m| m.embed_texts(&inputs)).await.unwrap();
    for (t, v) in texts.iter().zip(vecs.rows) {
        rows.push(row("notes.txt", Modality::Text, t, None, v));
    }
    store.insert(&rows).await.unwrap();
    let all: Vec<String> = ["the_beach.jpg", "tree.jpg", "pasta.wav", "clip.mp4", "notes.txt"].map(String::from).to_vec();

    for (q, want) in [
        ("waves washing onto a sandy beach", ["the_beach.jpg", "clip.mp4"].as_slice()),
        ("how do I make fresh pasta at home", ["pasta.wav", "clip.mp4"].as_slice()),
        ("a big tree", ["tree.jpg", "clip.mp4"].as_slice()),
        ("company profits this quarter", ["notes.txt"].as_slice()),
    ] {
        let qs = vec![format!("{QUERY_PREFIX}{q}")];
        let qv = embedder.run(Lane::Priority, move |m| m.embed_texts(&qs)).await.unwrap().rows.remove(0);
        let hits = store.search(&qv, 4, SearchMode::Exact, &all, &[]).await.unwrap();
        let top: Vec<String> = hits.iter().map(|h| format!("{}[{:?}{}] {:.3}", h.sha512, h.modality, h.time_start_ms.map(|t| format!(" @{}s", t / 1000)).unwrap_or_default(), h.score)).collect();
        eprintln!("{q:>40} -> {top:?}");
        assert!(want.contains(&hits[0].sha512.as_str()), "{q}: top hit {}", hits[0].sha512);
    }
}
