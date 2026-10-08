//! Step-by-step smoke test of every `Model` method, printing as it goes (for
//! CI on each platform, and for bisecting crashes). Media steps run when files
//! are given.
//!
//!   cargo run --release --example smoke -- <model_dir> [image] [audio]
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use semantra_embed::media::{av, image, mel};
use semantra_embed::{document_prompt, Model, QUERY_PREFIX};

fn step<T>(name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    print!("{name:32} ");
    std::io::stdout().flush().ok();
    let t = Instant::now();
    let out = f()?;
    println!("ok ({:.0} ms)", t.elapsed().as_secs_f64() * 1e3);
    Ok(out)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if let Some(lib) = std::env::var_os("ORT_DYLIB_PATH") {
        semantra_embed::set_onnxruntime_path(Path::new(&lib))?;
    }
    println!("backend: {}", semantra_embed::BACKEND);
    let model = step("load", || Model::load(&a[1], 768))?;
    let p = document_prompt(None);
    let q = step("embed 1 text", || Ok(model.embed_texts(&[format!("{QUERY_PREFIX}hello world")])?.rows.remove(0)))?;
    let docs = vec![format!("{p}short"), format!("{p}a considerably longer document about trade policy and tariffs")];
    step("embed 2 texts (padded)", || model.embed_texts(&docs))?;
    let pairs = vec![(p.clone(), "short".to_string()), (p.clone(), "trade policy and tariffs".to_string())];
    let ex = step("explain 2 docs", || model.explain_similarity_batch(&q, &pairs))?;
    println!("  explain totals {:?}", ex.iter().map(|e| e.total).collect::<Vec<_>>());
    step("mixed query (text only)", || model.embed_query_mixed("hello world", &[], &[]))?;
    if let Some(img) = a.get(2) {
        let prepared = step("decode + prepare image", || image::prepare(&image::decode(Path::new(img))?, 280))?;
        step("embed image", || model.embed_visual(&image::stack(&[&prepared])?, 1))?;
        step("mixed query (text + image)", || model.embed_query_mixed("at night", &[prepared], &[]))?;
    }
    if let Some(aud) = a.get(3) {
        let mut clip = Vec::new();
        let max = 10 * mel::SAMPLE_RATE as usize;
        step("decode audio", || {
            av::stream_audio(Path::new(aud), |c| {
                clip.extend_from_slice(&c[..c.len().min(max.saturating_sub(clip.len()))]);
                Ok(())
            })
        })?;
        step("embed audio", || model.embed_audio(&[&clip]))?;
    }
    println!("smoke: all steps passed");
    Ok(())
}
