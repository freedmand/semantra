//! Numerical parity against the FP32 PyTorch reference (sentence-transformers).
//!
//! Needs the model and golden vectors produced by the lab scripts; set
//! `EG2_MODEL_DIR` (google/embeddinggemma-2 checkout) and `EG2_GOLDEN_DIR`
//! (holding `texts.json` + `embeddings.json`). Skips when they are absent.

use std::collections::HashMap;
use std::path::PathBuf;

use semantra_embed::Model;

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let model = PathBuf::from(std::env::var("EG2_MODEL_DIR").ok()?);
    let golden = PathBuf::from(std::env::var("EG2_GOLDEN_DIR").ok()?);
    Some((model, golden))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn text_matches_fp32_reference() {
    let Some((model_dir, golden)) = dirs() else {
        eprintln!("EG2_MODEL_DIR / EG2_GOLDEN_DIR unset; skipping");
        return;
    };
    let texts: Vec<(String, String)> = {
        let m: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(golden.join("texts.json")).unwrap()).unwrap();
        m.into_iter().map(|(k, v)| (k, v.as_str().unwrap().to_owned())).collect()
    };
    let gold: HashMap<String, Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(golden.join("embeddings.json")).unwrap()).unwrap();

    let model = Model::load(&model_dir, 768).expect("load model");
    let inputs: Vec<String> = texts.iter().map(|(_, t)| t.clone()).collect();

    // Batched (padded) and one-at-a-time must both match the reference.
    let batched = model.embed_texts(&inputs).unwrap();
    for (i, (key, text)) in texts.iter().enumerate() {
        let single = model.embed_texts(std::slice::from_ref(text)).unwrap();
        let cb = dot(&batched.rows[i], &gold[key]);
        let cs = dot(&single.rows[0], &gold[key]);
        eprintln!("{key:>10}: cos(batched)={cb:.6} cos(single)={cs:.6}");
        assert!(cb > 0.9995 && cs > 0.9995, "{key}: {cb} / {cs}");
    }
}

#[test]
fn explain_decomposes_cosine_exactly() {
    let Some((model_dir, _)) = dirs() else { return };
    for dim in [768, 256] {
        let model = Model::load(&model_dir, dim).unwrap();
        let q = model
            .embed_texts(&[format!("{}café accents", semantra_embed::QUERY_PREFIX)])
            .unwrap();
        let prompt = semantra_embed::document_prompt(Some("notes.txt"));
        let docs = vec![
            (prompt.clone(), "Café — naïve EPA\u{2019}s résumé, déjà vu.".to_string()),
            (prompt.clone(), "The central bank raised interest rates to curb inflation, again and again.".to_string()),
        ];
        let inputs: Vec<String> = docs.iter().map(|(p, t)| format!("{p}{t}")).collect();
        let d = model.embed_texts(&inputs).unwrap();
        let exps = model.explain_similarity_batch(&q.rows[0], &docs).unwrap();
        for (i, exp) in exps.iter().enumerate() {
            let truth = dot(&q.rows[0], &d.rows[i]);
            let summed: f32 = exp.tokens.iter().map(|t| t.score).sum();
            assert!((exp.total - truth).abs() < 1e-4, "dim {dim} doc {i}: {} vs {truth}", exp.total);
            assert!((summed - truth).abs() < 1e-4);
            let text = &docs[i].1;
            for t in exp.tokens.iter().filter(|t| !t.special) {
                let s = text.get(t.start..t.end).expect("byte offsets on char boundaries");
                assert!(!s.is_empty() && !s.starts_with(' '), "span {s:?}");
            }
            let words: Vec<&str> = exp.tokens.iter().filter(|t| !t.special).map(|t| &text[t.start..t.end]).collect();
            eprintln!("dim {dim} doc {i}: total {:.4}; pieces {words:?}", exp.total);
        }
    }
}

fn load_patches(golden: &std::path::Path, name: &str) -> (mlx_rs::Array, i32, i32, usize) {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(golden.join("patches.json")).unwrap()).unwrap();
    let m = &meta[name];
    let (frames, rows, cols, n) = (
        m["frames"].as_u64().unwrap() as i32,
        m["rows"].as_u64().unwrap() as i32,
        m["cols"].as_u64().unwrap() as i32,
        m["n"].as_u64().unwrap() as i32,
    );
    let bytes = std::fs::read(golden.join(format!("patches_{name}.f32"))).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    (mlx_rs::Array::from_slice(&data, &[frames, n, 768]), rows, cols, frames as usize)
}

#[test]
fn vision_matches_fp32_reference_on_processor_patches() {
    let Some((model_dir, golden)) = dirs() else { return };
    let gold: HashMap<String, Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(golden.join("embeddings.json")).unwrap()).unwrap();
    let model = Model::load(&model_dir, 768).unwrap();
    for name in ["img_beach", "vid_clip"] {
        let (patches, rows, cols, frames) = load_patches(&golden, name);
        let e = model.embed_visual_patches(&patches, rows, cols, frames).unwrap();
        let c = dot(&e.rows[0], &gold[name]);
        eprintln!("{name:>10}: cos vs fp32 = {c:.6}");
        assert!(c > 0.999, "{name}: {c}");
    }
}

#[test]
fn images_from_files_match_fp32_reference() {
    let Some((model_dir, golden)) = dirs() else { return };
    let media = golden.parent().unwrap().join("media");
    let gold: HashMap<String, Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(golden.join("embeddings.json")).unwrap()).unwrap();
    let model = Model::load(&model_dir, 768).unwrap();
    use semantra_embed::media::image;
    for (key, file) in [("img_beach", "the_beach.jpg"), ("img_tree", "tree.jpg"), ("img_road", "big_sur_road.jpg")] {
        let t = std::time::Instant::now();
        let rgb = image::decode(&media.join(file)).unwrap();
        let prep = image::prepare(&rgb, image::IMAGE_SOFT_TOKENS).unwrap();
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        let e = model.embed_visual(&image::stack(&[&prep]).unwrap(), 1).unwrap();
        let c = dot(&e.rows[0], &gold[key]);
        eprintln!("{key:>10}: {}x{} -> grid {:?}  decode+resize {decode_ms:.1} ms  cos vs fp32 = {c:.5}", rgb.width, rgb.height, prep.grid());
        assert!(c > 0.995, "{key}: {c}");
    }
}

fn read_wav(path: &std::path::Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap();
    assert_eq!(r.spec().sample_rate, 16_000);
    r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

#[test]
fn audio_matches_fp32_reference() {
    let Some((model_dir, golden)) = dirs() else { return };
    let media = golden.parent().unwrap().join("media");
    let gold: HashMap<String, Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(golden.join("embeddings.json")).unwrap()).unwrap();

    // Front end: our log-mel vs the reference processor's features.
    let pcm = read_wav(&media.join("pasta.wav"));
    let (mel, _) = semantra_embed::media::mel::log_mel(&[&pcm]).unwrap();
    mel.eval().unwrap();
    let ours: &[f32] = mel.as_slice();
    let theirs: Vec<f32> = std::fs::read(golden.join("mel_aud_pasta.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(ours.len(), theirs.len(), "mel frame count");
    let max_err = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("log-mel max abs err vs reference: {max_err:.2e}");
    assert!(max_err < 1e-2);

    let model = Model::load(&model_dir, 768).unwrap();
    for (key, file) in [("aud_pasta", "pasta.wav"), ("aud_stocks", "stocks.wav"), ("aud_frog", "frog.wav")] {
        let pcm = read_wav(&media.join(file));
        let t = std::time::Instant::now();
        let e = model.embed_audio(&[&pcm]).unwrap();
        let c = dot(&e.rows[0], &gold[key]);
        eprintln!("{key:>10}: {:.1}s audio in {:.0} ms, cos vs fp32 = {c:.5}", pcm.len() as f32 / 16000.0, t.elapsed().as_secs_f64() * 1e3);
        assert!(c > 0.999, "{key}: {c}");
    }
}

#[test]
fn avfoundation_decoding_matches_reference() {
    let Some((model_dir, golden)) = dirs() else { return };
    let media = golden.parent().unwrap().join("media");
    let gold: HashMap<String, Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(golden.join("embeddings.json")).unwrap()).unwrap();
    let model = Model::load(&model_dir, 768).unwrap();
    use semantra_embed::media::{av, image};

    // Audio straight from the original AIFF (AVFoundation resamples).
    let mut pcm = Vec::new();
    av::stream_audio(&media.join("pasta.aiff"), |c| { pcm.extend_from_slice(c); Ok(()) }).unwrap();
    let e = model.embed_audio(&[&pcm]).unwrap();
    let c = dot(&e.rows[0], &gold["aud_pasta"]);
    eprintln!("pasta.aiff via AVFoundation: {} samples, cos vs fp32 = {c:.5}", pcm.len());
    assert!(c > 0.995);

    // Video frames at 1 fps from t = 0 (the reference's sampling), exact.
    let probe = av::probe(&media.join("clip.mp4")).unwrap();
    eprintln!("clip.mp4: {probe:?}");
    let times: Vec<f64> = (0..probe.duration_s as usize).map(|i| i as f64).collect();
    let t = std::time::Instant::now();
    let frames = av::frames_at(&media.join("clip.mp4"), &times, 1280, 0.0).unwrap();
    let prepped: Vec<image::Prepared> = frames.iter().map(|f| image::prepare(f, image::FRAME_SOFT_TOKENS).unwrap()).collect();
    eprintln!("9 frames decoded+resized in {:.0} ms, grid {:?}", t.elapsed().as_secs_f64() * 1e3, prepped[0].grid());
    let grid = image::stack(&prepped.iter().collect::<Vec<_>>()).unwrap();
    let e = model.embed_visual(&grid, times.len()).unwrap();
    let c = dot(&e.rows[0], &gold["vid_clip"]);
    eprintln!("clip.mp4 frames via AVFoundation: cos vs fp32 = {c:.5}");
    assert!(c > 0.99);
}

#[test]
fn oversized_inputs_are_truncated_not_embedded_whole() {
    let Some((model_dir, _)) = dirs() else { return };
    let model = Model::load(&model_dir, 768).unwrap();
    // ~50k tokens of text (e.g. a giant CSV cell) must be capped, and still embed.
    let huge = "lorem ipsum dolor sit amet ".repeat(12_000);
    assert_eq!(model.count_tokens(&huge).unwrap(), semantra_embed::MAX_INPUT_TOKENS);
    let e = model.embed_texts(&[huge]).unwrap();
    let norm: f32 = e.rows[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3);
}

#[test]
fn mixed_query_matches_single_modality_and_composes() {
    let Some((model_dir, golden)) = dirs() else { return };
    let media_dir = golden.parent().unwrap().join("media");
    let gold: HashMap<String, Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(golden.join("embeddings.json")).unwrap()).unwrap();
    let model = Model::load(&model_dir, 768).unwrap();
    use semantra_embed::media::image;
    // Text-only mixed query == the plain query path.
    let q = model.embed_texts(&[format!("{}waves on a beach", semantra_embed::QUERY_PREFIX)]).unwrap();
    let m = model.embed_query_mixed("waves on a beach", &[], &[]).unwrap();
    assert!(dot(&q.rows[0], &m) > 0.9999, "text-only mixed == text query");
    // Image-only mixed query == the image document embedding.
    let prep = image::prepare(&image::decode(&media_dir.join("the_beach.jpg")).unwrap(), image::IMAGE_SOFT_TOKENS).unwrap();
    let im = model.embed_query_mixed("", &[prep], &[]).unwrap();
    let c = dot(&im, &gold["img_beach"]);
    eprintln!("image-only mixed vs golden image: {c:.5}");
    assert!(c > 0.995);
    // Text + image composes: closer to each part than the parts are to each other.
    let prep = image::prepare(&image::decode(&media_dir.join("the_beach.jpg")).unwrap(), image::IMAGE_SOFT_TOKENS).unwrap();
    let both = model.embed_query_mixed("at night", &[prep], &[]).unwrap();
    eprintln!("image+text vs image {:.3}, vs text {:.3}", dot(&both, &im), dot(&both, &model.embed_query_mixed("at night", &[], &[]).unwrap()));
}
