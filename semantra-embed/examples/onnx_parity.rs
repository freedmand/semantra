//! ONNX Runtime backend experiment: image + audio embeddings through the
//! onnx-community EmbeddingGemma 2 export, from the same decoded inputs as the
//! MLX model, compared for parity and timed.
//!
//!   ORT_DYLIB_PATH=…/libonnxruntime.dylib cargo run --release --example onnx_parity -- \
//!     <mlx_model_dir> <onnx_dir> <cpu|webgpu> <media files…>
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, Ix2};
use ort::session::Session;
use ort::value::Tensor;
use semantra_embed::media::{av, image, mel};
use semantra_embed::Model;

/// Image soft-token budget (the app's default for images).
const IMAGE_SOFT_TOKENS: u32 = 280;
/// Audio window the app embeds (15 s).
const AUDIO_SECONDS: usize = 15;
/// Video window the app embeds: 16 frames at 1 fps, 140 soft tokens each,
/// decoded at <= 1344 px.
const VIDEO_FRAMES: usize = 16;

fn session(path: &Path, provider: &str) -> Result<Session> {
    let mut b = Session::builder().map_err(|e| anyhow!("{e}"))?;
    if provider == "webgpu" {
        b = b
            .with_execution_providers([ort::ep::WebGPU::default().build().error_on_failure()])
            .map_err(|e| anyhow!("{e}"))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!("{e}"))
}

struct Onnx {
    text: Session,
    vision: Session,
    audio: Session,
    bos: i64,
    eos: i64,
}

impl Onnx {
    /// `<bos> (<open> placeholder×n <close>)×segments <eos>` through the text
    /// model with `features` (segments·n rows) filling the placeholders (named
    /// input `slot`). A video window is one segment per frame.
    fn embed_soft(&mut self, slot: &str, features: Array2<f32>, segments: usize, open: u32, placeholder: u32, close: u32) -> Result<Vec<f32>> {
        let n = features.shape()[0] / segments;
        let mut ids = vec![self.bos];
        for _ in 0..segments {
            ids.push(open as i64);
            ids.extend(std::iter::repeat(placeholder as i64).take(n));
            ids.push(close as i64);
        }
        ids.push(self.eos);
        let len = ids.len();
        let empty = || Tensor::from_array(Array2::<f32>::zeros((0, 512))).unwrap();
        let feats = Tensor::from_array(features)?;
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, len), ids)?)?,
            "attention_mask" => Tensor::from_array(Array2::<i64>::ones((1, len)))?,
        ];
        for name in ["image_features", "video_features", "audio_features"] {
            inputs.push((name.into(), if name == slot { feats.clone().into() } else { empty().into() }));
        }
        let out = self.text.run(inputs)?;
        Ok(out["sentence_embedding"].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?.row(0).to_vec())
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (mlx_dir, onnx_dir, provider) = (&a[1], Path::new(&a[2]), a[3].as_str());
    let model = Model::load(mlx_dir, 768)?;
    let c = model.config().clone();
    let special = model.tokenizer().encode("", true).map_err(|e| anyhow!("{e}"))?;
    let ids = special.get_ids();
    let mut onnx = Onnx {
        text: session(&onnx_dir.join("model_fp16.onnx"), provider)?,
        vision: session(&onnx_dir.join("vision_encoder_fp16.onnx"), provider)?,
        audio: session(&onnx_dir.join("audio_encoder_fp16.onnx"), provider)?,
        bos: ids[0] as i64,
        eos: *ids.last().unwrap() as i64,
    };
    println!("[{provider}] sessions ready (bos {}, eos {})", onnx.bos, onnx.eos);

    for f in &a[4..] {
        let path = Path::new(f);
        let name = path.file_name().unwrap().to_string_lossy();
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if ["jpg", "jpeg", "png", "heic"].contains(&ext.as_str()) {
            let img = image::decode(path)?;
            let prepared = image::prepare(&img, IMAGE_SOFT_TOKENS)?;
            let t = Instant::now();
            let mlx = model.embed_visual(&image::stack(&[&prepared])?, 1)?.rows.remove(0);
            let t_mlx = t.elapsed();

            let t = Instant::now();
            let (pixels, positions) = image::patchify(&prepared);
            let n = (prepared.rows * prepared.cols) as usize;
            let out = onnx.vision.run(ort::inputs![
                "pixel_values" => Tensor::from_array(Array3::from_shape_vec((1, n, 768), pixels)?)?,
                "pixel_position_ids" => Tensor::from_array(Array3::from_shape_vec((1, n, 2), positions)?)?,
            ])?;
            let feats = out["image_features"].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?.to_owned();
            drop(out);
            let ours = onnx.embed_soft("image_features", feats, 1, c.boi_token_id, c.image_token_id, c.eoi_token_id)?;
            println!(
                "  image {name:40} {}x{} patches: cos {:.4} | mlx {:5.0} ms, onnx {:5.0} ms",
                prepared.rows,
                prepared.cols,
                dot(&mlx, &ours),
                t_mlx.as_secs_f64() * 1e3,
                t.elapsed().as_secs_f64() * 1e3
            );
        } else if ["mp4", "mov"].contains(&ext.as_str()) {
            let times: Vec<f64> = (0..VIDEO_FRAMES).map(|i| i as f64 + 0.5).collect();
            let frames = av::frames_at(path, &times, 1344, 0.5)?;
            let prepared: Vec<image::Prepared> =
                frames.iter().map(|f| image::prepare(f, image::FRAME_SOFT_TOKENS)).collect::<Result<_>>()?;
            let t = Instant::now();
            let soft = model.visual_soft_tokens(&image::stack(&prepared.iter().collect::<Vec<_>>())?)?;
            let mlx = model.embed_frame_window(&[&soft])?;
            let t_mlx = t.elapsed();

            let t = Instant::now();
            let n = (prepared[0].rows * prepared[0].cols) as usize;
            let (mut pixels, mut positions) = (Vec::new(), Vec::new());
            for p in &prepared {
                let (px, pos) = image::patchify(p);
                pixels.extend(px);
                positions.extend(pos);
            }
            let f = prepared.len();
            let out = onnx.vision.run(ort::inputs![
                "pixel_values" => Tensor::from_array(Array3::from_shape_vec((f, n, 768), pixels)?)?,
                "pixel_position_ids" => Tensor::from_array(Array3::from_shape_vec((f, n, 2), positions)?)?,
            ])?;
            let feats = out["image_features"].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?.to_owned();
            drop(out);
            let ours = onnx.embed_soft("image_features", feats, f, c.boi_token_id, c.image_token_id, c.eoi_token_id)?;
            println!(
                "  video {name:40} {f} frames {}x{} patches: cos {:.4} | mlx {:5.0} ms, onnx {:5.0} ms",
                prepared[0].rows,
                prepared[0].cols,
                dot(&mlx, &ours),
                t_mlx.as_secs_f64() * 1e3,
                t.elapsed().as_secs_f64() * 1e3
            );
        } else if ["wav", "aiff", "mp3", "m4a"].contains(&ext.as_str()) {
            let mut clip = Vec::new();
            let max = AUDIO_SECONDS * mel::SAMPLE_RATE as usize;
            av::stream_audio(path, |chunk| {
                clip.extend_from_slice(&chunk[..chunk.len().min(max.saturating_sub(clip.len()))]);
                Ok(())
            })?;
            let t = Instant::now();
            let mlx = model.embed_audio(&[&clip])?.rows.remove(0);
            let t_mlx = t.elapsed();

            let t = Instant::now();
            let (features, frames, valid) = mel::log_mel_cpu(&clip)?;
            let mask: Vec<bool> = (0..frames).map(|f| f < valid).collect();
            let out = onnx.audio.run(ort::inputs![
                "input_features" => Tensor::from_array(Array3::from_shape_vec((1, frames, 128), features)?)?,
                "input_features_mask" => Tensor::from_array(Array1::from_vec(mask).into_shape_with_order((1, frames))?)?,
            ])?;
            let feats = out["audio_features"].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?.to_owned();
            drop(out);
            let n_soft = feats.shape()[0];
            let ours = onnx.embed_soft("audio_features", feats, 1, c.boa_token_id, c.audio_token_id, c.eoa_token_index)?;
            println!(
                "  audio {name:40} {:.1}s, {n_soft} soft: cos {:.4} | mlx {:5.0} ms, onnx {:5.0} ms",
                clip.len() as f64 / mel::SAMPLE_RATE as f64,
                dot(&mlx, &ours),
                t_mlx.as_secs_f64() * 1e3,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}
