//! The ONNX Runtime backend (Windows, Linux, Intel Macs): the onnx-community
//! EmbeddingGemma 2 export, three graphs —
//!
//! - `model_fp16.onnx`: the text encoder. Takes token ids plus the soft tokens
//!   of any images/video frames/audio (filled in at their placeholder ids) and
//!   returns `last_hidden_state`, the per-token 768-d vectors after the final
//!   projection — the same quantity the MLX backend pools, so pooling,
//!   Matryoshka truncation and the exact per-token attribution are shared.
//! - `vision_encoder_fp16.onnx`: 16×16 patches + (x, y) positions -> soft
//!   tokens (one per 3×3 patch block).
//! - `audio_encoder_fp16.onnx`: log-mel frames + mask -> soft tokens (one per
//!   40 ms).
//!
//! fp16 matches the BF16 MLX model to cos ≥ 0.9997 on every modality (the
//! export keeps overflow-prone ops in F32). Preprocessing — resize, patchify,
//! log-mel — runs on the CPU (`media::image::patchify`,
//! `media::mel::log_mel_cpu`), identical to what the MLX towers consume.
//!
//! GPU: WebGPU (D3D12 / Vulkan / Metal via Dawn) or DirectML when the loaded
//! ONNX Runtime library includes them, else CPU. `SEMANTRA_ORT_EP=cpu` forces
//! the CPU.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Error, Result};
use ndarray::{Array1, Array2, Array3, Axis, Ix2, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::media::{self, image::Prepared};
use crate::{Embeddings, Explanation, MAX_INPUT_TOKENS, QUERY_PREFIX};

/// Soft-token feature width shared by the vision and audio encoders.
const SOFT_DIM: usize = 512;
const BOS: i64 = 2;
const EOS: i64 = 1;

/// No-op: Metal kernels are an MLX concern. Kept so callers can be
/// backend-agnostic.
pub fn set_metallib_path(_path: &Path) -> Result<()> {
    Ok(())
}

static ORT_DYLIB: OnceLock<String> = OnceLock::new();

/// Serializes every session run in the process. The WebGPU device (Dawn) is
/// shared across sessions and not safe to drive from several threads at once
/// — concurrent runs abort inside Metal/D3D. The app already funnels model
/// work through one inference thread; this keeps any other caller (several
/// `Model`s, tests) safe too, at no cost: the GPU runs one graph at a time.
static RUN: Mutex<()> = Mutex::new(());

fn run_lock() -> std::sync::MutexGuard<'static, ()> {
    RUN.lock().unwrap_or_else(|p| p.into_inner())
}

/// Point the backend at the ONNX Runtime shared library to load (the app
/// bundles one per platform). Must be called before [`Model::load`]; without
/// it, `ORT_DYLIB_PATH` or the platform's default library name is used.
pub fn set_onnxruntime_path(path: &Path) -> Result<()> {
    let p = path.to_str().ok_or_else(|| anyhow!("non-UTF-8 onnxruntime path"))?;
    // Windows resolves a DLL's own dependencies (onnxruntime_providers_shared,
    // the DirectX shader compiler for WebGPU) from the process's search path,
    // not the DLL's folder — so put that folder first. Called at startup,
    // before any other thread reads the environment.
    if cfg!(target_os = "windows") {
        if let Some(dir) = path.parent() {
            let mut paths = vec![dir.to_path_buf()];
            paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
            std::env::set_var("PATH", std::env::join_paths(paths)?);
        }
    }
    ORT_DYLIB.set(p.to_string()).map_err(|_| anyhow!("onnxruntime path already set"))
}

fn init_runtime() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        // `commit` reports whether this call created the environment; either
        // way one now exists.
        match ORT_DYLIB.get() {
            Some(p) => ort::init_from(p).map_err(|e| e.to_string())?.with_name("semantra").commit(),
            None => ort::init().with_name("semantra").commit(),
        };
        Ok(())
    })
    .clone()
    .map_err(|e| anyhow!("load ONNX Runtime: {e}"))
}

/// The execution providers to try, best first; unavailable ones are skipped.
fn providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
    if std::env::var("SEMANTRA_ORT_EP").is_ok_and(|v| v.eq_ignore_ascii_case("cpu")) {
        return Vec::new();
    }
    let mut eps = Vec::new();
    if cfg!(target_os = "windows") {
        eps.push(ort::ep::DirectML::default().build());
    }
    eps.push(ort::ep::WebGPU::default().build());
    eps
}

fn session(path: &Path) -> Result<Session> {
    Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_execution_providers(providers())
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow!("load {}: {e}", path.display()))
}

/// Patches for the vision encoder: `frames` images of `rows × cols` patches,
/// (frames · rows · cols × 768) values in [0, 1] plus (x, y) per patch.
pub struct PatchGrid {
    pub pixels: Vec<f32>,
    pub positions: Vec<i64>,
    pub frames: usize,
    pub rows: i32,
    pub cols: i32,
}

/// Vision-encoder output for a run of video frames — (frames · per_frame,
/// 512) soft tokens — so overlapping video windows reuse each frame's tokens.
pub struct SoftFrames {
    features: Array2<f32>,
    frames: usize,
}

impl SoftFrames {
    pub fn frames(&self) -> usize {
        self.frames
    }
}

pub struct Model {
    text: Mutex<Session>,
    vision: Option<Mutex<Session>>,
    audio: Option<Mutex<Session>>,
    tokenizer: Tokenizer,
    config: ModelConfig,
    dim: usize,
}

/// Soft tokens for one text-model call: placeholder ids are filled from these,
/// in order.
#[derive(Default)]
struct Soft {
    image: Option<Array2<f32>>,
    audio: Option<Array2<f32>>,
}

impl Model {
    /// Load from `dir`: `config.json`, `tokenizer.json`, and the ONNX graphs
    /// under `onnx/` (`model_fp16.onnx`, plus the vision/audio encoders when
    /// present). `dim` selects the Matryoshka output size: 768, 512, 256 or 128.
    pub fn load(dir: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let dir = dir.as_ref();
        if ![768, 512, 256, 128].contains(&dim) {
            bail!("unsupported embedding dim {dim}; expected 768, 512, 256 or 128");
        }
        init_runtime()?;
        let config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(Error::msg)?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.text_config.pad_token_id,
            pad_token: "<pad>".into(),
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_INPUT_TOKENS,
                ..Default::default()
            }))
            .map_err(Error::msg)?;
        let graphs = dir.join("onnx");
        let optional = |name: &str| -> Result<Option<Mutex<Session>>> {
            let p = graphs.join(name);
            p.exists().then(|| session(&p).map(Mutex::new)).transpose()
        };
        Ok(Model {
            text: Mutex::new(session(&graphs.join("model_fp16.onnx"))?),
            vision: optional("vision_encoder_fp16.onnx")?,
            audio: optional("audio_encoder_fp16.onnx")?,
            tokenizer,
            config,
            dim,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.dim
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Token count of `text` as the model sees it (including `<bos>`/`<eos>`).
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokenizer.encode(text, true).map_err(Error::msg)?.len())
    }

    /// Embed already-prompted strings (callers prepend [`QUERY_PREFIX`] or
    /// [`crate::document_prompt`]) in one batched pass.
    pub fn embed_texts(&self, inputs: &[String]) -> Result<Embeddings> {
        if inputs.is_empty() {
            return Ok(Embeddings { rows: vec![], dim: self.dim });
        }
        let (ids, mask) = self.tokenize(inputs)?;
        let per_token = self.run_text(ids, mask.clone(), Soft::default())?;
        self.pooled(&per_token, &mask)
    }

    /// Embed images or video windows (no text prompt). `grid` holds
    /// `B × frames_per_item` same-shaped frames, item-major; each item becomes
    /// `<bos> ( <|image> soft×n <image|> ) × frames_per_item <eos>`.
    pub fn embed_visual(&self, grid: &PatchGrid, frames_per_item: usize) -> Result<Embeddings> {
        let soft = self.vision_soft(grid)?;
        self.embed_frames(soft, grid.frames, frames_per_item.max(1))
    }

    /// Embed equal-length 16 kHz mono clips (≤ 30 s each; no text prompt).
    /// Each becomes `<bos> <|audio> soft×n <audio|> <eos>`.
    pub fn embed_audio(&self, clips: &[&[f32]]) -> Result<Embeddings> {
        if clips.is_empty() {
            return Ok(Embeddings { rows: vec![], dim: self.dim });
        }
        let soft = self.audio_soft(clips)?;
        let n = soft.shape()[0] / clips.len();
        let c = &self.config;
        let row = delimited(&[(c.boa_token_id, c.audio_token_id, c.eoa_token_index, n)]);
        let (ids, mask) = repeat_rows(&row, clips.len());
        let per_token = self.run_text(ids, mask.clone(), Soft { image: None, audio: Some(soft) })?;
        self.pooled(&per_token, &mask)
    }

    /// Run only the vision encoder over `grid` (one video frame per image) and
    /// keep the soft tokens for [`embed_frame_window`](Self::embed_frame_window).
    pub fn visual_soft_tokens(&self, grid: &PatchGrid) -> Result<SoftFrames> {
        Ok(SoftFrames { features: self.vision_soft(grid)?, frames: grid.frames })
    }

    /// Embed one video window made of consecutive `blocks` of frames (all from
    /// [`visual_soft_tokens`](Self::visual_soft_tokens), same frame size).
    pub fn embed_frame_window(&self, blocks: &[&SoftFrames]) -> Result<Vec<f32>> {
        let views: Vec<_> = blocks.iter().map(|b| b.features.view()).collect();
        let soft = ndarray::concatenate(Axis(0), &views)?;
        let frames = blocks.iter().map(|b| b.frames).sum();
        let e = self.embed_frames(soft, frames, frames)?;
        Ok(e.rows.into_iter().next().unwrap_or_default())
    }

    /// Embed one **mixed** query — text plus images and audio clips — as a
    /// single interleaved sequence:
    ///
    /// ```text
    ///   <bos> task: search result | query: {text}
    ///         ( <|image> soft… <image|> )*  ( <|audio> soft… <audio|> )*  <eos>
    /// ```
    ///
    /// The query prompt is included only when there is text. Audio clips longer
    /// than 30 s are truncated to 30 s.
    pub fn embed_query_mixed(&self, text: &str, images: &[Prepared], audio: &[&[f32]]) -> Result<Vec<f32>> {
        let c = &self.config;
        let mut ids: Vec<i64> = vec![BOS];
        if !text.trim().is_empty() {
            let enc = self.tokenizer.encode(format!("{QUERY_PREFIX}{text}"), false).map_err(Error::msg)?;
            ids.extend(enc.get_ids().iter().map(|&i| i as i64));
        }
        let mut image_parts = Vec::new();
        for img in images {
            let soft = self.vision_soft(&media::image::stack(&[img])?)?;
            ids.extend(delimited_body(c.boi_token_id, c.image_token_id, c.eoi_token_id, soft.shape()[0]));
            image_parts.push(soft);
        }
        let mut audio_parts = Vec::new();
        for clip in audio {
            let clip = &clip[..clip.len().min(media::mel::MAX_SAMPLES)];
            if clip.is_empty() {
                continue;
            }
            let soft = self.audio_soft(&[clip])?;
            ids.extend(delimited_body(c.boa_token_id, c.audio_token_id, c.eoa_token_index, soft.shape()[0]));
            audio_parts.push(soft);
        }
        ids.push(EOS);
        let concat = |parts: &[Array2<f32>]| -> Result<Option<Array2<f32>>> {
            if parts.is_empty() {
                return Ok(None);
            }
            let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
            Ok(Some(ndarray::concatenate(Axis(0), &views)?))
        };
        let soft = Soft { image: concat(&image_parts)?, audio: concat(&audio_parts)? };
        let (ids, mask) = repeat_rows(&ids, 1);
        let per_token = self.run_text(ids, mask.clone(), soft)?;
        Ok(self.pooled(&per_token, &mask)?.rows.into_iter().next().unwrap_or_default())
    }

    /// Attribute the cosine similarity between `query_vec` and each document to
    /// its individual tokens (exact: see the MLX backend's derivation — the
    /// model is linear after the encoder). Each document is `(prompt, text)`,
    /// the exact prompt it was indexed with.
    pub fn explain_similarity_batch(&self, query_vec: &[f32], docs: &[(String, String)]) -> Result<Vec<Explanation>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if query_vec.len() != self.dim {
            bail!("query_vec has dim {}, but this model emits {}", query_vec.len(), self.dim);
        }
        let inputs: Vec<String> = docs.iter().map(|(p, t)| format!("{p}{t}")).collect();
        let enc = self.tokenizer.encode_batch(inputs.clone(), true).map_err(Error::msg)?;
        let (ids, mask) = self.tokenize(&inputs)?;
        let per_token = self.run_text(ids, mask.clone(), Soft::default())?;
        let (b, l) = (per_token.shape()[0], per_token.shape()[1]);
        let mut raw = vec![0f32; b * l];
        for i in 0..b {
            for t in 0..l {
                let e = per_token.slice(ndarray::s![i, t, ..self.dim]);
                raw[i * l + t] = e.iter().zip(query_vec).map(|(x, q)| x * q).sum();
            }
        }
        let norms: Vec<f32> = (0..b)
            .map(|i| masked_mean(&per_token, &mask, i, self.dim).iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();
        Ok(crate::attribute(docs, &enc, &raw, l, &norms))
    }

    /// Run a throwaway pass so GPU pipeline setup happens up front.
    pub fn warmup(&self) -> Result<()> {
        self.embed_texts(&[format!("{QUERY_PREFIX}warmup")])?;
        Ok(())
    }

    // --- internals ---------------------------------------------------------

    fn tokenize(&self, inputs: &[String]) -> Result<(Array2<i64>, Array2<i64>)> {
        let enc = self.tokenizer.encode_batch(inputs.to_vec(), true).map_err(Error::msg)?;
        let (b, l) = (enc.len(), enc[0].len());
        let ids: Vec<i64> = enc.iter().flat_map(|e| e.get_ids().iter().map(|&i| i as i64)).collect();
        let mask: Vec<i64> = enc.iter().flat_map(|e| e.get_attention_mask().iter().map(|&m| m as i64)).collect();
        Ok((Array2::from_shape_vec((b, l), ids)?, Array2::from_shape_vec((b, l), mask)?))
    }

    /// The text encoder -> per-token (B, L, 768) states.
    fn run_text(&self, ids: Array2<i64>, mask: Array2<i64>, soft: Soft) -> Result<Array3<f32>> {
        let empty = || Array2::<f32>::zeros((0, SOFT_DIM));
        let _run = run_lock();
        let mut text = self.text.lock().map_err(|_| anyhow!("text session poisoned"))?;
        let out = text.run(ort::inputs![
            "input_ids" => Tensor::from_array(ids)?,
            "attention_mask" => Tensor::from_array(mask)?,
            "image_features" => Tensor::from_array(soft.image.unwrap_or_else(empty))?,
            "video_features" => Tensor::from_array(empty())?,
            "audio_features" => Tensor::from_array(soft.audio.unwrap_or_else(empty))?,
        ])?;
        Ok(out["last_hidden_state"].try_extract_array::<f32>()?.into_dimensionality::<Ix3>()?.to_owned())
    }

    /// Masked mean -> truncate to `dim` -> L2 normalize, per row.
    fn pooled(&self, per_token: &Array3<f32>, mask: &Array2<i64>) -> Result<Embeddings> {
        let rows = (0..per_token.shape()[0])
            .map(|i| {
                let mean = masked_mean(per_token, mask, i, self.dim);
                let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                mean.iter().map(|x| x / norm).collect()
            })
            .collect();
        Ok(Embeddings { rows, dim: self.dim })
    }

    fn vision_soft(&self, grid: &PatchGrid) -> Result<Array2<f32>> {
        let vision = self.vision.as_ref().ok_or_else(|| anyhow!("this model has no vision encoder"))?;
        let n = (grid.rows * grid.cols) as usize;
        let _run = run_lock();
        let mut vision = vision.lock().map_err(|_| anyhow!("vision session poisoned"))?;
        let out = vision.run(ort::inputs![
            "pixel_values" => Tensor::from_array(Array3::from_shape_vec((grid.frames, n, 768), grid.pixels.clone())?)?,
            "pixel_position_ids" => Tensor::from_array(Array3::from_shape_vec((grid.frames, n, 2), grid.positions.clone())?)?,
        ])?;
        Ok(out["image_features"].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?.to_owned())
    }

    /// Soft tokens for equal-length clips, (B · n, 512), clip-major.
    fn audio_soft(&self, clips: &[&[f32]]) -> Result<Array2<f32>> {
        let audio = self.audio.as_ref().ok_or_else(|| anyhow!("this model has no audio encoder"))?;
        let mut features = Vec::new();
        let mut masks = Vec::new();
        let mut frames = 0;
        for clip in clips {
            let (f, n, valid) = media::mel::log_mel_cpu(clip)?;
            if frames != 0 && n != frames {
                bail!("audio clips in one batch must be equal-length");
            }
            frames = n;
            features.extend(f);
            masks.extend((0..n).map(|i| i < valid));
        }
        let _run = run_lock();
        let mut audio = audio.lock().map_err(|_| anyhow!("audio session poisoned"))?;
        let out = audio.run(ort::inputs![
            "input_features" => Tensor::from_array(Array3::from_shape_vec((clips.len(), frames, 128), features)?)?,
            "input_features_mask" => Tensor::from_array(Array1::from_vec(masks).into_shape_with_order((clips.len(), frames))?)?,
        ])?;
        Ok(out["audio_features"].try_extract_array::<f32>()?.into_dimensionality::<Ix2>()?.to_owned())
    }

    /// `frames` frames of soft tokens (frame-major) -> one embedding per item
    /// of `per_item` frames.
    fn embed_frames(&self, soft: Array2<f32>, frames: usize, per_item: usize) -> Result<Embeddings> {
        if frames == 0 || frames % per_item != 0 {
            bail!("{frames} frames do not divide into items of {per_item}");
        }
        let n = soft.shape()[0] / frames;
        let c = &self.config;
        let row = delimited(&vec![(c.boi_token_id, c.image_token_id, c.eoi_token_id, n); per_item]);
        let (ids, mask) = repeat_rows(&row, frames / per_item);
        let per_token = self.run_text(ids, mask.clone(), Soft { image: Some(soft), audio: None })?;
        self.pooled(&per_token, &mask)
    }
}

/// `<open> placeholder×n <close>`.
fn delimited_body(open: u32, placeholder: u32, close: u32, n: usize) -> Vec<i64> {
    let mut ids = vec![open as i64];
    ids.extend(std::iter::repeat_n(placeholder as i64, n));
    ids.push(close as i64);
    ids
}

/// `<bos> (<open> placeholder×n <close>)… <eos>`.
fn delimited(segments: &[(u32, u32, u32, usize)]) -> Vec<i64> {
    let mut ids = vec![BOS];
    for &(open, placeholder, close, n) in segments {
        ids.extend(delimited_body(open, placeholder, close, n));
    }
    ids.push(EOS);
    ids
}

/// `count` copies of one unpadded row, with an all-ones mask.
fn repeat_rows(row: &[i64], count: usize) -> (Array2<i64>, Array2<i64>) {
    let ids = Array2::from_shape_fn((count, row.len()), |(_, j)| row[j]);
    (ids, Array2::ones((count, row.len())))
}

/// Mean of row `i`'s valid tokens, truncated to the first `dim` features.
fn masked_mean(per_token: &Array3<f32>, mask: &Array2<i64>, i: usize, dim: usize) -> Vec<f32> {
    let mut sum = vec![0f32; dim];
    let mut n = 0f32;
    for t in 0..per_token.shape()[1] {
        if mask[[i, t]] == 0 {
            continue;
        }
        n += 1.0;
        for (s, x) in sum.iter_mut().zip(per_token.slice(ndarray::s![i, t, ..dim])) {
            *s += x;
        }
    }
    let n = n.max(1e-9);
    sum.iter().map(|s| s / n).collect()
}
