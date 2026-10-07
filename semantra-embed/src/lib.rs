//! EmbeddingGemma 2 on MLX: one shared embedding space for text, images, audio
//! and video.
//!
//! Pipeline (text): tokenize -> `embed_tokens` × √hidden -> 24-layer
//! bidirectional Gemma-4 encoder -> per-token 512→768 projection -> masked mean
//! pool -> L2 normalize. Media inputs swap their placeholder tokens for soft
//! tokens produced by the vision/audio towers before the encoder runs.
//!
//! Precision: weights and activations are BF16 (the checkpoint's native dtype).
//! Never F16 — the model's activations overflow its range and silently produce
//! NaN/degraded vectors. Pooling and normalization run in F32.

mod audio;
pub mod config;
pub mod media;
pub mod service;
mod text;
pub mod vision;
mod weights;

use std::path::Path;

use anyhow::{anyhow, bail, Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use tokenizers::Tokenizer;

use crate::audio::AudioModel;
use crate::config::ModelConfig;
use crate::text::TextModel;
use crate::vision::{PatchGrid, VisionModel};
use crate::weights::Weights;

pub use crate::service::{EmbedService, Lane};

/// Search-query prompt (the model's `SearchQuery` / `Retrieval-query` prompt).
pub const QUERY_PREFIX: &str = "task: search result | query: ";

/// Format a document chunk the way the model was trained to see documents:
/// `title: {title} | text: {content}`, with `title: none` when untitled.
pub fn document_prompt(title: Option<&str>) -> String {
    let title = title.map(str::trim).filter(|t| !t.is_empty()).unwrap_or("none");
    format!("title: {title} | text: ")
}

/// Native output width. Matryoshka-truncated sizes (512/256/128) are prefixes
/// of this vector, re-normalized.
pub const FULL_DIM: usize = 768;

/// Longest text input embedded (tokens, incl. specials); longer inputs are
/// truncated. Well within the model's 8K context.
pub const MAX_INPUT_TOKENS: usize = 2048;

pub struct Model {
    text: TextModel,
    /// Present when the checkpoint ships the vision tower (images, PDF page
    /// renders, video frames).
    vision: Option<VisionModel>,
    /// Present when the checkpoint ships the audio tower (audio files and
    /// video soundtracks).
    audio: Option<AudioModel>,
    tokenizer: Tokenizer,
    config: ModelConfig,
    /// Output dimension after Matryoshka truncation (<= [`FULL_DIM`]).
    dim: usize,
}

/// Vision-tower output for a run of video frames, (frames, n, hidden), kept
/// on the inference thread's device. Lets overlapping video windows reuse each
/// frame's soft tokens instead of re-running the vision tower per window.
pub struct SoftFrames(Array);

impl SoftFrames {
    pub fn frames(&self) -> usize {
        self.0.shape()[0] as usize
    }
}

/// Row-major unit-norm embeddings.
pub struct Embeddings {
    pub rows: Vec<Vec<f32>>,
    pub dim: usize,
}

impl Model {
    /// Load the model from `dir` (a copy of the `google/embeddinggemma-2` repo:
    /// `config.json`, `tokenizer.json`, `model.safetensors`). `dim` selects the
    /// Matryoshka output size: 768 (native), 512, 256 or 128.
    pub fn load(dir: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let dir = dir.as_ref();
        if ![768, 512, 256, 128].contains(&dim) {
            bail!("unsupported embedding dim {dim}; expected 768, 512, 256 or 128");
        }
        let config: ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(Error::msg)?;
        // Pad to the longest row in each batch (right side, `<pad>` = 0).
        // Callers sort inputs by length before batching so this stays small.
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.text_config.pad_token_id,
            pad_token: "<pad>".into(),
            ..Default::default()
        }));
        // Hard cap on any single text input. Prose chunks are ~265 tokens, but a
        // CSV cell (embedded whole) or a pathological input could otherwise be
        // tens of thousands of tokens — quadratic attention memory in one pass.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_INPUT_TOKENS,
                ..Default::default()
            }))
            .map_err(Error::msg)?;

        let weights = Weights::load(dir.join("model.safetensors"))?;
        let text = TextModel::load(&weights.scope("language_model"), &config.text_config)?;
        let vision = weights
            .get("vision_tower.patch_embedder.input_proj.weight")
            .is_ok()
            .then(|| VisionModel::load(&weights.scope("vision_tower"), &weights.scope("embed_vision")))
            .transpose()?;
        let audio = weights
            .get("audio_tower.output_proj.weight")
            .is_ok()
            .then(|| {
                let a = &config.audio_config;
                AudioModel::load(&weights.scope("audio_tower"), &weights.scope("embed_audio"), a.hidden_size, a.num_attention_heads)
            })
            .transpose()?;
        if text.dtype() == Dtype::Float16 {
            bail!("refusing F16 weights: EmbeddingGemma 2 overflows F16; use BF16 or F32");
        }
        Ok(Model {
            text,
            vision,
            audio,
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
    /// [`document_prompt`]) in one batched forward pass.
    pub fn embed_texts(&self, inputs: &[String]) -> Result<Embeddings> {
        if inputs.is_empty() {
            return Ok(Embeddings { rows: vec![], dim: self.dim });
        }
        let (tokens, valid) = self.tokenize(inputs)?;
        let per_token = self.text.forward(&self.text.embed(&tokens)?, valid.as_ref())?;
        let pooled = self.pool(&per_token, valid.as_ref())?;
        Ok(Embeddings {
            rows: rows_of(&pooled, self.dim)?,
            dim: self.dim,
        })
    }

    /// Embed images or video windows (no text prompt — the model takes media
    /// bare). `grid` holds `B × frames_per_item` same-shaped frames, item-major;
    /// each item becomes one sequence
    ///
    /// ```text
    ///   <bos> ( <|image> soft×n <image|> ) × frames_per_item <eos>
    /// ```
    ///
    /// and one embedding. An image is `frames_per_item = 1`; a video window
    /// stacks its sampled frames (each frame block uses the same delimiters as
    /// an image — only the placeholder id differs, and placeholders are
    /// replaced by soft tokens anyway).
    pub fn embed_visual(&self, grid: &PatchGrid, frames_per_item: usize) -> Result<Embeddings> {
        let soft = self.vision()?.forward(grid)?;
        let c = &self.config;
        self.embed_soft_sequences(soft, frames_per_item, [c.boi_token_id, c.eoi_token_id])
    }

    /// [`embed_visual`](Self::embed_visual) on pre-patchified frames
    /// ((B·F, rows·cols, 768) in [0, 1]) — the processor's own layout, used to
    /// verify the tower independently of image resizing.
    pub fn embed_visual_patches(
        &self,
        patches: &Array,
        rows: i32,
        cols: i32,
        frames_per_item: usize,
    ) -> Result<Embeddings> {
        let soft = self.vision()?.forward_patches(patches, rows, cols)?;
        let c = &self.config;
        self.embed_soft_sequences(soft, frames_per_item, [c.boi_token_id, c.eoi_token_id])
    }

    /// Embed equal-length 16 kHz mono clips (≤ 30 s each; no text prompt).
    /// Each becomes `<bos> <|audio> soft×n <audio|> <eos>` with one soft token
    /// per 40 ms. Equal lengths keep the batch padding-free, so a ragged final
    /// window of a recording should be embedded on its own.
    pub fn embed_audio(&self, clips: &[&[f32]]) -> Result<Embeddings> {
        let tower = self
            .audio
            .as_ref()
            .ok_or_else(|| anyhow!("this checkpoint has no audio tower"))?;
        let (mel, valid) = media::mel::log_mel(clips)?;
        let (soft, soft_valid) = tower.forward(&mel, &valid)?;
        // Keep only real soft tokens (the count is identical across the
        // batch, since clips are equal-length).
        let n_real = soft_valid.as_dtype(Dtype::Int32)?.sum_axis(1, false)?.max(None)?.item_cast::<i32>();
        let soft = soft.index((.., 0..n_real, ..));
        let c = &self.config;
        self.embed_soft_sequences(soft, 1, [c.boa_token_id, c.eoa_token_index])
    }

    /// Run only the vision tower over `grid` (one video frame per row) and
    /// keep the soft tokens for [`embed_frame_window`](Self::embed_frame_window).
    pub fn visual_soft_tokens(&self, grid: &PatchGrid) -> Result<SoftFrames> {
        let soft = self.vision()?.forward(grid)?;
        soft.eval()?;
        Ok(SoftFrames(soft))
    }

    /// Embed one video window made of consecutive `blocks` of frames (all from
    /// [`visual_soft_tokens`](Self::visual_soft_tokens), same frame size).
    pub fn embed_frame_window(&self, blocks: &[&SoftFrames]) -> Result<Vec<f32>> {
        let parts: Vec<&Array> = blocks.iter().map(|b| &b.0).collect();
        let soft = mlx_rs::ops::concatenate(&parts, 0)?;
        let frames = soft.shape()[0] as usize;
        let c = &self.config;
        let e = self.embed_soft_sequences(soft, frames, [c.boi_token_id, c.eoi_token_id])?;
        Ok(e.rows.into_iter().next().unwrap_or_default())
    }

    fn vision(&self) -> Result<&VisionModel> {
        self.vision
            .as_ref()
            .ok_or_else(|| anyhow!("this checkpoint has no vision tower"))
    }

    /// Splice per-segment soft tokens (B·F, n, hidden) into sequences
    /// `<bos> (open soft×n close) × F <eos>` and run the encoder: one pooled
    /// embedding per item.
    fn embed_soft_sequences(
        &self,
        soft: Array,
        frames_per_item: usize,
        [open, close]: [u32; 2],
    ) -> Result<Embeddings> {
        let f = frames_per_item.max(1) as i32;
        let (bf, n, hidden) = (soft.shape()[0], soft.shape()[1], soft.shape()[2]);
        if bf % f != 0 {
            bail!("{bf} frames do not divide into items of {f}");
        }
        let b = bf / f;
        // <bos>, open, close, <eos>
        let delims = self.text.embed(&Array::from_slice(&[2, open as i32, close as i32, 1], &[4]))?;
        let row = |i: i32| delims.take_axis(Array::from_int(i), 0).and_then(|r| r.reshape(&[1, 1, hidden]));
        let (bos, open, close, eos) = (row(0)?, row(1)?, row(2)?, row(3)?);
        let tile = |r: &Array, count: i32| mlx_rs::ops::broadcast_to(r, &[count, 1, hidden]);
        // Per segment: [open, soft…, close] -> (B·F, n+2, hidden)
        let framed = mlx_rs::ops::concatenate(
            &[tile(&open, bf)?, soft.as_dtype(self.text.dtype())?, tile(&close, bf)?],
            1,
        )?;
        let framed = framed.reshape(&[b, f * (n + 2), hidden])?;
        let seq = mlx_rs::ops::concatenate(&[tile(&bos, b)?, framed, tile(&eos, b)?], 1)?;
        let per_token = self.text.forward(&seq, None)?;
        let pooled = self.pool(&per_token, None)?;
        Ok(Embeddings {
            rows: rows_of(&pooled, self.dim)?,
            dim: self.dim,
        })
    }

    /// Tokenize a batch -> ((B, L) int32 ids, (B, L) bool validity mask).
    /// The mask is `None` when no row needed padding.
    fn tokenize(&self, inputs: &[String]) -> Result<(Array, Option<Array>)> {
        let enc = self
            .tokenizer
            .encode_batch(inputs.to_vec(), true)
            .map_err(Error::msg)?;
        let (b, l) = (enc.len() as i32, enc[0].len() as i32);
        let ids: Vec<i32> = enc.iter().flat_map(|e| e.get_ids().iter().map(|&i| i as i32)).collect();
        let tokens = Array::from_slice(&ids, &[b, l]);
        let padded = enc.iter().any(|e| e.get_attention_mask().iter().any(|&m| m == 0));
        let valid = padded.then(|| {
            let m: Vec<bool> = enc
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|&m| m == 1))
                .collect();
            Array::from_slice(&m, &[b, l])
        });
        Ok((tokens, valid))
    }

    /// Masked mean over tokens in F32 -> (B, FULL_DIM), un-normalized.
    fn pool(&self, per_token: &Array, valid: Option<&Array>) -> Result<Array> {
        let x = per_token.as_dtype(Dtype::Float32)?;
        Ok(match valid {
            None => x.mean_axis(1, false)?,
            Some(v) => {
                let m = v.as_dtype(Dtype::Float32)?.expand_dims(-1)?;
                let summed = x.multiply(&m)?.sum_axis(1, false)?;
                summed.divide(mlx_rs::ops::maximum(m.sum_axis(1, false)?, Array::from_f32(1e-9))?)?
            }
        })
    }

    /// Attribute the cosine similarity between `query_vec` and each document to
    /// its individual tokens, in one batched forward pass.
    ///
    /// Exact, not a saliency heuristic: the model is linear after the encoder
    /// (per-token bias-free projection eᵢ, masked mean, truncate, L2), so with
    /// p = (1/N)·Σᵢ eᵢ[..dim]
    ///
    /// ```text
    ///   score = q·p / ‖p‖ = Σᵢ (q·eᵢ[..dim]) / (N·‖p‖)      (bias = 0)
    /// ```
    ///
    /// Each document is `(prompt, text)` — the exact prompt it was indexed with
    /// (see [`document_prompt`]), so the reproduced vector equals the stored one.
    /// Offsets are UTF-8 **byte** offsets into `text` alone; prompt tokens and
    /// `<bos>`/`<eos>` are returned as `special` (they carry real score mass but
    /// are not chunk text).
    pub fn explain_similarity_batch(
        &self,
        query_vec: &[f32],
        docs: &[(String, String)],
    ) -> Result<Vec<Explanation>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if query_vec.len() != self.dim {
            bail!("query_vec has dim {}, but this model emits {}", query_vec.len(), self.dim);
        }
        let inputs: Vec<String> = docs.iter().map(|(p, t)| format!("{p}{t}")).collect();
        let enc = self.tokenizer.encode_batch(inputs.clone(), true).map_err(Error::msg)?;
        let (tokens, valid) = self.tokenize(&inputs)?;
        let per_token = self
            .text
            .forward(&self.text.embed(&tokens)?, valid.as_ref())?
            .as_dtype(Dtype::Float32)?;
        let dim = self.dim as i32;
        let truncated = per_token.index((.., .., 0..dim));
        let q = Array::from_slice(query_vec, &[dim, 1]);
        // raw[b, t] = q · eₜ[..dim]; ‖p‖ from the same truncated mean as `embed`.
        let raw = truncated.matmul(&q)?.squeeze_axes(&[-1])?;
        let pooled = self.pool(&truncated, valid.as_ref())?;
        let norms = pooled.square()?.sum_axis(-1, false)?.sqrt()?;
        raw.eval()?;
        norms.eval()?;
        let l = raw.shape()[1] as usize;
        let raw: &[f32] = raw.try_as_slice().map_err(|e| anyhow!("{e}"))?;
        let norms: &[f32] = norms.try_as_slice().map_err(|e| anyhow!("{e}"))?;

        let mut out = Vec::with_capacity(docs.len());
        for (b, ((prompt, text), e)) in docs.iter().zip(&enc).enumerate() {
            let mask = e.get_attention_mask();
            let n = mask.iter().filter(|&&m| m == 1).count() as f32;
            let denom = if n > 0.0 && norms[b] > 0.0 { n * norms[b] } else { f32::INFINITY };
            let p_len = prompt.len();
            let mut word: Option<u32> = None;
            let mut tokens = Vec::new();
            let mut total = 0.0;
            for (t, &(start, end)) in e.get_offsets().iter().enumerate() {
                if mask[t] == 0 {
                    continue;
                }
                let score = raw[b * l + t] / denom;
                total += score;
                let special = e.get_special_tokens_mask()[t] == 1 || end <= p_len;
                if special {
                    tokens.push(TokenContribution { start: 0, end: 0, word_id: None, score, special });
                    continue;
                }
                // Gemma pieces carry their leading space ("▁word"). A piece that
                // starts at whitespace begins a new word; trim that space so the
                // span covers only visible text.
                let (start, end) = (start.max(p_len) - p_len, end - p_len);
                let lead = text[start..end].len() - text[start..end].trim_start().len();
                if lead > 0 || word.is_none() {
                    word = Some(word.map_or(0, |w| w + 1));
                }
                tokens.push(TokenContribution {
                    start: start + lead,
                    end,
                    word_id: word,
                    score,
                    special: false,
                });
            }
            out.push(Explanation { tokens, bias: 0.0, total });
        }
        Ok(out)
    }

    /// Run a throwaway pass so Metal pipeline compilation happens up front
    /// rather than on the user's first real embed.
    pub fn warmup(&self) -> Result<()> {
        self.embed_texts(&[format!("{QUERY_PREFIX}warmup")])?;
        Ok(())
    }
}

/// One token's exact, signed share of a cosine similarity score. Summing every
/// token's `score` (plus [`Explanation::bias`], always 0 for this model)
/// reproduces [`Explanation::total`].
#[derive(Clone, Debug)]
pub struct TokenContribution {
    /// Byte offset (inclusive) of this token's visible text in the document.
    pub start: usize,
    /// Byte offset (exclusive).
    pub end: usize,
    /// Sub-word pieces of one whitespace-delimited word share an id; `None` for
    /// special/prompt tokens.
    pub word_id: Option<u32>,
    pub score: f32,
    /// `<bos>`/`<eos>` and prompt tokens: real score mass, not document text.
    pub special: bool,
}

/// The full additive decomposition of one query↔document cosine similarity.
#[derive(Clone, Debug)]
pub struct Explanation {
    pub tokens: Vec<TokenContribution>,
    /// Constant share attributable to no token. The projection has no bias, so
    /// this is 0; kept so the wire format matches the previous model's.
    pub bias: f32,
    pub total: f32,
}

/// Truncate pooled (B, FULL_DIM) vectors to `dim` (Matryoshka) and L2-normalize
/// each row, returning host vectors. Truncation must precede normalization.
fn rows_of(pooled: &Array, dim: usize) -> Result<Vec<Vec<f32>>> {
    pooled.eval()?;
    let b = pooled.shape()[0] as usize;
    let full = pooled.shape()[1] as usize;
    let flat: &[f32] = pooled.try_as_slice().map_err(|e| anyhow!("{e}"))?;
    Ok((0..b)
        .map(|r| {
            let row = &flat[r * full..r * full + dim];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            row.iter().map(|x| x / norm).collect()
        })
        .collect())
}
