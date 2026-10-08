//! EmbeddingGemma 2: one shared embedding space for text, images, audio and
//! video, on MLX (Apple Silicon) or ONNX Runtime (everywhere else) — see
//! build.rs. This root holds what both backends share: prompts, output types,
//! and the exact per-token attribution math.
//!
//! Pipeline (text): tokenize -> `embed_tokens` × √hidden -> 24-layer
//! bidirectional Gemma-4 encoder -> per-token 512→768 projection -> masked mean
//! pool -> L2 normalize. Media inputs swap their placeholder tokens for soft
//! tokens produced by the vision/audio towers before the encoder runs.
//!
//! Precision (MLX): weights and activations are BF16 (the checkpoint's native
//! dtype). Never F16 on MLX — the model's activations overflow its range and
//! silently produce NaN/degraded vectors. (The ONNX fp16 export keeps the
//! overflow-prone ops in F32; its vectors match MLX to cos ≥ 0.9997.) Pooling
//! and normalization run in F32.

pub mod config;
pub mod media;
pub mod service;

#[cfg(backend_mlx)]
mod mlx;
#[cfg(backend_mlx)]
pub use mlx::vision::{self, PatchGrid};
#[cfg(backend_mlx)]
pub use mlx::{set_metallib_path, set_onnxruntime_path, Model, SoftFrames};

#[cfg(backend_onnx)]
mod onnx;
#[cfg(backend_onnx)]
pub use onnx::{set_metallib_path, set_onnxruntime_path, Model, PatchGrid, SoftFrames};

pub use crate::service::{EmbedService, Lane};

/// The backend this build runs: `"mlx"` (Apple Silicon) or `"onnx"`.
#[cfg(backend_mlx)]
pub const BACKEND: &str = "mlx";
/// The backend this build runs: `"mlx"` (Apple Silicon) or `"onnx"`.
#[cfg(backend_onnx)]
pub const BACKEND: &str = "onnx";

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

/// Row-major unit-norm embeddings.
pub struct Embeddings {
    pub rows: Vec<Vec<f32>>,
    pub dim: usize,
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

/// Turn raw per-token scores into [`Explanation`]s — shared by both backends.
///
/// `raw[b·l + t]` is q·eₜ[..dim] for row `b`'s token `t` (rows padded to `l`),
/// and `norms[b]` is ‖p‖, the norm of row `b`'s truncated mean-pooled vector,
/// so token t's exact share of the cosine is raw / (N·‖p‖). `enc` are the
/// tokenizer encodings of `prompt + text` for each doc (with specials).
pub(crate) fn attribute(
    docs: &[(String, String)],
    enc: &[tokenizers::Encoding],
    raw: &[f32],
    l: usize,
    norms: &[f32],
) -> Vec<Explanation> {
    let mut out = Vec::with_capacity(docs.len());
    for (b, ((prompt, text), e)) in docs.iter().zip(enc).enumerate() {
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
    out
}
