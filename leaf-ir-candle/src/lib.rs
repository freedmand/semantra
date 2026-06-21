// Shared pipeline for mdbr-leaf-ir, used by both main.rs and verify.rs so the
// test exercises the exact same code path as production.
//
// Pipeline (all facts confirmed from the model repo's config files):
//   BERT -> mean-pool (384-d) -> Dense linear 384->768 (bias, Identity) -> L2-norm

use std::path::Path;

use anyhow::{bail, Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;

pub const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

// File layout inside the model directory (produced by fetch.py / the notebook
// zip). `setup_model` joins these onto a caller-supplied base dir so the same
// pipeline works whether the files sit in ./models (CLI) or in a bundled app's
// resource directory (Tauri).
const CONFIG: &str = "config.json";
const TOKENIZER: &str = "tokenizer.json";
const BERT_WEIGHTS: &str = "model.safetensors";
const DENSE_WEIGHTS: &str = "2_Dense/model.safetensors";

pub struct ModelCtx {
    model: BertModel,
    dense: Linear,
    tokenizer: Tokenizer,
    device: Device,
    /// Output embedding dimension (the Dense layer's output width), detected from
    /// the model files. mdbr-leaf-ir is 768; mdbr-leaf-mt is 1024.
    dim: usize,
}

impl ModelCtx {
    /// The dimension of the unit-norm embeddings this model produces. Callers
    /// (e.g. a vector store) should size themselves to this rather than assume.
    pub fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// Row-major embeddings, decoupled from candle tensors for transparent test math.
pub struct Embeddings {
    pub rows: Vec<Vec<f32>>,
    pub dim: usize,
}

/// Choose the compute device for inference.
///
/// Default: prefer the Apple GPU via Metal, falling back to CPU when Metal is
/// unavailable (not compiled in on non-Apple targets — `new_metal` then returns
/// `NotCompiledWithMetalSupport` — or no usable GPU at runtime). Set the env var
/// `LEAF_IR_DEVICE=cpu` or `LEAF_IR_DEVICE=metal` to force a device for A/B
/// benchmarking; an explicit `metal` request that fails still falls back to CPU.
fn select_device() -> Device {
    let device = match std::env::var("LEAF_IR_DEVICE").ok().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("metal") => Device::new_metal(0).unwrap_or_else(|e| {
            eprintln!("[leaf-ir] LEAF_IR_DEVICE=metal requested but Metal unavailable ({e}); using CPU");
            Device::Cpu
        }),
        _ => Device::new_metal(0).unwrap_or(Device::Cpu),
    };
    eprintln!("[leaf-ir] embedding device: {:?}", device.location());
    device
}

/// Load tokenizer + BERT + dense head from `base_dir`, the directory that holds
/// `config.json`, `tokenizer.json`, `model.safetensors`, and `2_Dense/`.
pub fn setup_model(base_dir: impl AsRef<Path>) -> Result<ModelCtx> {
    let base_dir = base_dir.as_ref();
    let device = select_device();

    let mut tokenizer = Tokenizer::from_file(base_dir.join(TOKENIZER)).map_err(Error::msg)?;
    // Always pad to the longest sequence in the *batch*, overriding whatever the
    // tokenizer file ships (mdbr-leaf-mt ships Fixed(128)). Our chunks are ~50
    // words ≈ 65-80 tokens, so Fixed(128) computes ~1.5-2x more token-positions
    // than needed on every chunk. Pad tokens are masked out in the mean-pool, so
    // BatchLongest yields *identical* embeddings — this is a pure throughput win.
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    let config: Config = serde_json::from_str(&std::fs::read_to_string(base_dir.join(CONFIG))?)?;
    let bert_weights = base_dir.join(BERT_WEIGHTS);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[bert_weights], DTYPE, &device)? };
    let model = BertModel::load(vb, &config)?;

    // The Dense head maps BERT's hidden size to the model's embedding dim. Both
    // are detected from the files (hidden size from config; output dim from the
    // weight) rather than hardcoded, so different "leaf" models load unchanged —
    // e.g. mdbr-leaf-ir projects 384->768 (with bias) and mdbr-leaf-mt 384->1024
    // (no bias).
    let (dense, dim) = load_dense(base_dir.join(DENSE_WEIGHTS), config.hidden_size, &device)?;

    Ok(ModelCtx {
        model,
        dense,
        tokenizer,
        device,
        dim,
    })
}

/// Load the Dense projection head, returning it alongside its detected output
/// dimension. The bias is optional (some models, e.g. mdbr-leaf-mt, omit it). The
/// input width is validated against `in_dim` (BERT's hidden size).
fn load_dense(path: impl AsRef<Path>, in_dim: usize, device: &Device) -> Result<(Linear, usize)> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let keys: Vec<&String> = tensors.keys().collect();
    let Some(weight) = tensors
        .get("linear.weight")
        .or_else(|| tensors.get("weight"))
        .cloned()
    else {
        bail!("Dense safetensors missing a weight tensor; keys: {keys:?}");
    };
    // Bias is optional; accept either naming, tolerate absence.
    let bias = tensors
        .get("linear.bias")
        .or_else(|| tensors.get("bias"))
        .cloned();

    let (out_dim, wi) = weight.dims2()?;
    if wi != in_dim {
        bail!("Dense weight input {wi}, expected {in_dim} (BERT hidden size)");
    }
    Ok((Linear::new(weight, bias), out_dim))
}

/// Embed sentences EXACTLY as already-prefixed/raw strings (caller adds the
/// query prefix where appropriate). Returns row-major unit-norm embeddings.
pub fn embed_sentences(ctx: &ModelCtx, sentences: &[String]) -> Result<Embeddings> {
    let encodings = ctx
        .tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(Error::msg)?;

    let (mut ids, mut masks) = (Vec::new(), Vec::new());
    for enc in &encodings {
        ids.push(enc.get_ids().to_vec());
        masks.push(enc.get_attention_mask().to_vec());
    }
    let (batch, seq_len) = (ids.len(), ids[0].len());

    let token_ids = Tensor::from_vec(ids.concat(), (batch, seq_len), &ctx.device)?;
    let attention_mask = Tensor::from_vec(masks.concat(), (batch, seq_len), &ctx.device)?;
    let token_type_ids = token_ids.zeros_like()?;

    let hidden = ctx
        .model
        .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

    // mask-aware mean pool
    let mask_f = attention_mask.to_dtype(DTYPE)?;
    let mask_exp = mask_f.unsqueeze(2)?.broadcast_as(hidden.shape())?;
    let summed = (hidden * &mask_exp)?.sum(1)?;
    let counts = mask_f.sum(1)?.unsqueeze(1)?;
    let pooled = summed.broadcast_div(&counts)?;

    // dense projection (hidden size -> embedding dim)
    let projected = ctx.dense.forward(&pooled)?;

    // L2 normalize
    let norm = projected.sqr()?.sum_keepdim(1)?.sqrt()?;
    let normalized = projected.broadcast_div(&norm)?;

    let rows: Vec<Vec<f32>> = normalized.to_vec2()?;
    let dim = rows.first().map(|r| r.len()).unwrap_or(0);
    Ok(Embeddings { rows, dim })
}

/// One token's exact, signed share of a cosine similarity score.
///
/// Because this model is **linear after BERT** (mask-aware mean-pool, then a
/// Dense projection `W·x + b`), the cosine similarity between a query `q` and a
/// document decomposes EXACTLY into a sum of per-token terms plus one constant:
///
/// ```text
///   score = q·d / ‖d‖,   d = W·(1/N · Σᵢ hᵢ) + b
///         = Σᵢ (hᵢ·u)/(N·‖d‖)  +  (q·b)/‖d‖,    where u = Wᵀq
///           └──── token i's contribution ────┘     └─── bias ───┘
/// ```
///
/// so `score` below is token `i`'s additive share of the final similarity:
/// summing every returned token's `score` and adding [`Explanation::bias`]
/// reproduces [`Explanation::total`]. Values are signed — a negative `score`
/// is a token that pulls the similarity *down*.
#[derive(Clone, Debug)]
pub struct TokenContribution {
    /// Byte offset (inclusive start) of this token in the original document.
    pub start: usize,
    /// Byte offset (exclusive end) of this token in the original document.
    pub end: usize,
    /// Groups WordPiece sub-tokens belonging to the same whitespace word so a
    /// caller can aggregate pieces back to whole words; `None` for special tokens.
    pub word_id: Option<u32>,
    /// This token's exact additive share of the cosine similarity (see above).
    pub score: f32,
    /// True for model special tokens (`[CLS]`/`[SEP]`): they carry real
    /// contribution mass but are not part of the document text, so callers
    /// usually skip *rendering* them while still counting their `score`.
    pub special: bool,
}

/// The full additive decomposition of one query↔document cosine similarity.
///
/// `total == bias + Σ tokens[i].score` (up to f32 rounding), so a caller never
/// has to recompute the score — it can present any subset of tokens and still
/// know each one's exact fraction of the whole.
#[derive(Clone, Debug)]
pub struct Explanation {
    /// One entry per non-padding token, in document order.
    pub tokens: Vec<TokenContribution>,
    /// The constant share attributable to no single token (`q·b / ‖d‖`).
    pub bias: f32,
    /// The cosine similarity this explanation decomposes.
    pub total: f32,
}

/// Attribute the cosine similarity between `query_vec` and each document in
/// `documents` to its individual tokens, in a single batched BERT forward pass.
///
/// `query_vec` must be a unit-norm embedding produced by this same model (i.e.
/// one row of [`embed_sentences`]), already including the query prefix. Each
/// returned [`Explanation`] carries the exact per-token decomposition described
/// on [`TokenContribution`]; the math is exact (not a saliency heuristic)
/// because the post-BERT pipeline is linear.
///
/// Like [`embed_sentences`] this is synchronous and CPU/GPU-bound; run it on a
/// blocking thread when an event loop must stay responsive.
pub fn explain_similarity_batch(
    ctx: &ModelCtx,
    query_vec: &[f32],
    documents: &[String],
) -> Result<Vec<Explanation>> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }
    if query_vec.len() != ctx.dim {
        bail!(
            "query_vec has dim {}, but this model emits {}",
            query_vec.len(),
            ctx.dim
        );
    }

    let encodings = ctx
        .tokenizer
        .encode_batch(documents.to_vec(), true)
        .map_err(Error::msg)?;

    let mut ids = Vec::with_capacity(encodings.len());
    let mut masks = Vec::with_capacity(encodings.len());
    for enc in &encodings {
        ids.push(enc.get_ids().to_vec());
        masks.push(enc.get_attention_mask().to_vec());
    }
    let (batch, seq_len) = (ids.len(), ids[0].len());

    let token_ids = Tensor::from_vec(ids.concat(), (batch, seq_len), &ctx.device)?;
    let attention_mask = Tensor::from_vec(masks.concat(), (batch, seq_len), &ctx.device)?;
    let token_type_ids = token_ids.zeros_like()?;

    // (batch, seq_len, hidden) contextual hidden states — the same tensor the
    // mean-pool in `embed_sentences` consumes.
    let hidden = ctx
        .model
        .forward(&token_ids, &token_type_ids, Some(&attention_mask))?
        .contiguous()?;
    let hidden_size = hidden.dim(2)?;

    // u = Wᵀq in BERT-hidden space, so that hᵢ·u == (W·hᵢ)·q without ever
    // materializing the 768-d per-token projections. q (1,dim) · W (dim,hidden).
    let q = Tensor::from_vec(query_vec.to_vec(), (1, ctx.dim), &ctx.device)?;
    let u_t = q.matmul(ctx.dense.weight())?.t()?.contiguous()?; // (hidden, 1)

    // raw[b][t] = hidden[b,t] · u  (the un-scaled token term, pre 1/(N·‖d‖)).
    let raw = hidden
        .reshape((batch * seq_len, hidden_size))?
        .matmul(&u_t)?
        .reshape((batch, seq_len))?
        .to_vec2::<f32>()?;

    // Reproduce the document embedding exactly to recover N and ‖d‖ per row:
    // mask-aware mean-pool -> Dense (adds bias ONCE) -> L2 norm. `counts` is N.
    let mask_f = attention_mask.to_dtype(DTYPE)?;
    let mask_exp = mask_f.unsqueeze(2)?.broadcast_as(hidden.shape())?;
    let summed = (&hidden * &mask_exp)?.sum(1)?;
    let counts = mask_f.sum(1)?.unsqueeze(1)?; // (batch, 1) == N per row
    let pooled = summed.broadcast_div(&counts)?;
    let projected = ctx.dense.forward(&pooled)?;
    let norms = projected.sqr()?.sum_keepdim(1)?.sqrt()?; // (batch, 1) == ‖d‖

    let counts_v = counts.to_vec2::<f32>()?;
    let norms_v = norms.to_vec2::<f32>()?;

    // (q·b) is shared across rows; combined with each row's ‖d‖ it is the bias share.
    let qb: f32 = match ctx.dense.bias() {
        Some(b) => {
            let bv = b.to_vec1::<f32>()?;
            query_vec.iter().zip(&bv).map(|(x, y)| x * y).sum()
        }
        None => 0.0,
    };

    let mut out = Vec::with_capacity(batch);
    for (b, enc) in encodings.iter().enumerate() {
        let n = counts_v[b][0];
        let norm = norms_v[b][0];
        // Degenerate row (empty/zero-norm): everything is zero rather than NaN.
        let denom = if n > 0.0 && norm > 0.0 { n * norm } else { f32::INFINITY };
        let bias = if norm > 0.0 { qb / norm } else { 0.0 };

        let offsets = enc.get_offsets();
        let word_ids = enc.get_word_ids();
        let special = enc.get_special_tokens_mask();
        let mask = enc.get_attention_mask();

        let mut tokens = Vec::new();
        let mut sum = bias;
        for t in 0..seq_len {
            if mask[t] == 0 {
                continue; // padding — not part of this document
            }
            let score = raw[b][t] / denom;
            sum += score;
            let (start, end) = offsets[t];
            tokens.push(TokenContribution {
                start,
                end,
                word_id: word_ids[t],
                score,
                special: special[t] == 1,
            });
        }

        out.push(Explanation {
            tokens,
            bias,
            total: sum,
        });
    }

    Ok(out)
}

/// Run one throwaway forward pass to trigger lazy, one-time device work up front.
///
/// On Metal this compiles the compute pipelines the model uses (~2s the first
/// time); doing it now — e.g. on a background thread at app startup — keeps that
/// cost off the user's first real embed. On CPU it's a cheap no-op-equivalent.
/// The result is discarded.
pub fn warmup(ctx: &ModelCtx) -> Result<()> {
    embed_sentences(ctx, &["warmup".to_string()])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    // The whole approach rests on one claim: because the model is linear after
    // BERT, the cosine similarity equals the sum of per-token contributions plus
    // the bias. This proves it on the real model — `Explanation::total` and the
    // summed token scores must both reproduce the similarity computed the normal
    // way (embed both, dot the unit vectors).
    #[test]
    fn decomposition_reproduces_cosine() {
        let ctx = setup_model("models").expect("load model from ./models");

        let query = format!("{QUERY_PREFIX}What is machine learning?");
        let docs = [
            "Machine learning is a subset of artificial intelligence that learns from data."
                .to_string(),
            "The central bank raised interest rates to curb rising inflation.".to_string(),
        ];

        let q = embed_sentences(&ctx, &[query]).unwrap();
        let d = embed_sentences(&ctx, &docs).unwrap();
        let qvec = &q.rows[0];

        let explanations = explain_similarity_batch(&ctx, qvec, &docs).unwrap();
        assert_eq!(explanations.len(), docs.len());

        for (i, exp) in explanations.iter().enumerate() {
            let truth = dot(qvec, &d.rows[i]); // reference cosine (unit vectors)
            let summed = exp.bias + exp.tokens.iter().map(|t| t.score).sum::<f32>();

            // `total` is the bias + per-token sum the function itself built.
            assert!(
                (exp.total - truth).abs() < 1e-3,
                "doc {i}: total {} vs cosine {}",
                exp.total,
                truth
            );
            // And re-summing the returned tokens independently must agree too.
            assert!(
                (summed - truth).abs() < 1e-3,
                "doc {i}: Σtokens+bias {} vs cosine {}",
                summed,
                truth
            );
            // Offsets must land on char boundaries of the original doc (byte offsets).
            for t in &exp.tokens {
                assert!(docs[i].get(t.start..t.end).is_some(), "bad offsets {t:?}");
            }
        }
    }
}

// The frontend slices chunk text by these offsets using a UTF-8 byte array, so
// the offsets MUST be byte offsets that land on char boundaries — including for
// multi-byte typographic characters (curly quotes, em-dashes, accents) that are
// common in real PDFs. ASCII-only tests can't catch a byte/char unit mismatch.
#[test]
fn token_offsets_are_utf8_byte_offsets() {
    let ctx = setup_model("models").expect("load model from ./models");
    // Multi-byte: é (2 bytes), — (3), ï (2), ’ (3) — char_len 18, byte_len 24.
    let doc = "café — naïve EPA\u{2019}s".to_string();

    let q = embed_sentences(&ctx, &[format!("{QUERY_PREFIX}accents")]).unwrap();
    let exp = &explain_similarity_batch(&ctx, &q.rows[0], &[doc.clone()]).unwrap()[0];

    for t in &exp.tokens {
        // `str::get(byte_range)` returns Some ONLY when both ends are char
        // boundaries — the precise property the frontend's byte-slicing needs.
        let slice = doc.get(t.start..t.end);
        assert!(
            slice.is_some(),
            "token offsets ({},{}) are not UTF-8 char boundaries in {doc:?}",
            t.start,
            t.end
        );
        // Non-special tokens must cover real text; special tokens are (0,0).
        if !t.special {
            assert!(!slice.unwrap().is_empty(), "non-special token has empty span");
        }
    }
}
