//! The Gemma-4 (USM-style Conformer) audio tower: log-mel frames -> soft
//! tokens in text-model space, one per 40 ms of audio.
//!
//! ```text
//!   mel (T, 128) -> 2 × [conv3×3 stride 2 -> LayerNorm -> ReLU] -> linear 1024
//!   -> 12 Conformer blocks:
//!        ½·FFN -> chunked local self-attention (rel-pos, softcap 50)
//!        -> light conv (GLU, causal depthwise k=5) -> ½·FFN -> RMSNorm
//!   -> linear 1024→1536 (+bias) -> RMSNorm (no scale) -> 1536→512
//! ```
//!
//! Attention is *causal* and local: each 12-frame chunk attends to itself and
//! the 12 frames before it. Linears clamp their inputs/outputs to calibrated
//! ranges stored in the checkpoint ("clippable linears").
//!
//! Precision mirrors the reference: Q/K/V are upcast to F32 for attention, and
//! (as in the reference) that F32 then carries through the residual stream.

use anyhow::{bail, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fast, ops, Array, Dtype};

use crate::text::{split_last, Linear};
use crate::weights::Scope;

const EPS: f32 = 1e-6;
const CHUNK: i32 = 12;
const PAST: i32 = 12; // attention_context_left - 1
const CONTEXT: i32 = CHUNK + PAST; // no future context
const SOFTCAP: f32 = 50.0;
const INVALID_LOGIT: f32 = -1e9;
const RESIDUAL_WEIGHT: f32 = 0.5;
const CONV_KERNEL: i32 = 5;

/// Linear with input/output clamping to checkpoint-calibrated ranges.
struct ClipLinear {
    lin: Linear,
    in_range: (Array, Array),
    out_range: (Array, Array),
}

impl ClipLinear {
    fn load(w: &Scope, name: &str) -> Result<Self> {
        let s = w.scope(name);
        Ok(ClipLinear {
            lin: Linear::load(&s, "linear")?,
            in_range: (s.get("input_min")?, s.get("input_max")?),
            out_range: (s.get("output_min")?, s.get("output_max")?),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = ops::clip(x, (&self.in_range.0, &self.in_range.1))?;
        let y = self.lin.forward(&x)?;
        Ok(ops::clip(&y, (&self.out_range.0, &self.out_range.1))?)
    }
}

fn rms(w: &Scope, name: &str) -> Result<Array> {
    w.get(&format!("{name}.weight"))
}

fn rms_norm(x: &Array, weight: &Array) -> Result<Array> {
    Ok(fast::rms_norm(x, Some(weight), EPS)?)
}

fn silu(x: &Array) -> Result<Array> {
    Ok(x.multiply(ops::sigmoid(x)?)?)
}

struct FeedForward {
    pre: Array,
    ffw1: ClipLinear,
    ffw2: ClipLinear,
    post: Array,
}

impl FeedForward {
    fn load(w: &Scope) -> Result<Self> {
        Ok(FeedForward {
            pre: rms(w, "pre_layer_norm")?,
            ffw1: ClipLinear::load(w, "ffw_layer_1")?,
            ffw2: ClipLinear::load(w, "ffw_layer_2")?,
            post: rms(w, "post_layer_norm")?,
        })
    }

    /// Macaron half-step: `x + ½ · post(ffw2(silu(ffw1(pre(x)))))`. (The
    /// reference's ±1e10 "gradient clipping" clamps are no-ops at inference.)
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.ffw1.forward(&rms_norm(x, &self.pre)?)?;
        let h = self.ffw2.forward(&silu(&h)?)?;
        let h = rms_norm(&h, &self.post)?;
        Ok(x.add(h.multiply(Array::from_f32(RESIDUAL_WEIGHT).as_dtype(h.dtype())?)?)?)
    }
}

struct Attention {
    q: ClipLinear,
    k: ClipLinear,
    v: ClipLinear,
    post: ClipLinear,
    /// Relative-position key projection applied to sinusoidal offsets.
    rel_k: Linear,
    /// softplus(per_dim_scale) · head_dim^-½ / ln 2, precomputed (F32).
    q_scale: Array,
    heads: i32,
    head_dim: i32,
    hidden: i32,
}

impl Attention {
    fn load(w: &Scope, hidden: i32, heads: i32) -> Result<Self> {
        let head_dim = hidden / heads;
        let per_dim = w.get("per_dim_scale")?.as_dtype(Dtype::Float32)?;
        let softplus = ops::logaddexp(&per_dim, Array::from_f32(0.0))?;
        let base = (head_dim as f32).powf(-0.5) / std::f32::consts::LN_2;
        Ok(Attention {
            q: ClipLinear::load(w, "q_proj")?,
            k: ClipLinear::load(w, "k_proj")?,
            v: ClipLinear::load(w, "v_proj")?,
            post: ClipLinear::load(w, "post")?,
            rel_k: Linear::load(w, "relative_k_proj")?,
            q_scale: softplus.multiply(Array::from_f32(base))?,
            heads,
            head_dim,
            hidden,
        })
    }

    /// Sinusoidal embeddings for relative offsets 12, 11, …, 0 (past → now),
    /// projected per head: (heads, head_dim, 13).
    fn rel_pos(&self, dtype: Dtype) -> Result<Array> {
        let span = PAST + 1;
        let half = self.hidden / 2;
        let inc = (10_000f32).ln() / (half - 1).max(1) as f32;
        let inv: Vec<f32> = (0..half).map(|i| (-(i as f32) * inc).exp()).collect();
        let pos: Vec<f32> = (0..span).map(|i| (PAST - i) as f32).collect();
        let t = Array::from_slice(&pos, &[span, 1]).multiply(Array::from_slice(&inv, &[1, half]))?;
        let sig = ops::concatenate(&[ops::sin(&t)?, ops::cos(&t)?], -1)?; // (13, hidden)
        let proj = self.rel_k.forward(&sig.as_dtype(self.rel_k.dtype())?)?.as_dtype(dtype)?;
        Ok(proj
            .reshape(&[span, self.heads, self.head_dim])?
            .transpose_axes(&[1, 2, 0])?)
    }

    /// `x`: (B, T, hidden); `valid`: (B, T) bool.
    fn forward(&self, x: &Array, valid: &Array, local_mask: &Array) -> Result<Array> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let (n, h) = (self.heads, self.head_dim);
        let f32_heads = |y: Array| -> Result<Array> { Ok(y.as_dtype(Dtype::Float32)?.reshape(&[b, t, n, h])?) };
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        let q = f32_heads(self.q.forward(x)?)?.multiply(&self.q_scale)?;
        let k = f32_heads(self.k.forward(x)?)?.multiply(Array::from_f32(k_scale))?;
        let v = f32_heads(self.v.forward(x)?)?;

        // Queries in non-overlapping 12-frame blocks; keys/values (and key
        // validity) as each block's 24-frame window [block - 12, block + 12).
        let u = (t + CHUNK - 1) / CHUNK;
        let q_blocks = pad_time(&q, 0, u * CHUNK - t)?.reshape(&[b, u, CHUNK, n, h])?;
        let k_blocks = block_context(&k, u)?; // (B, U, C, N, H)
        let v_blocks = block_context(&v, u)?;
        let valid_blocks = block_context(&valid.as_dtype(Dtype::Float32)?, u)?; // (B, U, C)

        // Content term: (B, N, U, W, H) @ (B, N, U, H, C) -> (B, N, U, W, C)
        let qp = q_blocks.transpose_axes(&[0, 3, 1, 2, 4])?;
        let kp = k_blocks.transpose_axes(&[0, 3, 1, 4, 2])?;
        let ac = qp.matmul(&kp)?;
        // Position term over the 13 relative offsets, then shifted so column c
        // lines up with key c of the window.
        let bd = qp
            .reshape(&[b, n, u * CHUNK, h])?
            .matmul(self.rel_pos(Dtype::Float32)?)?
            .reshape(&[b, n, u, CHUNK, PAST + 1])?;
        let bd = relative_shift(&bd, b, n, u)?;
        let logits = ac.add(bd)?;
        let logits = ops::tanh(logits.divide(Array::from_f32(SOFTCAP))?)?.multiply(Array::from_f32(SOFTCAP))?;

        let cond = ops::logical_and(
            valid_blocks.gt(Array::from_f32(0.5))?.reshape(&[b, 1, u, 1, CONTEXT])?,
            local_mask.reshape(&[1, 1, 1, CHUNK, CONTEXT])?,
        )?;
        let logits = ops::select(&cond, &logits, Array::from_f32(INVALID_LOGIT))?;
        let probs = ops::softmax_axis(&logits, -1, None)?;
        // (B, N, U, W, C) @ (B, N, U, C, H) -> (B, N, U, W, H) -> (B, U·W, N·H)
        let ctx = probs.matmul(v_blocks.transpose_axes(&[0, 3, 1, 2, 4])?)?;
        let ctx = ctx
            .transpose_axes(&[0, 2, 3, 1, 4])?
            .reshape(&[b, u * CHUNK, n * h])?
            .index((.., 0..t, ..));
        self.post.forward(&ctx)
    }
}

/// Zero-pad axis 1 by `(before, after)`.
fn pad_time(x: &Array, before: i32, after: i32) -> Result<Array> {
    if before == 0 && after == 0 {
        return Ok(x.clone());
    }
    let mut widths = vec![(0, 0); x.ndim()];
    widths[1] = (before, after);
    Ok(ops::pad(x, widths.as_slice(), None, None)?)
}

/// (B, T, …) -> (B, U, CONTEXT, …): for each 12-frame block u, frames
/// [12u - 12, 12u + 12) (zero outside the sequence).
fn block_context(x: &Array, u: i32) -> Result<Array> {
    let t = x.shape()[1];
    let padded = pad_time(x, PAST, (u * CHUNK + CHUNK - 1) - t)?;
    let idx: Vec<i32> = (0..u).flat_map(|b| (0..CONTEXT).map(move |c| b * CHUNK + c)).collect();
    let gathered = padded.take_axis(Array::from_slice(&idx, &[u * CONTEXT]), 1)?;
    let mut shape = vec![x.shape()[0], u, CONTEXT];
    shape.extend_from_slice(&x.shape()[2..]);
    Ok(gathered.reshape(&shape)?)
}

/// Turn per-offset scores (…, W, 13) into per-key scores (…, W, 24): pad to 25,
/// flatten each block's rows, keep the first W·24, and re-fold. (Transformer-XL
/// style shift, as in the reference.)
fn relative_shift(bd: &Array, b: i32, n: i32, u: i32) -> Result<Array> {
    let padded = ops::pad(
        bd,
        &[(0, 0), (0, 0), (0, 0), (0, 0), (0, CONTEXT + 1 - (PAST + 1))][..],
        None,
        None,
    )?;
    Ok(padded
        .reshape(&[b, n, u, CHUNK * (CONTEXT + 1)])?
        .index((.., .., .., 0..CHUNK * CONTEXT))
        .reshape(&[b, n, u, CHUNK, CONTEXT])?)
}

/// (W, C) bool: query w may see window key c iff it is one of the 12 frames
/// strictly before-or-at w (causal, 12-frame horizon).
fn local_mask() -> Array {
    let mut m = Vec::with_capacity((CHUNK * CONTEXT) as usize);
    for w in 0..CHUNK {
        for c in 0..CONTEXT {
            let dist = w + PAST - c;
            m.push(dist >= 0 && dist < PAST);
        }
    }
    Array::from_slice(&m, &[CHUNK, CONTEXT])
}

struct LightConv {
    pre: Array,
    start: ClipLinear,
    /// Depthwise kernel in MLX layout (C, K, 1).
    depthwise: Array,
    conv_norm: Array,
    end: ClipLinear,
}

impl LightConv {
    fn load(w: &Scope) -> Result<Self> {
        let k = w.get("depthwise_conv1d.weight")?; // PyTorch (C, 1, K)
        let depthwise = if k.shape()[2] == 1 { k } else { k.transpose_axes(&[0, 2, 1])? };
        Ok(LightConv {
            pre: rms(w, "pre_layer_norm")?,
            start: ClipLinear::load(w, "linear_start")?,
            depthwise,
            conv_norm: rms(w, "conv_norm")?,
            end: ClipLinear::load(w, "linear_end")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.start.forward(&rms_norm(x, &self.pre)?)?;
        let width = h.shape()[2] / 2;
        let g = split_last(&h, &[width, width])?;
        let h = g[0].multiply(ops::sigmoid(&g[1])?)?;
        let h = pad_time(&h, CONV_KERNEL - 1, 0)?; // causal
        let h = ops::conv1d(&h, self.depthwise.as_dtype(h.dtype())?, 1, 0, 1, width)?;
        let h = silu(&rms_norm(&h, &self.conv_norm)?)?;
        Ok(self.end.forward(&h)?.add(x)?)
    }
}

struct Block {
    ff1: FeedForward,
    attn: Attention,
    conv: LightConv,
    ff2: FeedForward,
    norm_pre_attn: Array,
    norm_post_attn: Array,
    norm_out: Array,
}

struct ConvBlock {
    /// MLX layout (C_out, 3, 3, C_in).
    weight: Array,
    norm: Array,
}

pub struct AudioModel {
    sscp: [ConvBlock; 2],
    sscp_proj: Linear,
    blocks: Vec<Block>,
    out_proj: Linear,
    out_bias: Array,
    /// embed_audio: 1536 -> text hidden (512), after a scale-free RMSNorm.
    to_text: Linear,
    local_mask: Array,
}

impl AudioModel {
    pub fn load(tower: &Scope, embed: &Scope, hidden: i32, heads: i32) -> Result<Self> {
        let sub = tower.scope("subsample_conv_projection");
        let conv = |i: usize| -> Result<ConvBlock> {
            let w = sub.get(&format!("layer{i}.conv.weight"))?; // PyTorch (O, I, 3, 3)
            let weight = if w.shape()[1] == 3 && w.shape()[2] == 3 { w } else { w.transpose_axes(&[0, 2, 3, 1])? };
            Ok(ConvBlock { weight, norm: sub.get(&format!("layer{i}.norm.weight"))? })
        };
        let mut blocks = Vec::new();
        for i in 0.. {
            let w = tower.scope(&format!("layers.{i}"));
            if w.get("norm_out.weight").is_err() {
                break;
            }
            blocks.push(Block {
                ff1: FeedForward::load(&w.scope("feed_forward1"))?,
                attn: Attention::load(&w.scope("self_attn"), hidden, heads)?,
                conv: LightConv::load(&w.scope("lconv1d"))?,
                ff2: FeedForward::load(&w.scope("feed_forward2"))?,
                norm_pre_attn: rms(&w, "norm_pre_attn")?,
                norm_post_attn: rms(&w, "norm_post_attn")?,
                norm_out: rms(&w, "norm_out")?,
            });
        }
        if blocks.is_empty() {
            bail!("audio tower has no layers");
        }
        Ok(AudioModel {
            sscp: [conv(0)?, conv(1)?],
            sscp_proj: Linear::load(&sub, "input_proj_linear")?,
            blocks,
            out_proj: Linear::load(tower, "output_proj")?,
            out_bias: tower.get("output_proj.bias")?,
            to_text: Linear::load(embed, "embedding_projection")?,
            local_mask: local_mask(),
        })
    }

    /// `mel`: (B, T, 128) F32 log-mel; `valid`: (B, T) bool. Returns
    /// (B, T/4, text_hidden) soft tokens and their (B, T/4) validity.
    pub fn forward(&self, mel: &Array, valid: &Array) -> Result<(Array, Array)> {
        let mut x = mel.expand_dims(-1)?; // (B, T, F, 1)
        let mut valid = valid.clone();
        for blk in &self.sscp {
            let keep = valid.as_dtype(x.dtype())?.reshape(&[x.shape()[0], x.shape()[1], 1, 1])?;
            x = x.multiply(keep)?;
            let padded = ops::pad(&x, &[(0, 0), (1, 1), (1, 1), (0, 0)][..], None, None)?;
            x = ops::conv2d(&padded, blk.weight.as_dtype(x.dtype())?, (2, 2), (0, 0), (1, 1), 1)?;
            let t_out = x.shape()[1];
            valid = strided_time(&valid, t_out)?;
            x = fast::layer_norm(&x, Some(&blk.norm), None, EPS)?;
            x = ops::maximum(&x, Array::from_f32(0.0).as_dtype(x.dtype())?)?;
        }
        let (b, t, f, c) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let mut h = self.sscp_proj.forward(&x.reshape(&[b, t, f * c])?)?;

        for blk in &self.blocks {
            h = blk.ff1.forward(&h)?;
            let a = blk.attn.forward(&rms_norm(&h, &blk.norm_pre_attn)?, &valid, &self.local_mask)?;
            h = h.add(rms_norm(&a, &blk.norm_post_attn)?)?;
            h = h.multiply(valid.as_dtype(h.dtype())?.expand_dims(-1)?)?;
            h = blk.conv.forward(&h)?;
            h = blk.ff2.forward(&h)?;
            h = rms_norm(&h, &blk.norm_out)?;
        }
        let h = self.out_proj.forward(&h)?.add(&self.out_bias)?;
        let h = h.multiply(valid.as_dtype(h.dtype())?.expand_dims(-1)?)?;
        // Projected in the stream's dtype (F32); cast to BF16 at the splice.
        let soft = self.to_text.forward(&fast::rms_norm(&h, None, EPS)?)?;
        Ok((soft, valid))
    }
}

/// Every other frame of a (B, T) mask, truncated to `t_out`.
fn strided_time(valid: &Array, t_out: i32) -> Result<Array> {
    let t = valid.shape()[1];
    let idx: Vec<i32> = (0..t_out).map(|i| (2 * i).min(t - 1)).collect();
    Ok(valid.take_axis(Array::from_slice(&idx, &[t_out]), 1)?)
}
