//! The Gemma-4 vision tower: image patches -> soft tokens in text-model space.
//!
//! ```text
//!   patches (16×16 RGB, [0,1]) -> 2x-0.5 -> input_proj + pos_emb[x] + pos_emb[y]
//!   -> 16 bidirectional blocks (2-D RoPE, QK-norm) -> 3×3 avg-pool × √768
//!   -> RMSNorm (no scale) -> 768→512 projection  =  one soft token per 3×3 patches
//! ```
//!
//! Images (and video frames) are fed at their exact patch grid with no padding:
//! callers batch inputs that share a grid shape, so attention never needs a
//! mask and always takes MLX's fused SDPA path (head_dim 64 is supported).

use anyhow::{bail, Result};
use mlx_rs::{fast, ops, Array, Dtype};

use crate::text::{gelu_mul, Linear, RmsNorm};
use crate::weights::Scope;

const PATCH: i32 = 16;
const POOL: i32 = 3;
const EPS: f32 = 1e-6;
const ROPE_BASE: f32 = 100.0;

/// A batch of same-shaped images, already resized so both sides are multiples
/// of 48 px: `pixels` is (B, H, W, 3) in [0, 1].
pub struct PatchGrid {
    pub pixels: Array,
    /// Patch columns / rows (W / 16, H / 16); both multiples of 3.
    pub cols: i32,
    pub rows: i32,
}

struct Block {
    qkv: Linear,
    qkv_sizes: Vec<i32>,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate_up: Linear,
    gate_up_sizes: Vec<i32>,
    down: Linear,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
}

pub struct VisionModel {
    input_proj: Linear,
    /// (2, 10240, 768): row 0 embeds x (column), row 1 embeds y (row).
    pos_table: Array,
    blocks: Vec<Block>,
    /// embed_vision: 768 -> text hidden (512), after a scale-free RMSNorm.
    to_text: Linear,
    heads: i32,
    head_dim: i32,
    hidden: i32,
}

impl VisionModel {
    pub fn load(tower: &Scope, embed: &Scope) -> Result<Self> {
        let enc = tower.scope("encoder");
        let mut blocks = Vec::new();
        for i in 0.. {
            let w = enc.scope(&format!("layers.{i}"));
            if w.get("input_layernorm.weight").is_err() {
                break;
            }
            let attn = w.scope("self_attn");
            let mlp = w.scope("mlp");
            // Vision projections are stored as `<name>.linear.weight` (the
            // checkpoint's clippable-linear wrapper; clipping is disabled here).
            let (qkv, qkv_sizes) = Linear::load_fused(
                &attn,
                &["q_proj.linear", "k_proj.linear", "v_proj.linear"],
            )?;
            let (gate_up, gate_up_sizes) =
                Linear::load_fused(&mlp, &["gate_proj.linear", "up_proj.linear"])?;
            blocks.push(Block {
                qkv,
                qkv_sizes,
                o: Linear::load(&attn, "o_proj.linear")?,
                q_norm: RmsNorm::load(&attn, "q_norm", EPS)?,
                k_norm: RmsNorm::load(&attn, "k_norm", EPS)?,
                gate_up,
                gate_up_sizes,
                down: Linear::load(&mlp, "down_proj.linear")?,
                input_norm: RmsNorm::load(&w, "input_layernorm", EPS)?,
                post_attn_norm: RmsNorm::load(&w, "post_attention_layernorm", EPS)?,
                pre_ff_norm: RmsNorm::load(&w, "pre_feedforward_layernorm", EPS)?,
                post_ff_norm: RmsNorm::load(&w, "post_feedforward_layernorm", EPS)?,
            });
        }
        if blocks.is_empty() {
            bail!("vision tower has no encoder layers");
        }
        let input_proj = Linear::load(&tower.scope("patch_embedder"), "input_proj")?;
        let hidden = input_proj.out_features();
        let head_dim = blocks[0].q_norm.width();
        Ok(VisionModel {
            input_proj,
            pos_table: tower.get("patch_embedder.position_embedding_table")?,
            blocks,
            to_text: Linear::load(embed, "embedding_projection")?,
            heads: hidden / head_dim,
            head_dim,
            hidden,
        })
    }

    /// Encode a batch of same-shaped images -> (B, rows/3 · cols/3, text_hidden)
    /// soft tokens, row-major over the pooled grid.
    pub fn forward(&self, grid: &PatchGrid) -> Result<Array> {
        let (b, rows, cols) = (grid.pixels.shape()[0], grid.rows, grid.cols);
        // (B, H, W, 3) -> (B, rows, 16, cols, 16, 3) -> (B, rows, cols, 16, 16, 3)
        // -> (B, n, 768): each patch flattened row, col, channel — the
        // processor's layout.
        let patches = grid
            .pixels
            .reshape(&[b, rows, PATCH, cols, PATCH, 3])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[b, rows * cols, PATCH * PATCH * 3])?;
        self.forward_patches(&patches, rows, cols)
    }

    /// [`forward`](Self::forward) on pre-patchified input: (B, rows·cols, 768)
    /// in [0, 1], patches row-major.
    pub fn forward_patches(&self, patches: &Array, rows: i32, cols: i32) -> Result<Array> {
        let dtype = self.input_proj.dtype();
        let (b, n) = (patches.shape()[0], rows * cols);
        if rows % POOL != 0 || cols % POOL != 0 || patches.shape()[1] != n {
            bail!("patch grid {rows}x{cols} must match the input and be a multiple of {POOL}");
        }
        let c = |v: f32| Array::from_f32(v).as_dtype(dtype);
        let patches = patches.as_dtype(dtype)?.subtract(c(0.5)?)?.multiply(c(2.0)?)?;

        // Patch (x, y) for row-major patches; the position embedding is the sum
        // of the x row and y row of the table (the reference's one-hot matmul).
        let xs: Vec<i32> = (0..n).map(|i| i % cols).collect();
        let ys: Vec<i32> = (0..n).map(|i| i / cols).collect();
        let (xs, ys) = (Array::from_slice(&xs, &[n]), Array::from_slice(&ys, &[n]));
        let table = |row: i32| self.pos_table.take_axis(Array::from_int(row), 0);
        let pos = table(0)?.take_axis(&xs, 0)?.add(table(1)?.take_axis(&ys, 0)?)?;
        let mut h = self.input_proj.forward(&patches)?.add(&pos)?;

        let (cos, sin) = rope_tables(&xs, &ys, self.head_dim, dtype)?;
        for blk in &self.blocks {
            h = self.block(blk, &h, &cos, &sin)?;
        }

        // 3×3 average pool over the patch grid (in F32), then × √hidden.
        let pooled = h
            .as_dtype(Dtype::Float32)?
            .reshape(&[b, rows / POOL, POOL, cols / POOL, POOL, self.hidden])?
            .mean_axes(&[2, 4], false)?
            .reshape(&[b, (rows / POOL) * (cols / POOL), self.hidden])?
            .as_dtype(dtype)?
            .multiply(c((self.hidden as f32).sqrt())?)?;
        let normed = fast::rms_norm(&pooled, None, EPS)?;
        self.to_text.forward(&normed)
    }

    fn block(&self, blk: &Block, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let qkv = crate::text::split_last(&blk.qkv.forward(&blk.input_norm.forward(x)?)?, &blk.qkv_sizes)?;
        let heads = |y: &Array| y.reshape(&[b, l, self.heads, self.head_dim]);
        let q = rope_2d(&blk.q_norm.forward(&heads(&qkv[0])?)?, cos, sin)?;
        let k = rope_2d(&blk.k_norm.forward(&heads(&qkv[1])?)?, cos, sin)?;
        let v = fast::rms_norm(heads(&qkv[2])?, None, EPS)?;
        let t = |y: Array| y.transpose_axes(&[0, 2, 1, 3]);
        let out = fast::scaled_dot_product_attention(t(q)?, t(k)?, t(v)?, 1.0, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        let x = x.add(blk.post_attn_norm.forward(&blk.o.forward(&out)?)?)?;

        let gu = crate::text::split_last(&blk.gate_up.forward(&blk.pre_ff_norm.forward(&x)?)?, &blk.gate_up_sizes)?;
        let m = blk.down.forward(&gelu_mul(&gu[0], &gu[1])?)?;
        Ok(x.add(blk.post_ff_norm.forward(&m)?)?)
    }
}

/// cos/sin tables for 2-D axial RoPE, shaped (1, L, 1, head_dim) to broadcast
/// over (B, L, H, D). The head splits in two halves of `head_dim/2` channels:
/// the first rotates by the patch's x, the second by its y — each half a
/// standard rotate-half RoPE with base 100.
fn rope_tables(xs: &Array, ys: &Array, head_dim: i32, dtype: Dtype) -> Result<(Array, Array)> {
    let part = head_dim / 2; // channels per spatial axis
    let half = part / 2;
    let exps: Vec<f32> = (0..half).map(|i| ROPE_BASE.powf(-(2.0 * i as f32) / part as f32)).collect();
    let inv = Array::from_slice(&exps, &[1, half]);
    let ang = |p: &Array| -> Result<Array> {
        let a = p.as_dtype(Dtype::Float32)?.expand_dims(-1)?.multiply(&inv)?; // (L, half)
        Ok(ops::concatenate(&[a.clone(), a], -1)?) // (L, part)
    };
    let angles = ops::concatenate(&[ang(xs)?, ang(ys)?], -1)?; // (L, head_dim)
    let l = angles.shape()[0];
    let shape = [1, l, 1, head_dim];
    Ok((
        ops::cos(&angles)?.as_dtype(dtype)?.reshape(&shape)?,
        ops::sin(&angles)?.as_dtype(dtype)?.reshape(&shape)?,
    ))
}

/// Apply 2-D RoPE: rotate-half *within* each spatial half of the head, so x
/// and y features never mix.
fn rope_2d(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let d = x.shape()[3];
    let q = d / 4;
    let parts = x.split_at_indices(&[q, 2 * q, 3 * q], -1)?; // x1 x2 | y1 y2
    let rotated = ops::concatenate(
        &[ops::negative(&parts[1])?, parts[0].clone(), ops::negative(&parts[3])?, parts[2].clone()],
        -1,
    )?;
    Ok(x.multiply(cos)?.add(rotated.multiply(sin)?)?)
}
