//! The Gemma-4 text backbone of EmbeddingGemma 2 (bidirectional encoder).
//!
//! Every modality ends here: text tokens are looked up in `embed_tokens`, media
//! tokens are replaced by the vision/audio towers' soft tokens, and this stack
//! turns the combined sequence into per-token 768-d vectors. The sentence
//! embedding is the masked mean of those vectors, L2-normalized.
//!
//! Layout per layer (24 total; every 6th is global, the rest a ±512 sliding
//! window — both *bidirectional*):
//!
//! ```text
//!   x += post_attn_norm(attn(input_norm(x)))
//!   x += post_ff_norm(mlp(pre_ff_norm(x)))
//!   x += post_ple_norm(W_p · (gelu(W_g · x) ⊙ ple_i))   // per-layer input
//!   x *= layer_scalar
//! ```
//!
//! Note the projection to 768-d is applied **per token, before pooling** and
//! has no bias. That keeps the post-transformer pipeline linear, so a cosine
//! score decomposes exactly into per-token shares (see `explain`).

use anyhow::Result;
use std::cell::RefCell;

use mlx_rs::error::Exception;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{fast, ops, Array, Dtype};

use crate::config::TextConfig;
use crate::weights::Scope;

/// Weight-only linear layer (`y = x · Wᵀ`); none of the towers use biases.
pub(crate) struct Linear(Array);

impl Linear {
    pub(crate) fn load(w: &Scope, name: &str) -> Result<Self> {
        Ok(Linear(w.get(&format!("{name}.weight"))?))
    }

    /// Stack several projections that share an input into one weight, so they
    /// run as a single larger GEMM (better Neural Accelerator utilization than
    /// three/two small ones). Split the output with [`split_last`].
    pub(crate) fn load_fused(w: &Scope, names: &[&str]) -> Result<(Self, Vec<i32>)> {
        let parts = names
            .iter()
            .map(|n| w.get(&format!("{n}.weight")))
            .collect::<Result<Vec<_>>>()?;
        let sizes = parts.iter().map(|p| p.shape()[0]).collect();
        let fused = ops::concatenate(&parts, 0)?;
        fused.eval()?; // materialize now; the mmapped originals can then drop
        Ok((Linear(fused), sizes))
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        Ok(x.matmul(self.0.t())?)
    }

    pub(crate) fn out_features(&self) -> i32 {
        self.0.shape()[0]
    }

    pub(crate) fn dtype(&self) -> Dtype {
        self.0.dtype()
    }
}

/// Split the last axis of a fused projection's output into its parts.
pub(crate) fn split_last(y: &Array, sizes: &[i32]) -> Result<Vec<Array>> {
    let mut cuts = Vec::with_capacity(sizes.len() - 1);
    let mut acc = 0;
    for s in &sizes[..sizes.len() - 1] {
        acc += s;
        cuts.push(acc);
    }
    Ok(y.split_at_indices(&cuts, -1)?)
}

pub(crate) struct RmsNorm {
    weight: Option<Array>,
    eps: f32,
}

impl RmsNorm {
    pub(crate) fn load(w: &Scope, name: &str, eps: f32) -> Result<Self> {
        Ok(RmsNorm {
            weight: Some(w.get(&format!("{name}.weight"))?),
            eps,
        })
    }

    /// Scale-free RMSNorm (the value-norm in attention has no weight).
    fn unscaled(eps: f32) -> Self {
        RmsNorm { weight: None, eps }
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::rms_norm(x, self.weight.as_ref(), self.eps)?)
    }

    /// Normalized width (the weight's length; 0 for a scale-free norm).
    pub(crate) fn width(&self) -> i32 {
        self.weight.as_ref().map_or(0, |w| w.shape()[0])
    }
}

struct Attention {
    /// Fused [q; k; v] projection; `qkv_sizes` splits its output.
    qkv_proj: Linear,
    qkv_sizes: Vec<i32>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
}

impl Attention {
    fn load(w: &Scope, cfg: &TextConfig, layer: usize) -> Result<Self> {
        let (heads, kv_heads, head_dim) = cfg.attn_shape(layer);
        let eps = cfg.rms_norm_eps;
        let (qkv_proj, qkv_sizes) = Linear::load_fused(w, &["q_proj", "k_proj", "v_proj"])?;
        Ok(Attention {
            qkv_proj,
            qkv_sizes,
            o_proj: Linear::load(w, "o_proj")?,
            q_norm: RmsNorm::load(w, "q_norm", eps)?,
            k_norm: RmsNorm::load(w, "k_norm", eps)?,
            v_norm: RmsNorm::unscaled(eps),
            heads,
            kv_heads,
            head_dim,
            rope_theta: cfg.rope_theta(layer),
        })
    }

    fn forward(&self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let split = |y: Array, h: i32| y.reshape(&[b, l, h, self.head_dim]);
        let qkv = split_last(&self.qkv_proj.forward(x)?, &self.qkv_sizes)?;
        let q = self.q_norm.forward(&split(qkv[0].clone(), self.heads)?)?;
        let k = self.k_norm.forward(&split(qkv[1].clone(), self.kv_heads)?)?;
        let v = self.v_norm.forward(&split(qkv[2].clone(), self.kv_heads)?)?;

        // (B, L, H, D) -> (B, H, L, D); RoPE rotates along the L axis.
        let to_heads = |y: Array| y.transpose_axes(&[0, 2, 1, 3]);
        let rope = |y: Array| fast::rope(y, self.head_dim, false, self.rope_theta, 1.0, 0, None);
        let q = rope(to_heads(q)?)?;
        let k = rope(to_heads(k)?)?;
        let v = to_heads(v)?;

        // Q/K are RMS-normalized, so the attention scale is 1 (not 1/√d).
        let out = match mask {
            Some(m) => fast::scaled_dot_product_attention(&q, &k, &v, 1.0, m, None)?,
            None => fast::scaled_dot_product_attention(&q, &k, &v, 1.0, None, None)?,
        };
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        self.o_proj.forward(&out)
    }
}

/// Tanh-approximated GELU that **preserves the input dtype**.
///
/// `mlx_rs::nn::gelu_approximate` multiplies by F32 0-d constants, which (unlike
/// Python's weakly-typed float literals) promote BF16 activations to F32 — the
/// whole residual stream then silently runs in F32 at ~2.5x the cost. Casting
/// the constants to the activation dtype keeps everything BF16.
fn gelu_tanh(x: &Array) -> std::result::Result<Array, Exception> {
    let c = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
    let inner = x.add(c(0.044_715)?.multiply(x.multiply(x)?.multiply(x)?)?)?;
    let t = ops::tanh(c((2.0 / std::f32::consts::PI).sqrt())?.multiply(inner)?)?;
    c(0.5)?.multiply(x)?.multiply(c(1.0)?.add(t)?)
}

type Fused2 = Box<dyn for<'a> FnMut((&'a Array, &'a Array)) -> std::result::Result<Array, Exception>>;

thread_local! {
    /// `gelu(a) * b` compiled into one Metal kernel (shapeless, so one trace
    /// serves every batch shape). Both the MLP and PLE gates use it; unfused
    /// it is ~8 memory-bound passes over a (B, L, 2048) tensor.
    ///
    /// Held for the thread's lifetime on purpose: every `compile()` call in
    /// mlx-rs mints a new cache id, so compiling per call re-traces each time
    /// and grows MLX's compile cache without bound. (The model lives on one
    /// inference thread, so a thread-local is exactly one trace.)
    static GELU_MUL: RefCell<Fused2> = RefCell::new(Box::new(compile(gelu_mul_raw, true)));
}

fn gelu_mul_raw((a, b): (&Array, &Array)) -> std::result::Result<Array, Exception> {
    gelu_tanh(a)?.multiply(b)
}

pub(crate) fn gelu_mul(a: &Array, b: &Array) -> Result<Array> {
    Ok(GELU_MUL.with(|f| (f.borrow_mut())((a, b)))?)
}

struct Mlp {
    /// Fused [gate; up] projection (2 × intermediate wide).
    gate_up: Linear,
    gate_up_sizes: Vec<i32>,
    down: Linear,
}

impl Mlp {
    fn load(w: &Scope) -> Result<Self> {
        let (gate_up, gate_up_sizes) = Linear::load_fused(w, &["gate_proj", "up_proj"])?;
        Ok(Mlp {
            gate_up,
            gate_up_sizes,
            down: Linear::load(w, "down_proj")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gu = split_last(&self.gate_up.forward(x)?, &self.gate_up_sizes)?;
        self.down.forward(&gelu_mul(&gu[0], &gu[1])?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    ple_gate: Linear,
    ple_proj: Linear,
    ple_norm: RmsNorm,
    scalar: Array,
    global: bool,
}

impl Layer {
    fn load(w: &Scope, cfg: &TextConfig, layer: usize) -> Result<Self> {
        let eps = cfg.rms_norm_eps;
        let ple = w.scope("ple_block");
        Ok(Layer {
            attn: Attention::load(&w.scope("self_attn"), cfg, layer)?,
            mlp: Mlp::load(&w.scope("mlp"))?,
            input_norm: RmsNorm::load(w, "input_layernorm", eps)?,
            post_attn_norm: RmsNorm::load(w, "post_attention_layernorm", eps)?,
            pre_ff_norm: RmsNorm::load(w, "pre_feedforward_layernorm", eps)?,
            post_ff_norm: RmsNorm::load(w, "post_feedforward_layernorm", eps)?,
            ple_gate: Linear::load(&ple, "per_layer_input_gate")?,
            ple_proj: Linear::load(&ple, "per_layer_projection")?,
            ple_norm: RmsNorm::load(&ple, "post_per_layer_input_norm", eps)?,
            scalar: w.get("layer_scalar")?,
            global: cfg.is_global(layer),
        })
    }

    fn forward(&self, x: &Array, mask: Option<&Array>, ple_input: &Array) -> Result<Array> {
        let a = self.attn.forward(&self.input_norm.forward(x)?, mask)?;
        let x = x.add(self.post_attn_norm.forward(&a)?)?;
        let m = self.mlp.forward(&self.pre_ff_norm.forward(&x)?)?;
        let x = x.add(self.post_ff_norm.forward(&m)?)?;
        let gate = gelu_mul(&self.ple_gate.forward(&x)?, ple_input)?;
        let x = x.add(self.ple_norm.forward(&self.ple_proj.forward(&gate)?)?)?;
        Ok(x.multiply(&self.scalar)?)
    }
}

pub struct TextModel {
    embed_tokens: Array,
    ple_proj: Linear,
    ple_norm: RmsNorm,
    layers: Vec<Layer>,
    norm: RmsNorm,
    projection: Linear,
    cfg: TextConfig,
}

impl TextModel {
    pub fn load(w: &Scope, cfg: &TextConfig) -> Result<Self> {
        let ple = w.scope("ple");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(&w.scope(&format!("layers.{i}")), cfg, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(TextModel {
            embed_tokens: w.get("embed_tokens.weight")?,
            ple_proj: Linear::load(&ple, "per_layer_model_projection")?,
            ple_norm: RmsNorm::load(&ple, "per_layer_projection_norm", cfg.rms_norm_eps)?,
            layers,
            norm: RmsNorm::load(w, "norm", cfg.rms_norm_eps)?,
            projection: Linear::load(w, "embedding_projection")?,
            cfg: cfg.clone(),
        })
    }

    pub fn dtype(&self) -> Dtype {
        self.embed_tokens.dtype()
    }

    /// Scaled token embeddings for `ids` (B, L) -> (B, L, hidden). Gemma scales
    /// by √hidden, with the scalar rounded to the activation dtype first (BF16:
    /// √512 = 22.627 -> 22.625) exactly as the reference does.
    pub fn embed(&self, ids: &Array) -> Result<Array> {
        let scale = Array::from_f32((self.cfg.hidden_size as f32).sqrt()).as_dtype(self.dtype())?;
        Ok(self.embed_tokens.take_axis(ids, 0)?.multiply(&scale)?)
    }

    /// Run the encoder over already-embedded inputs.
    ///
    /// - `x`: (B, L, hidden) input embeddings (text and/or media soft tokens).
    /// - `valid`: (B, L) bool, false at padding. `None` means no padding.
    ///
    /// Returns per-token (B, L, embedding_dim) vectors in the model dtype; the
    /// caller pools them.
    pub fn forward(&self, x: &Array, valid: Option<&Array>) -> Result<Array> {
        let (b, l, hidden) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let n_layers = self.layers.len() as i32;
        let width = self.cfg.hidden_size_per_layer_input;

        // Key-padding mask (B, 1, 1, L): padded keys never receive attention.
        let full_mask = valid
            .map(|v| v.reshape(&[b, 1, 1, l]))
            .transpose()?;
        // Sliding layers see keys within ±window. For L <= window + 1 that band
        // covers the whole sequence, so they share the full mask — the common
        // case for text chunks, letting every layer take the fused SDPA path.
        let window = self.cfg.sliding_window;
        let sliding_mask = if l > window + 1 {
            let pos = Array::from_iter(0..l, &[l]);
            let dist = ops::abs(pos.reshape(&[l, 1])?.subtract(pos.reshape(&[1, l])?)?)?;
            let band = dist.le(Array::from_int(window))?.reshape(&[1, 1, l, l])?;
            Some(match &full_mask {
                Some(m) => ops::logical_and(m, &band)?,
                None => band,
            })
        } else {
            full_mask.clone()
        };

        // Per-layer inputs: one 512-d vector per (token, layer), computed once
        // from the input embeddings and fed to each layer's PLE block.
        let ple = self
            .ple_proj
            .forward(x)?
            .multiply(Array::from_f32((hidden as f32).powf(-0.5)).as_dtype(x.dtype())?)?
            .reshape(&[b, l, n_layers, width])?;
        let ple = self.ple_norm.forward(&ple)?;

        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.global { &full_mask } else { &sliding_mask };
            let ple_i = ple.take_axis(Array::from_int(i as i32), 2)?;
            h = layer.forward(&h, mask.as_ref(), &ple_i)?;
        }
        self.projection.forward(&self.norm.forward(&h)?)
    }
}
