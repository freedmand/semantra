//! `config.json` for EmbeddingGemma 2, deserialized just far enough to build
//! the towers. Field names mirror the Hugging Face config so the file loads
//! unmodified; anything the forward pass doesn't read is ignored.

use std::collections::HashMap;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    pub audio_config: AudioConfig,
    pub image_token_id: u32,
    pub audio_token_id: u32,
    pub video_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_index: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AudioConfig {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub hidden_size_per_layer_input: i32,
    pub embedding_dim: i32,
    pub rms_norm_eps: f32,
    pub sliding_window: i32,
    pub pad_token_id: u32,
    pub layer_types: Vec<String>,
    /// Per-layer overrides keyed by zero-padded layer index ("05"): the global
    /// layers use a wider head (512) and a single KV head.
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerOverride>,
    pub rope_parameters: HashMap<String, RopeParams>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct LayerOverride {
    pub head_dim: Option<i32>,
    pub num_attention_heads: Option<i32>,
    pub num_key_value_heads: Option<i32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f32,
}

impl TextConfig {
    pub fn is_global(&self, layer: usize) -> bool {
        self.layer_types[layer] == "full_attention"
    }

    fn overrides(&self, layer: usize) -> LayerOverride {
        self.per_layer_config
            .get(&format!("{layer:02}"))
            .cloned()
            .unwrap_or_default()
    }

    /// `(num_heads, num_kv_heads, head_dim)` for `layer`, with overrides applied.
    pub fn attn_shape(&self, layer: usize) -> (i32, i32, i32) {
        let o = self.overrides(layer);
        (
            o.num_attention_heads.unwrap_or(self.num_attention_heads),
            o.num_key_value_heads.unwrap_or(self.num_key_value_heads),
            o.head_dim.unwrap_or(self.head_dim),
        )
    }

    pub fn rope_theta(&self, layer: usize) -> f32 {
        self.rope_parameters[&self.layer_types[layer]].rope_theta
    }
}
