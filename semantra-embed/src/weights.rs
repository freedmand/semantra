//! Safetensors weight map with prefix-scoped lookups.
//!
//! The checkpoint is Google's original `model.safetensors` (BF16), loaded as-is:
//! keys are only stripped of their leading `model.` so they read
//! `language_model.layers.3.mlp.up_proj.weight`. MLX memory-maps the file, so
//! towers we never touch (e.g. audio, for a text-only session) cost no RAM.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_rs::Array;

pub struct Weights {
    map: HashMap<String, Array>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = Array::load_safetensors(path)
            .map_err(|e| anyhow!("load {}: {e}", path.display()))?;
        let map = raw
            .into_iter()
            .map(|(k, v)| (k.strip_prefix("model.").map(str::to_owned).unwrap_or(k), v))
            .collect();
        Ok(Weights { map })
    }

    pub fn get(&self, key: &str) -> Result<Array> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow!("missing weight {key:?}"))
    }

    /// A view that prepends `prefix.` to every lookup.
    pub fn scope<'a>(&'a self, prefix: &str) -> Scope<'a> {
        Scope {
            weights: self,
            prefix: prefix.to_owned(),
        }
    }
}

pub struct Scope<'a> {
    weights: &'a Weights,
    prefix: String,
}

impl<'a> Scope<'a> {
    pub fn get(&self, key: &str) -> Result<Array> {
        self.weights.get(&format!("{}.{key}", self.prefix))
    }

    pub fn scope(&self, prefix: &str) -> Scope<'a> {
        Scope {
            weights: self.weights,
            prefix: format!("{}.{prefix}", self.prefix),
        }
    }
}
