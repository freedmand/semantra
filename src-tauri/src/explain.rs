//! Wire types for per-token similarity attribution ("why did this chunk match?").
//!
//! The heavy lifting — the exact linear decomposition of a cosine similarity
//! into per-token contributions — lives in the embedding crate
//! (`explain_similarity_batch`). This module only adapts that crate's plain
//! structs into camelCase-serialized DTOs the frontend can consume, keeping the
//! crate free of any wire-format concerns.
//!
//! Everything here is *raw, un-presentational* data: signed per-token scores and
//! byte offsets into the original chunk text. How those map to words/sentences,
//! how they are normalized, and how they are colored is decided entirely on the
//! frontend, so display choices can change without touching Rust.

use leaf_ir_candle_test::{Explanation as CoreExplanation, TokenContribution as CoreToken};

/// One token's signed share of a cosine similarity, with byte offsets into the
/// original chunk text. Mirrors the frontend `TokenSpan`.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenSpan {
    /// Byte offset (inclusive start) of this token in the chunk text.
    pub start: usize,
    /// Byte offset (exclusive end) of this token in the chunk text.
    pub end: usize,
    /// WordPiece sub-tokens of one whitespace word share a `word_id`; `null` for
    /// special tokens. Lets the frontend aggregate pieces back to whole words.
    pub word_id: Option<u32>,
    /// Exact additive share of the cosine similarity (signed).
    pub score: f32,
    /// True for `[CLS]`/`[SEP]`: real contribution mass, but not chunk text.
    pub special: bool,
}

/// The full additive decomposition of one query↔chunk similarity. Mirrors the
/// frontend `Explanation`; `total == bias + Σ tokens[i].score`.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Explanation {
    pub tokens: Vec<TokenSpan>,
    /// Constant share attributable to no single token.
    pub bias: f32,
    /// The cosine similarity this explanation decomposes.
    pub total: f32,
}

impl From<CoreToken> for TokenSpan {
    fn from(t: CoreToken) -> Self {
        TokenSpan {
            start: t.start,
            end: t.end,
            word_id: t.word_id,
            score: t.score,
            special: t.special,
        }
    }
}

impl From<CoreExplanation> for Explanation {
    fn from(e: CoreExplanation) -> Self {
        Explanation {
            tokens: e.tokens.into_iter().map(Into::into).collect(),
            bias: e.bias,
            total: e.total,
        }
    }
}
