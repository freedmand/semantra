//! Portable audio/video decoding (placeholder until the portable decoder
//! lands; mirrors `media::av` on macOS).

use std::path::Path;

use anyhow::{bail, Result};

use crate::media::image::Rgb8;

/// What a media file contains.
#[derive(Clone, Debug)]
pub struct Probe {
    pub duration_s: f64,
    pub has_audio: bool,
    pub has_video: bool,
}

pub fn probe(path: &Path) -> Result<Probe> {
    bail!("audio/video is not supported on this platform yet ({})", path.display())
}

pub fn stream_audio(path: &Path, _on_chunk: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
    bail!("audio is not supported on this platform yet ({})", path.display())
}

pub fn frames_at(path: &Path, _times_s: &[f64], _max_side: u32, _tolerance_s: f64) -> Result<Vec<Rgb8>> {
    bail!("video frames are not supported on this platform yet ({})", path.display())
}
