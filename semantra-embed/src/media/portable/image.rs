//! Portable still-image decode/encode (placeholder until the portable decoder
//! lands; mirrors `media::image`'s macOS functions).

use std::path::Path;

use anyhow::{bail, Result};

use crate::media::image::{Rgb8, DECODE_MAX_SIDE};

pub fn decode(path: &Path) -> Result<Rgb8> {
    decode_max(path, DECODE_MAX_SIDE)
}

pub fn decode_max(path: &Path, _max_side: u32) -> Result<Rgb8> {
    bail!("image decoding is not implemented on this platform yet ({})", path.display())
}

pub fn encode_jpeg(_img: &Rgb8, _quality: f64) -> Result<Vec<u8>> {
    bail!("JPEG encoding is not implemented on this platform yet")
}
