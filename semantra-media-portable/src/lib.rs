//! Portable media decoding for Semantra on Windows and Linux: drop-in
//! replacements for semantra-embed's macOS `media::image` (ImageIO) and
//! `media::av` (AVFoundation) with the same public API and semantics.
//!
//! - [`image`]: stills via the `image` crate — first frame, EXIF orientation,
//!   ICC -> sRGB, alpha over white, downscale-only cap on the longer side.
//! - [`av`]: audio via Symphonia + Rubato (16 kHz mono, streamed), video
//!   frames and exotic audio codecs via an ffmpeg CLI sidecar.
//!
//! Every AVFoundation/ImageIO behavior mirrored here was measured, not
//! assumed: the macOS-only `parity` workspace member compares both decoders
//! on real files (`cargo run --release -p parity -- all`) and holds the
//! probes behind each constant (`filters`, `downmix`, `phase`, `response`,
//! `color`).

pub mod av;
pub mod image;
