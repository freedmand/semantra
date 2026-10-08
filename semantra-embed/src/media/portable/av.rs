//! Portable audio/video decoding (the `semantra-media-portable` crate:
//! Symphonia + an ffmpeg sidecar), mirroring `media::av` on macOS.

pub use semantra_media_portable::av::{frames_at, probe, set_ffmpeg_path, stream_audio, Probe};
