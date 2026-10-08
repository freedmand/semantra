//! Audio and video, portably (Windows/Linux stand-ins for AVFoundation).
//!
//! - [`probe`]: duration and which tracks a file has.
//! - [`stream_audio`]: the first audio track decoded, downmixed (as
//!   AVFoundation does: (L + R)/√2 for stereo) and resampled to 16 kHz mono
//!   F32, delivered in ~1 s chunks (a three-hour podcast is never held in
//!   memory at once).
//! - [`frames_at`]: still frames at given times, with the track's rotation
//!   applied, scaled down to a maximum side.
//!
//! Audio decodes in-process with Symphonia (MP3, AAC-LC, ALAC, FLAC, WAV,
//! AIFF, Vorbis, PCM in MP4/M4A/MOV/MKV/WebM/Ogg/…) and falls back to the
//! ffmpeg sidecar for everything else (Opus, HE-AAC, AC-3, WMA, …). Video
//! frames always come from a [`VideoBackend`] — by default the ffmpeg CLI
//! sidecar ([`ffmpeg`]); an OS-native backend (Media Foundation, GStreamer)
//! can be swapped in with [`set_video_backend`].

pub mod ffmpeg;
mod mp4edit;
mod native;
mod resample;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{bail, Result};

pub use ffmpeg::set_ffmpeg_path;

use crate::image::Rgb8;

/// Output rate of [`stream_audio`] (the audio tower's rate; semantra-embed's
/// `mel::SAMPLE_RATE`).
pub const SAMPLE_RATE: u32 = 16_000;

/// What a media file contains.
#[derive(Clone, Debug)]
pub struct Probe {
    pub duration_s: f64,
    pub has_audio: bool,
    pub has_video: bool,
}

/// A decoder for the video-capable half of the API. The audio path only uses
/// it as a fallback for codecs Symphonia can't decode.
pub trait VideoBackend: Send + Sync {
    fn probe(&self, path: &Path) -> Result<Probe>;
    fn frames_at(&self, path: &Path, times_s: &[f64], max_side: u32, tolerance_s: f64) -> Result<Vec<Rgb8>>;
    fn stream_audio(&self, path: &Path, on_chunk: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()>;
}

/// The ffmpeg CLI sidecar ([`ffmpeg`]).
pub struct FfmpegSidecar;

impl VideoBackend for FfmpegSidecar {
    fn probe(&self, path: &Path) -> Result<Probe> {
        ffmpeg::probe(path)
    }
    fn frames_at(&self, path: &Path, times_s: &[f64], max_side: u32, tolerance_s: f64) -> Result<Vec<Rgb8>> {
        ffmpeg::frames_at(path, times_s, max_side, tolerance_s)
    }
    fn stream_audio(&self, path: &Path, on_chunk: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        ffmpeg::stream_audio(path, on_chunk)
    }
}

static BACKEND: RwLock<Option<Arc<dyn VideoBackend>>> = RwLock::new(None);

/// Replace the video backend (default: [`FfmpegSidecar`]).
pub fn set_video_backend(backend: Arc<dyn VideoBackend>) {
    *BACKEND.write().unwrap() = Some(backend);
}

fn backend() -> Arc<dyn VideoBackend> {
    BACKEND.read().unwrap().clone().unwrap_or_else(|| Arc::new(FfmpegSidecar))
}

/// Containers that may hold video (ISO-BMFF: MP4/MOV/M4V/3GP, and
/// Matroska/WebM), sniffed from the magic bytes. Symphonia only exposes their
/// audio tracks, so whether there is video — and the container duration —
/// must come from the video backend.
fn maybe_video(path: &Path) -> bool {
    let mut head = [0u8; 12];
    let n = File::open(path).and_then(|mut f| f.read(&mut head)).unwrap_or(0);
    let head = &head[..n];
    let isobmff = head.len() >= 8 && &head[4..8] == b"ftyp";
    let matroska = head.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]);
    let riff_avi = head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"AVI ";
    let other_video = matches!(
        path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
        Some("avi" | "wmv" | "asf" | "flv" | "ts" | "mts" | "m2ts" | "mpg" | "mpeg" | "vob" | "ogv")
    );
    isobmff || matroska || riff_avi || other_video
}

/// Duration and track kinds of the file at `path`.
///
/// Pure audio containers are probed in-process (exact sample-count
/// durations, as AVFoundation reports); anything that may carry video goes to
/// the video backend.
pub fn probe(path: &Path) -> Result<Probe> {
    if !maybe_video(path) {
        if let Ok(mut o) = native::open(path) {
            let duration_s = native::duration(&mut o)?;
            if !duration_s.is_finite() {
                bail!("{} has no playable audio or video", path.display());
            }
            return Ok(Probe { duration_s, has_audio: true, has_video: false });
        }
    }
    match backend().probe(path) {
        Ok(p) => Ok(p),
        // No sidecar: still serve audio-only MP4/M4A/MKV via Symphonia.
        Err(e) => match native::open(path) {
            Ok(mut o) => Ok(Probe { duration_s: native::duration(&mut o)?, has_audio: true, has_video: false }),
            Err(_) => Err(e),
        },
    }
}

/// Decode the first audio track to 16 kHz mono F32, calling `on_chunk` with
/// consecutive runs of samples.
pub fn stream_audio(path: &Path, mut on_chunk: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
    let mut delivered = false;
    let native_err = match native::open(path) {
        // HE-AAC (explicit or implicit SBR signaling; ffmpeg detects both).
        Ok(o) if native::is_aac(&o) && ffmpeg::info_cached(path).is_ok_and(|i| i.he_aac) => {
            anyhow::anyhow!("HE-AAC: Symphonia has no SBR")
        }
        Ok(o) => {
            let mut wrapped = |c: &[f32]| {
                delivered = true;
                on_chunk(c)
            };
            match native::stream(o, &mut wrapped) {
                Ok(_) => return Ok(()),
                // Audio already reached the caller: falling back would
                // duplicate it.
                Err(e) if delivered => return Err(e),
                Err(e) => e,
            }
        }
        Err(e) => e,
    };
    backend().stream_audio(path, &mut on_chunk).map_err(|e| e.context(format!("Symphonia: {native_err:#}")))
}

/// Grab frames at `times_s` (seconds), oriented per the track transform and
/// scaled so the longer side is at most `max_side`. Frames are the picture
/// displayed at each time, or a keyframe within ±`tolerance_s` when that is
/// cheaper (keyframe-friendly, so long videos stay fast).
pub fn frames_at(path: &Path, times_s: &[f64], max_side: u32, tolerance_s: f64) -> Result<Vec<Rgb8>> {
    backend().frames_at(path, times_s, max_side, tolerance_s)
}
