//! Audio and video via AVFoundation.
//!
//! - [`probe`]: duration and which tracks a file has.
//! - [`stream_audio`]: the first audio track decoded, downmixed and resampled
//!   by AVFoundation to 16 kHz mono F32, delivered in chunks (a three-hour
//!   podcast is never held in memory at once).
//! - [`frames_at`]: still frames at given times, with the track's rotation
//!   applied, scaled down to a maximum side.
//!
//! Anything the OS can play works: MP3, AAC/M4A, WAV, AIFF, CAF, FLAC, ALAC;
//! MP4/MOV/M4V with H.264, HEVC, ProRes, …

use std::path::Path;
use std::ptr::NonNull;

use anyhow::{anyhow, bail, Result};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_av_foundation::{
    AVAsset, AVAssetImageGenerator, AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput,
    AVAssetTrack, AVMediaTypeAudio, AVMediaTypeVideo, AVURLAsset,
};
use objc2_core_foundation::CGSize;
use objc2_core_media::CMTime;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

use super::image::{rgb_from_cgimage, Rgb8};
use super::mel::SAMPLE_RATE;

/// What a media file contains.
#[derive(Clone, Debug)]
pub struct Probe {
    pub duration_s: f64,
    pub has_audio: bool,
    pub has_video: bool,
}

fn open(path: &Path) -> Result<Retained<AVURLAsset>> {
    let s = path.to_str().ok_or_else(|| anyhow!("non-UTF-8 path"))?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(s));
    Ok(unsafe { AVURLAsset::URLAssetWithURL_options(&url, None) })
}

#[allow(deprecated)] // the synchronous loaders are fine on a background thread
fn tracks(asset: &AVAsset, audio: bool) -> Retained<NSArray<AVAssetTrack>> {
    let kind = unsafe { if audio { AVMediaTypeAudio } else { AVMediaTypeVideo } }.unwrap();
    unsafe { asset.tracksWithMediaType(kind) }
}

/// Duration and track kinds of the file at `path`.
#[allow(deprecated)]
pub fn probe(path: &Path) -> Result<Probe> {
    let asset = open(path)?;
    let duration_s = unsafe { asset.duration().seconds() };
    let has_audio = !tracks(&asset, true).is_empty();
    let has_video = !tracks(&asset, false).is_empty();
    if !(has_audio || has_video) || !duration_s.is_finite() {
        bail!("{} has no playable audio or video", path.display());
    }
    Ok(Probe { duration_s, has_audio, has_video })
}

/// Decode the first audio track to 16 kHz mono F32, calling `on_chunk` with
/// consecutive runs of samples as AVFoundation produces them.
pub fn stream_audio(path: &Path, mut on_chunk: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
    let asset = open(path)?;
    let track = tracks(&asset, true)
        .firstObject()
        .ok_or_else(|| anyhow!("{} has no audio track", path.display()))?;
    // The constants' values are their own names (AVFAudio/AVAudioSettings.h).
    let key = |k: &str| NSString::from_str(k);
    let keys = [
        key("AVFormatIDKey"),
        key("AVSampleRateKey"),
        key("AVNumberOfChannelsKey"),
        key("AVLinearPCMBitDepthKey"),
        key("AVLinearPCMIsFloatKey"),
        key("AVLinearPCMIsNonInterleaved"),
        key("AVLinearPCMIsBigEndianKey"),
    ];
    const LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
    let values: [Retained<NSNumber>; 7] = [
        NSNumber::new_u32(LINEAR_PCM),
        NSNumber::new_f64(SAMPLE_RATE as f64),
        NSNumber::new_u32(1),
        NSNumber::new_u32(32),
        NSNumber::new_bool(true),
        NSNumber::new_bool(false),
        NSNumber::new_bool(false),
    ];
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let value_refs: Vec<&AnyObject> = values.iter().map(|v| v.as_ref()).collect();
    let settings = NSDictionary::from_slices(&key_refs, &value_refs);

    let reader = unsafe { AVAssetReader::assetReaderWithAsset_error(&asset) }
        .map_err(|e| anyhow!("cannot read {}: {}", path.display(), e.localizedDescription()))?;
    let output = unsafe {
        AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(&track, Some(&settings))
    };
    unsafe {
        reader.addOutput(&output);
        if !reader.startReading() {
            bail!("cannot start decoding {}", path.display());
        }
    }
    let mut buf: Vec<f32> = Vec::new();
    while let Some(sample) = unsafe { output.copyNextSampleBuffer() } {
        let Some(block) = (unsafe { sample.data_buffer() }) else { continue };
        let bytes = unsafe { block.data_length() };
        buf.resize(bytes / 4, 0.0);
        let dst = NonNull::new(buf.as_mut_ptr().cast()).unwrap();
        let status = unsafe { block.copy_data_bytes(0, bytes, dst) };
        if status != 0 {
            bail!("audio copy failed (OSStatus {status})");
        }
        on_chunk(&buf)?;
    }
    if unsafe { reader.status() } == AVAssetReaderStatus::Failed {
        bail!("decoding {} failed", path.display());
    }
    Ok(())
}

/// Grab frames at `times_s` (seconds), oriented per the track transform and
/// scaled so the longer side is at most `max_side`. Frames are taken at the
/// nearest decodable picture within ±`tolerance_s` (keyframe-friendly, so
/// long videos stay fast).
#[allow(deprecated)] // synchronous generator; called off the main thread
pub fn frames_at(path: &Path, times_s: &[f64], max_side: u32, tolerance_s: f64) -> Result<Vec<Rgb8>> {
    let asset = open(path)?;
    let generator = unsafe { AVAssetImageGenerator::assetImageGeneratorWithAsset(&asset) };
    let tol = unsafe { CMTime::with_seconds(tolerance_s, 600) };
    unsafe {
        generator.setAppliesPreferredTrackTransform(true);
        generator.setMaximumSize(CGSize::new(max_side as f64, max_side as f64));
        generator.setRequestedTimeToleranceBefore(tol);
        generator.setRequestedTimeToleranceAfter(tol);
    }
    times_s
        .iter()
        .map(|&t| {
            let at = unsafe { CMTime::with_seconds(t, 600) };
            let image = unsafe { generator.copyCGImageAtTime_actualTime_error(at, std::ptr::null_mut()) }
                .map_err(|e| anyhow!("frame at {t:.1}s: {}", e.localizedDescription()))?;
            rgb_from_cgimage(&image)
        })
        .collect()
}
