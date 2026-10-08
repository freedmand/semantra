//! Pure-Rust audio path: Symphonia demux + decode, AVFoundation's downmix,
//! then [`Mono16k`] resampling. Covers MP3, AAC-LC/ALAC in MP4/M4A/MOV, FLAC,
//! WAV, AIFF, Vorbis in Ogg/WebM/MKV, and PCM, with no external binaries.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Channels, Signal};
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::{FormatOptions, FormatReader, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::mp4edit::{self, Edit, Trim};
use super::resample::Mono16k;

/// An opened file and its first decodable audio track.
pub struct Opened {
    /// MP4 edit list of the chosen track (priming/padding trim).
    pub edit: Option<Edit>,
    pub format: Box<dyn FormatReader>,
    pub track_id: u32,
    pub decoder: Box<dyn Decoder>,
    pub track: Track,
}

/// Open `path` with Symphonia and pick the first track it can decode.
pub fn open(path: &Path) -> Result<Opened> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    // Gapless: trim encoder delay/padding (LAME/iTunSMPB/edit lists) the way
    // AVFoundation does, so sample counts and alignment match.
    let fmt_opts = FormatOptions { enable_gapless: true, ..Default::default() };
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &MetadataOptions::default())?;
    let format = probed.format;
    let codecs = symphonia::default::get_codecs();
    let (track, decoder) = format
        .tracks()
        .iter()
        .filter(|t| t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some())
        .find_map(|t| codecs.make(&t.codec_params, &DecoderOptions::default()).ok().map(|d| (t.clone(), d)))
        .ok_or_else(|| anyhow!("no audio track Symphonia can decode"))?;
    let p = &track.codec_params;
    let mut edit = p.sample_rate.and_then(|r| mp4edit::audio_edit(path, track.id as usize, r));
    // MP3 without a LAME/Xing header: Symphonia then keeps the decoder's
    // 529-sample synthesis delay; AVFoundation (and every gapless player)
    // drops it.
    if p.codec == symphonia::core::codecs::CODEC_TYPE_MP3 && p.delay.is_none() && edit.is_none() {
        edit = Some(Edit { skip: 529, ..Default::default() });
    }
    Ok(Opened { edit, format, track_id: track.id, decoder, track })
}

/// Whether the track is AAC (which may be HE-AAC: Symphonia decodes only
/// the AAC-LC core of those — no SBR high band, and a different delay than
/// real decoders — so the caller checks with ffmpeg).
pub fn is_aac(o: &Opened) -> bool {
    o.track.codec_params.codec == symphonia::core::codecs::CODEC_TYPE_AAC
}

/// Exact duration of the opened track in seconds, from the container's frame
/// count, or by summing packet durations (demux only, no decoding) when the
/// header doesn't say (e.g. VBR MP3 without a Xing header).
pub fn duration(o: &mut Opened) -> Result<f64> {
    let p = &o.track.codec_params;
    let rate = p.sample_rate.ok_or_else(|| anyhow!("unknown sample rate"))? as f64;
    if let Some(Edit { lead, keep: Some(keep), .. }) = o.edit {
        return Ok((lead + keep) as f64 / rate);
    }
    if let Some(n) = p.n_frames {
        return Ok(n as f64 / rate);
    }
    let tb = p.time_base;
    let mut ts: u64 = 0;
    loop {
        match o.format.next_packet() {
            Ok(pkt) if pkt.track_id() == o.track_id => ts += pkt.dur,
            Ok(_) => {}
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::DecodeError(_)) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(match tb {
        Some(tb) => {
            let t = tb.calc_time(ts);
            t.seconds as f64 + t.frac
        }
        None => ts as f64 / rate,
    })
}

/// Decode the opened track to 16 kHz mono. Returns the number of samples
/// emitted. Corrupt packets are skipped (as players do); a decode failure
/// before any audio comes out is an error so the caller can fall back.
pub fn stream(mut o: Opened, on_chunk: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<u64> {
    let rate = o.track.codec_params.sample_rate.unwrap();
    let mut emitted = 0u64;
    let mut count = |c: &[f32]| {
        emitted += c.len() as u64;
        on_chunk(c)
    };
    let mut out = Mono16k::new(rate, &mut count)?;
    let mut planar: Option<AudioBuffer<f32>> = None;
    let mut weights = (Channels::empty(), Vec::new());
    let mut mono: Vec<f32> = Vec::new();
    let mut trim = o.edit.map(Trim::new);
    if let Some(lead) = trim.as_mut().map(Trim::take_lead).filter(|&n| n > 0) {
        out.push(&vec![0.0; lead as usize])?;
    }
    let mut decoded_any = false;
    let mut errors = 0usize;
    loop {
        let pkt = match o.format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if pkt.track_id() != o.track_id {
            continue;
        }
        let buf = match o.decoder.decode(&pkt) {
            Ok(b) => b,
            Err(SymError::DecodeError(e)) => {
                errors += 1;
                if !decoded_any && errors > 8 {
                    bail!("cannot decode audio: {e}");
                }
                continue;
            }
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        decoded_any = true;
        let spec = *buf.spec();
        if spec.rate != out.rate() {
            // Mid-stream rate changes (chained Ogg, odd MP3s) aren't worth a
            // second resampler; AVFoundation doesn't handle them either.
            bail!("sample rate changed mid-stream ({} -> {} Hz)", out.rate(), spec.rate);
        }
        downmix(&buf, &mut planar, &mut weights, &mut mono);
        match trim.as_mut() {
            Some(t) => out.push(t.apply(&mono))?,
            None => out.push(&mono)?,
        }
    }
    out.finish()?;
    if !decoded_any {
        bail!("no audio decoded");
    }
    Ok(emitted)
}

/// Per-channel weights of AVFoundation's N -> mono downmix, measured with a
/// tone in one channel at a time (parity `downmix`): front L/R and surrounds
/// -3 dB (0.7071), center 1, LFE dropped, and in 7.1 the second surround
/// pair (sides, when rears exist) -6 dB. So stereo is (L + R)/√2 — an
/// equal-power sum, not the average — which ffmpeg's default matches too.
/// Channels without a position (no channel mask) are treated as front L/R.
pub fn downmix_weights(channels: Channels) -> Vec<f32> {
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let n = channels.count();
    if n == 1 {
        return vec![1.0];
    }
    let has_rear = channels.intersects(Channels::REAR_LEFT | Channels::REAR_RIGHT);
    channels
        .iter()
        .map(|c| {
            if c == Channels::FRONT_CENTRE {
                1.0
            } else if c == Channels::LFE1 || c == Channels::LFE2 {
                0.0
            } else if (c == Channels::SIDE_LEFT || c == Channels::SIDE_RIGHT) && has_rear {
                0.5
            } else if c == Channels::FRONT_LEFT
                || c == Channels::FRONT_RIGHT
                || c == Channels::REAR_LEFT
                || c == Channels::REAR_RIGHT
                || c == Channels::SIDE_LEFT
                || c == Channels::SIDE_RIGHT
            {
                H
            } else {
                0.5
            }
        })
        .collect()
}

/// Mix all channels into `mono` with [`downmix_weights`].
fn downmix(
    buf: &AudioBufferRef,
    planar: &mut Option<AudioBuffer<f32>>,
    weights: &mut (Channels, Vec<f32>),
    mono: &mut Vec<f32>,
) {
    let frames = buf.frames();
    let cap = buf.capacity() as u64;
    let p = match planar {
        Some(p) if p.capacity() >= frames && p.spec() == buf.spec() => p,
        _ => planar.insert(AudioBuffer::new(cap, *buf.spec())),
    };
    buf.convert(p);
    let channels = p.spec().channels;
    if weights.0 != channels || weights.1.is_empty() {
        *weights = (channels, downmix_weights(channels));
    }
    let w = &weights.1;
    mono.clear();
    mono.extend(p.chan(0)[..frames].iter().map(|s| s * w[0]));
    for (c, &wc) in w.iter().enumerate().skip(1) {
        if wc == 0.0 {
            continue;
        }
        for (m, s) in mono.iter_mut().zip(&p.chan(c)[..frames]) {
            *m += s * wc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avfoundation_downmix_weights() {
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(downmix_weights(Channels::FRONT_LEFT), vec![1.0]);
        assert_eq!(downmix_weights(Channels::FRONT_LEFT | Channels::FRONT_RIGHT), vec![h, h]);
        let s71 = Channels::FRONT_LEFT
            | Channels::FRONT_RIGHT
            | Channels::FRONT_CENTRE
            | Channels::LFE1
            | Channels::REAR_LEFT
            | Channels::REAR_RIGHT
            | Channels::SIDE_LEFT
            | Channels::SIDE_RIGHT;
        assert_eq!(downmix_weights(s71), vec![h, h, 1.0, 0.0, h, h, 0.5, 0.5]);
    }
}
