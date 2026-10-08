//! Log-mel features for the audio tower (the reference
//! `Gemma4AudioFeatureExtractor`): on the GPU with MLX ([`log_mel`]), or on the
//! CPU for the ONNX backend ([`log_mel_cpu`]).
//!
//! 16 kHz mono -> 20 ms Hann frames every 10 ms (semicausal: 10 ms of leading
//! silence) -> |rFFT₅₁₂| -> 128 HTK mel bins (0–8 kHz) -> ln(x + 0.001).

use anyhow::{bail, Result};
#[cfg(backend_mlx)]
use mlx_rs::{ops, Array, Dtype};

pub const SAMPLE_RATE: u32 = 16_000;
const FRAME: usize = 320; // 20 ms
const HOP: usize = 160; // 10 ms
const FFT: i32 = 512;
const MELS: usize = 128;
const MEL_FLOOR: f32 = 1e-3;
/// Inputs are zero-padded to a multiple of this many samples (reference).
const PAD_MULTIPLE: usize = 128;
/// Longest input the reference accepts (30 s); longer audio is truncated, so
/// callers window audio to at most this.
pub const MAX_SAMPLES: usize = 480_000;

/// Mel frames for `samples` of audio, and how many of them are real (frames
/// whose end falls in padding are masked out, as in the reference).
pub fn frame_counts(samples: usize) -> (usize, usize) {
    let padded = samples.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE + FRAME / 2;
    let frames = if padded > FRAME { (padded - (FRAME + 1)) / HOP + 1 } else { 0 };
    // Frame i ends at padded-index i·HOP + FRAME, i.e. original sample
    // i·HOP + FRAME/2; it is real while that sample exists.
    let valid = (0..frames).filter(|i| i * HOP + FRAME / 2 < samples).count();
    (frames, valid)
}

#[cfg(backend_mlx)]
/// Log-mel features for a batch of equal-length 16 kHz mono clips:
/// (B, frames, 128) F32, with padded frames zeroed, plus their (B, frames)
/// validity mask.
pub fn log_mel(clips: &[&[f32]]) -> Result<(Array, Array)> {
    let Some(first) = clips.first() else { bail!("no audio") };
    let n = first.len();
    if n == 0 || n > MAX_SAMPLES || clips.iter().any(|c| c.len() != n) {
        bail!("clips must be equal-length, non-empty and <= {MAX_SAMPLES} samples");
    }
    let (frames, valid) = frame_counts(n);
    let padded_len = n.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE + FRAME / 2;
    let b = clips.len();
    let mut wave = vec![0f32; b * padded_len];
    for (i, c) in clips.iter().enumerate() {
        wave[i * padded_len + FRAME / 2..i * padded_len + FRAME / 2 + n].copy_from_slice(c);
    }
    // Gather frames (B, frames, 320) via an index array, then window.
    let idx: Vec<i32> = (0..frames).flat_map(|f| (0..FRAME).map(move |k| (f * HOP + k) as i32)).collect();
    let wave = Array::from_slice(&wave, &[b as i32, padded_len as i32]);
    let framed = wave
        .take_axis(Array::from_slice(&idx, &[(frames * FRAME) as i32]), 1)?
        .reshape(&[b as i32, frames as i32, FRAME as i32])?;
    let window: Vec<f32> = (0..FRAME)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / FRAME as f32).cos())
        .collect();
    let windowed = framed.multiply(Array::from_slice(&window, &[1, 1, FRAME as i32]))?;
    let spec = ops::abs(mlx_rs::fft::rfft(&windowed, FFT, -1)?)?; // (B, frames, 257)
    let mel = spec.matmul(Array::from_slice(&mel_filters(), &[FFT / 2 + 1, MELS as i32]))?;
    let logmel = ops::log(mel.add(Array::from_f32(MEL_FLOOR))?)?;
    let mask: Vec<bool> = (0..b).flat_map(|_| (0..frames).map(|f| f < valid)).collect();
    let mask = Array::from_slice(&mask, &[b as i32, frames as i32]);
    let logmel = logmel.multiply(mask.as_dtype(Dtype::Float32)?.expand_dims(-1)?)?;
    Ok((logmel, mask))
}

/// [`log_mel`] on the CPU for one 16 kHz mono clip, for backends without
/// MLX: (frames × 128) row-major features with padded frames zeroed, the
/// frame count, and how many frames are real.
pub fn log_mel_cpu(clip: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    let n = clip.len();
    if n == 0 || n > MAX_SAMPLES {
        bail!("clip must be non-empty and <= {MAX_SAMPLES} samples");
    }
    let (frames, valid) = frame_counts(n);
    let padded_len = n.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE + FRAME / 2;
    let mut wave = vec![0f32; padded_len];
    wave[FRAME / 2..FRAME / 2 + n].copy_from_slice(clip);
    let window: Vec<f32> = (0..FRAME)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / FRAME as f32).cos())
        .collect();
    let filters = mel_filters();
    let bins = (FFT / 2 + 1) as usize;
    let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(FFT as usize);
    let (mut input, mut spectrum) = (fft.make_input_vec(), fft.make_output_vec());
    let mut mag = vec![0f32; bins];
    let mut out = vec![0f32; frames * MELS];
    for f in 0..valid {
        input.iter_mut().for_each(|x| *x = 0.0);
        for k in 0..FRAME {
            input[k] = wave[f * HOP + k] * window[k];
        }
        fft.process(&mut input, &mut spectrum).map_err(|e| anyhow::anyhow!("rfft: {e}"))?;
        for (m, c) in mag.iter_mut().zip(&spectrum) {
            *m = c.norm();
        }
        let row = &mut out[f * MELS..(f + 1) * MELS];
        for (k, &m) in mag.iter().enumerate() {
            let fk = &filters[k * MELS..(k + 1) * MELS];
            for (r, &w) in row.iter_mut().zip(fk) {
                *r += m * w;
            }
        }
        row.iter_mut().for_each(|r| *r = (*r + MEL_FLOOR).ln());
    }
    Ok((out, frames, valid))
}

/// (257, 128) triangular HTK mel filterbank over 0–8 kHz, no normalization.
fn mel_filters() -> Vec<f32> {
    let bins = (FFT / 2 + 1) as usize;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
    let pts: Vec<f64> = (0..MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64))
        .collect();
    let mut out = vec![0f32; bins * MELS];
    for k in 0..bins {
        let f = k as f64 * (SAMPLE_RATE as f64 / 2.0) / (bins - 1) as f64;
        for m in 0..MELS {
            let (l, c, r) = (pts[m], pts[m + 1], pts[m + 2]);
            let rising = (f - l) / (c - l).max(1e-10);
            let falling = (r - f) / (r - c).max(1e-10);
            out[k * MELS + m] = rising.min(falling).max(0.0) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(backend_mlx)]
    fn cpu_log_mel_matches_mlx() {
        let clip: Vec<f32> = (0..40_000).map(|i| ((i as f32) * 0.031).sin() * 0.3 + ((i as f32) * 0.0047).cos() * 0.2).collect();
        let (cpu, frames, valid) = log_mel_cpu(&clip).unwrap();
        let (gpu, mask) = log_mel(&[&clip]).unwrap();
        assert_eq!(gpu.shape(), &[1, frames as i32, MELS as i32]);
        assert_eq!(mask.as_dtype(Dtype::Int32).unwrap().sum(None).unwrap().item::<i32>() as usize, valid);
        let gpu: Vec<f32> = gpu.as_slice::<f32>().to_vec();
        // Both are F32 FFTs; log(x + 1e-3) amplifies rounding near the floor.
        let diffs: Vec<f32> = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).collect();
        let worst = diffs.iter().cloned().fold(0f32, f32::max);
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        assert!(worst < 1e-2 && mean < 1e-3, "|cpu - mlx|: max {worst}, mean {mean}");
    }

    #[test]
    fn frame_counts_match_reference() {
        // 159866 samples -> 999 frames (reference processor), all real but the
        // ones ending in the 6 samples of multiple-of-128 padding.
        let (frames, valid) = frame_counts(159_866);
        assert_eq!(frames, 999);
        assert!(valid <= frames && valid >= 997);
        assert_eq!(frame_counts(MAX_SAMPLES).0, 2999); // reference: 30 s -> 2999 frames
    }
}
