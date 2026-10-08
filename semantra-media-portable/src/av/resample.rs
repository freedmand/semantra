//! Streaming mono -> 16 kHz resampler that reproduces AVFoundation's
//! converter: the same filter shape (see [`make`]), the same alignment (group
//! delay trimmed to the sub-sample, plus AVFoundation's own small lead), and
//! `round(N · 16000 / R)` output samples for N inputs at rate R (the tail is
//! flushed through the filter).

use anyhow::Result;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use super::SAMPLE_RATE;

/// Input frames per resampler call. Larger = fewer calls, more latency;
/// latency is irrelevant here.
const CHUNK_IN: usize = 4096;
/// Samples handed to the caller per `on_chunk` (≈ 1 s), so callers aren't
/// flooded with tiny slices.
const EMIT: usize = 16_000;

/// Accepts mono samples at `rate`, emits 16 kHz mono in ~1 s chunks.
pub struct Mono16k<'a> {
    rate: u32,
    resampler: Option<SincFixedIn<f32>>,
    pending_in: Vec<f32>,
    out: Vec<Vec<f32>>,
    ready: Vec<f32>,
    delay_left: usize,
    in_total: u64,
    out_total: u64,
    on_chunk: &'a mut dyn FnMut(&[f32]) -> Result<()>,
}

/// AVFoundation's converter is not exactly delay-free: measured with 1 kHz
/// tones (parity `phase`), its output leads the input by a whole number of
/// *input* samples that depends on the rate. Reproducing it is free and
/// lifts correlation with macOS output from ~0.990 to ~0.997 on speech at
/// 44.1/48 kHz (a sub-sample skew is inaudible, and irrelevant to the mel
/// front end, but it shows up in sample-level comparisons). Unlisted rates
/// are aligned exactly.
fn avfoundation_shift(rate: u32) -> i32 {
    match rate {
        11_025 => 1,
        24_000 | 32_000 | 44_100 => -1,
        48_000 => -2,
        88_200 => -4,
        96_000 => -5,
        _ => 0,
    }
}

/// Group delay (output samples, fractional) of the resampler `new` builds
/// for `rate`: fit the phase of a 1 kHz tone pushed through it. The integer
/// part comes from Rubato's `output_delay`, so the fit only has to resolve
/// ±half a tone period (8 samples).
/// AVFoundation's sample-rate converter, reproduced: a 32-tap
/// Blackman-windowed sinc with its cutoff at 0.9 × the lower Nyquist. Fitted
/// to AVFoundation's measured magnitude response (parity `response`: tones
/// 1–9 kHz at 44.1 and 48 kHz, within 0.2–0.4 dB RMS — it rolls off gently,
/// -1.7 dB at 6 kHz, -10 dB at 8 kHz, and lets a little alias through), it
/// makes the output sample-for-sample equal (correlation 1.0000 vs 0.9993
/// with a brick-wall resampler) and audio embeddings match to cos 0.9999.
fn make(rate: u32) -> Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 32,
        f_cutoff: 0.9,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::Blackman,
    };
    Ok(SincFixedIn::<f32>::new(SAMPLE_RATE as f64 / rate as f64, 1.0, params, CHUNK_IN, 1)?)
}

/// Group delay (output samples, fractional) of the resampler [`make`] builds
/// for `rate`, from the phase of tones pushed through it: 100 Hz (period 160
/// samples) resolves the whole-sample part near Rubato's nominal
/// `output_delay`, 1 kHz refines the fraction.
fn measure_delay(rate: u32) -> Result<f64> {
    let nominal = make(rate)?.output_delay() as f64;
    let coarse = tone_delay(rate, 100.0, nominal)?;
    tone_delay(rate, 1000.0, coarse)
}

/// Delay of an `f` Hz tone through a fresh resampler, unwrapped to the
/// period nearest `near`.
fn tone_delay(rate: u32, f: f64, near: f64) -> Result<f64> {
    let mut r = make(rate)?;
    let mut out = Vec::new();
    let mut buf = vec![vec![0f32; r.output_frames_max()]];
    let chunks = (rate as usize / CHUNK_IN).max(4) * 2; // ~2 s
    for c in 0..chunks {
        let chunk: Vec<f32> = (0..CHUNK_IN)
            .map(|i| ((c * CHUNK_IN + i) as f64 * 2.0 * std::f64::consts::PI * f / rate as f64).sin() as f32)
            .collect();
        let (_, n) = r.process_into_buffer(&[&chunk], &mut buf, None)?;
        out.extend_from_slice(&buf[0][..n]);
    }
    // out[i] ≈ sin(w (i - d)) = sin(wi)cos(wd) - cos(wi)sin(wd), fitted over
    // whole periods after the warm-up.
    let w = 2.0 * std::f64::consts::PI * f / SAMPLE_RATE as f64;
    let period = SAMPLE_RATE as f64 / f;
    let start = near.max(0.0) as usize + 2 * period as usize;
    let len = ((out.len().saturating_sub(start)) as f64 / period).floor() as usize * period as usize;
    let (mut a, mut b) = (0.0, 0.0);
    for (i, &y) in out.iter().enumerate().skip(start).take(len) {
        a += y as f64 * (w * i as f64).sin();
        b += y as f64 * (w * i as f64).cos();
    }
    let mut d = (-b).atan2(a) / w;
    d += ((near - d) / period).round() * period;
    Ok(d)
}

impl<'a> Mono16k<'a> {
    pub fn new(rate: u32, on_chunk: &'a mut dyn FnMut(&[f32]) -> Result<()>) -> Result<Self> {
        let resampler = if rate == SAMPLE_RATE { None } else { Some(make(rate)?) };
        // Align to the input: the filter's true group delay is
        // fractional in output samples (it depends on the rate), and a
        // sub-sample skew costs ~0.3% correlation on speech. Prepend `pad`
        // zero inputs so delay + pad·ratio lands on a whole output sample,
        // then drop exactly that many outputs.
        let (pad, delay_left) = match &resampler {
            None => (0, 0),
            Some(_) => {
                let d = measure_delay(rate)?;
                let ratio = SAMPLE_RATE as f64 / rate as f64;
                let target = avfoundation_shift(rate) as f64 * ratio;
                let skip = |k: usize| d + k as f64 * ratio - target;
                let off = |k: usize| (skip(k) - skip(k).round()).abs();
                let k = (0..2000).min_by(|&a, &b| off(a).total_cmp(&off(b))).unwrap_or(0);
                (k, skip(k).round().max(0.0) as usize)
            }
        };
        let out = vec![vec![0f32; resampler.as_ref().map_or(0, |r| r.output_frames_max())]];
        let mut pending_in = Vec::with_capacity(CHUNK_IN * 2);
        pending_in.resize(pad, 0.0);
        Ok(Self {
            rate,
            resampler,
            pending_in,
            out,
            ready: Vec::with_capacity(EMIT * 2),
            delay_left,
            in_total: 0,
            out_total: 0,
            on_chunk,
        })
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Feed mono samples at the input rate.
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.in_total += samples.len() as u64;
        if self.resampler.is_none() {
            return self.take(samples, u64::MAX);
        }
        self.pending_in.extend_from_slice(samples);
        let mut start = 0;
        while self.pending_in.len() - start >= CHUNK_IN {
            let r = self.resampler.as_mut().unwrap();
            let (_, n) = r.process_into_buffer(&[&self.pending_in[start..start + CHUNK_IN]], &mut self.out, None)?;
            start += CHUNK_IN;
            let out = std::mem::take(&mut self.out);
            self.take(&out[0][..n], u64::MAX)?;
            self.out = out;
        }
        self.pending_in.drain(..start);
        Ok(())
    }

    /// Flush: zero-pad the tail through the resampler, cut to the exact
    /// expected length, and emit what's left.
    pub fn finish(mut self) -> Result<()> {
        let expected = if self.resampler.is_none() {
            self.in_total
        } else {
            (self.in_total as f64 * SAMPLE_RATE as f64 / self.rate as f64).round() as u64
        };
        if self.resampler.is_some() {
            let tail = std::mem::take(&mut self.pending_in);
            let mut first = true;
            while self.out_total + (self.ready.len() as u64) < expected {
                let r = self.resampler.as_mut().unwrap();
                let input: Option<&[&[f32]]> = if first { Some(&[&tail[..]]) } else { None };
                first = false;
                let (_, n) = r.process_partial_into_buffer(input, &mut self.out, None)?;
                if n == 0 {
                    break;
                }
                let out = std::mem::take(&mut self.out);
                self.take(&out[0][..n], expected)?;
                self.out = out;
            }
        }
        if !self.ready.is_empty() {
            let buf = std::mem::take(&mut self.ready);
            (self.on_chunk)(&buf)?;
            self.out_total += buf.len() as u64;
        }
        Ok(())
    }

    /// Queue resampled output (dropping the leading group delay and anything
    /// past `limit` total samples), emitting full chunks.
    fn take(&mut self, mut s: &[f32], limit: u64) -> Result<()> {
        let skip = self.delay_left.min(s.len());
        self.delay_left -= skip;
        s = &s[skip..];
        let room = limit.saturating_sub(self.out_total + self.ready.len() as u64);
        s = &s[..s.len().min(room.min(usize::MAX as u64) as usize)];
        self.ready.extend_from_slice(s);
        if self.ready.len() >= EMIT {
            let buf = std::mem::take(&mut self.ready);
            (self.on_chunk)(&buf)?;
            self.out_total += buf.len() as u64;
            self.ready = buf;
            self.ready.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(rate: u32, n: usize, piece: usize) -> Vec<f32> {
        let f = 440.0;
        let input: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / rate as f32).sin()).collect();
        let mut out = Vec::new();
        let mut sink = |c: &[f32]| {
            out.extend_from_slice(c);
            Ok(())
        };
        let mut r = Mono16k::new(rate, &mut sink).unwrap();
        for c in input.chunks(piece) {
            r.push(c).unwrap();
        }
        r.finish().unwrap();
        out
    }

    #[test]
    fn exact_length_and_alignment() {
        for (rate, n) in [(44_100, 44_100 * 3 + 17), (48_000, 1000), (22_050, 5), (16_000, 12_345), (8_000, 8_001)] {
            let out = run(rate, n, 1234);
            let want = (n as f64 * 16_000.0 / rate as f64).round() as usize;
            assert_eq!(out.len(), want, "rate {rate}");
            if n > 10_000 {
                // Delay trimmed to the sub-sample: in phase with an ideal sine
                // (shifted like AVFoundation's output).
                let err_at = |shift: f32| {
                    let ideal = |i: usize| (2.0 * std::f32::consts::PI * 440.0 * (i as f32 + shift) / 16_000.0).sin();
                    (2000..out.len() - 2000).map(|i| (out[i] - ideal(i)).abs()).fold(0.0, f32::max)
                };
                let err = err_at(-(avfoundation_shift(rate) as f32) * 16_000.0 / rate as f32);
                assert!(err < 3e-3, "rate {rate}: max err {err}");
            }
        }
    }
}
