//! FFmpeg CLI sidecar: probing, video frames and a fallback audio decoder.
//!
//! Why a sidecar process rather than linking libav* (ffmpeg-next) or the OS
//! frameworks (Media Foundation / GStreamer): one LGPL binary per platform
//! decodes every codec/container we care about identically on Windows and
//! Linux, needs no C toolchain or pkg-config at build time, keeps LGPL
//! obligations trivial (an unmodified, separately shipped executable), and a
//! crash or hang in a decoder on a hostile file kills a child process, not
//! the app. The cost is ~10–40 ms of process start per call, amortized by
//! batching all of a call's frames into as few processes as possible.
//!
//! Frames come back as PPM (`-c:v ppm -f image2pipe`): each frame carries its
//! own dimensions, so we never need to predict the scaler's rounding, and the
//! `showinfo` filter reports each emitted frame's timestamp on stderr so we can
//! map frames back to requested times.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::RwLock;

use anyhow::{anyhow, bail, Context, Result};

use super::resample::Mono16k;
use super::Probe;
use crate::image::Rgb8;

static FFMPEG: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Use this ffmpeg executable (e.g. the bundled sidecar) instead of the
/// default lookup.
pub fn set_ffmpeg_path(path: &Path) {
    *FFMPEG.write().unwrap() = Some(path.to_path_buf());
}

/// The ffmpeg executable to run: [`set_ffmpeg_path`], else `$SEMANTRA_FFMPEG`,
/// else an `ffmpeg` next to the current executable (where Tauri puts
/// `externalBin` sidecars), else `ffmpeg` on `PATH`.
pub fn ffmpeg_path() -> PathBuf {
    if let Some(p) = FFMPEG.read().unwrap().clone() {
        return p;
    }
    if let Some(p) = std::env::var_os("SEMANTRA_FFMPEG") {
        return p.into();
    }
    let exe = format!("ffmpeg{}", std::env::consts::EXE_SUFFIX);
    if let Some(dir) = std::env::current_exe().ok().as_deref().and_then(Path::parent) {
        let beside = dir.join(&exe);
        if beside.is_file() {
            return beside;
        }
    }
    PathBuf::from(exe)
}

/// Whether the ffmpeg executable can be started.
pub fn available() -> bool {
    command().arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

fn command() -> Command {
    let mut c = Command::new(ffmpeg_path());
    c.args(["-hide_banner", "-nostdin"]).stdin(Stdio::null());
    #[cfg(windows)]
    {
        // Don't flash a console window from a GUI app.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c
}

/// `file:` keeps ffmpeg from interpreting names like `concat:a|b` or
/// `http://…` as protocols.
fn input(path: &Path) -> std::ffi::OsString {
    let mut s = std::ffi::OsString::from("file:");
    s.push(path.as_os_str());
    s
}

fn spawn(c: &mut Command) -> Result<Child> {
    c.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            anyhow!("ffmpeg not found at {} (bundle the sidecar or call set_ffmpeg_path)", ffmpeg_path().display())
        } else {
            anyhow!("cannot start ffmpeg: {e}")
        }
    })
}

/// Collect a child's stderr on a thread (so a chatty stderr can't deadlock a
/// stdout reader).
fn stderr_thread(child: &mut Child) -> std::thread::JoinHandle<String> {
    let mut err = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    })
}

/// What ffmpeg's input summary says about a file.
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub duration_s: Option<f64>,
    pub start_s: f64,
    pub has_audio: bool,
    /// Sample rate of the first audio stream.
    pub sample_rate: Option<u32>,
    /// The first audio stream is HE-AAC (v1 or v2: AAC + SBR [+ PS]).
    pub he_aac: bool,
    /// A real video stream (not cover art / an attached picture).
    pub has_video: bool,
    /// Average frame rate of the first video stream, when reported.
    pub fps: Option<f64>,
    /// Coded size of the first video stream.
    pub size: Option<(u32, u32)>,
    /// Color tags of the first video stream as ffmpeg names them
    /// (`bt709`, `bt470bg`, `smpte170m`, `iec61966-2-1`, …); `None` when
    /// absent or `unknown`.
    pub matrix: Option<String>,
    pub transfer: Option<String>,
    /// Display-matrix rotation of the first video stream (degrees).
    pub rotation: f64,
}

impl Info {
    /// Video size as displayed (after the rotation ffmpeg's autorotate
    /// applies).
    pub fn display_size(&self) -> Option<(u32, u32)> {
        let (w, h) = self.size?;
        let quarter = ((self.rotation / 90.0).round() as i64).rem_euclid(2) == 1;
        Some(if quarter { (h, w) } else { (w, h) })
    }
}

/// Frame size for `maximumSize = max_side × max_side`, computed the way
/// AVAssetImageGenerator does — a single-precision scale factor, short side
/// floored — so 1920x1080 at 1344 is 1344x755 (0.7f · 1080 = 755.99998) and 1280x720 at
/// 448 is 448x251; matched on every test video. Never upscales.
pub fn fit_frame(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= max_side {
        return (w, h);
    }
    // A single-precision scale applied in double precision.
    let s = (max_side as f32 / long as f32) as f64;
    let short = |v: u32| ((v as f64 * s).floor() as u32).max(1);
    if w >= h {
        (max_side, short(h))
    } else {
        (short(w), max_side)
    }
}

/// [`info`], cached by path + size + mtime: the app calls `frames_at` once per
/// 8-frame block of the same video, and each `ffmpeg -i` costs ~50 ms.
pub(crate) fn info_cached(path: &Path) -> Result<Info> {
    type Key = (PathBuf, u64, Option<std::time::SystemTime>);
    static CACHE: std::sync::Mutex<Vec<(Key, Info)>> = std::sync::Mutex::new(Vec::new());
    let meta = std::fs::metadata(path)?;
    let key: Key = (path.to_path_buf(), meta.len(), meta.modified().ok());
    if let Some((_, i)) = CACHE.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return Ok(i.clone());
    }
    let i = info(path)?;
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 16 {
        c.remove(0);
    }
    c.push((key, i.clone()));
    Ok(i)
}

/// Probe with `ffmpeg -i` (no ffprobe needed: one binary to ship).
pub fn info(path: &Path) -> Result<Info> {
    let out = command().arg("-i").arg(input(path)).stdout(Stdio::null()).stderr(Stdio::piped()).output();
    let out = out.map_err(|e| anyhow!("cannot start ffmpeg ({}): {e}", ffmpeg_path().display()))?;
    let text = String::from_utf8_lossy(&out.stderr);
    let info = parse_info(&text);
    if !text.contains("Input #0") {
        let last = text.lines().last().unwrap_or("").trim();
        bail!("ffmpeg cannot open {}: {last}", path.display());
    }
    Ok(info)
}

/// Parse the `Input #0` block of ffmpeg's stderr.
pub fn parse_info(text: &str) -> Info {
    let mut info = Info::default();
    let mut in_input = false;
    let mut in_video = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("Input #0") {
            in_input = true;
            continue;
        }
        if !in_input {
            continue;
        }
        if t.starts_with("Input #") || t.starts_with("Output #") || !line.starts_with(' ') {
            break;
        }
        if let Some(rest) = t.strip_prefix("Duration: ") {
            info.duration_s = parse_hms(rest.split(',').next().unwrap_or(""));
            if let Some(s) = rest.split(", ").find_map(|p| p.strip_prefix("start: ")) {
                info.start_s = s.trim().parse().unwrap_or(0.0);
            }
        } else if t.starts_with("Stream #0:") {
            in_video = false;
            if t.contains(": Audio:") && !info.has_audio {
                info.has_audio = true;
                info.sample_rate =
                    t.split(", ").find_map(|p| p.strip_suffix(" Hz").and_then(|r| r.trim().parse().ok()));
                info.he_aac = t.contains("aac (HE-AAC");
            } else if t.contains(": Video:") && !t.contains("(attached pic)") && !info.has_video {
                info.has_video = true;
                in_video = true;
                info.size = t.split([' ', ',']).find_map(|tok| {
                    let (w, h) = tok.split_once('x')?;
                    Some((w.parse().ok()?, h.parse().ok()?))
                });
                (info.matrix, info.transfer) = parse_colors(t);
                info.fps = t
                    .split(", ")
                    .find_map(|p| p.strip_suffix(" fps").and_then(|f| f.trim().parse::<f64>().ok()))
                    .or_else(|| {
                        t.split(", ").find_map(|p| p.strip_suffix(" tbr").and_then(|f| f.trim().parse::<f64>().ok()))
                    });
            }
        } else if in_video {
            // Side data: "displaymatrix: rotation of -90.00 degrees" (ffmpeg
            // <= 8) or "Display Matrix: rotation of 90.00 degrees" (9+).
            if let Some(r) = t.split("rotation of ").nth(1).and_then(|r| r.split_whitespace().next()) {
                info.rotation = r.parse().unwrap_or(0.0);
            }
        }
    }
    info
}

/// Matrix and transfer from a video stream line's pixel-format group, e.g.
/// `yuv420p(tv, bt709/bt709/linear, progressive)` (matrix/primaries/transfer)
/// or `yuv420p(tv, bt709, progressive)` (all three the same).
fn parse_colors(line: &str) -> (Option<String>, Option<String>) {
    let Some(group) = line.split("), ").find_map(|part| {
        let (_, inner) = part.rsplit_once('(')?;
        // The pixel format group, not "(High)" / "(avc1 / 0x…)".
        (inner.contains("progressive")
            || inner.contains("tv")
            || inner.contains("pc")
            || inner.contains('/') && !inner.contains(" / "))
        .then_some(inner)
    }) else {
        return (None, None);
    };
    let known = |v: &str| (v != "unknown" && !v.is_empty()).then(|| v.to_string());
    for item in group.split(", ") {
        let item = item.trim_end_matches(')');
        if matches!(item, "tv" | "pc" | "progressive" | "top first" | "bottom first" | "top coded first (swapped)") {
            continue;
        }
        let parts: Vec<&str> = item.split('/').collect();
        return match parts.as_slice() {
            [m, _p, t] => (known(m), known(t)),
            [all] => (known(all), known(all)),
            _ => (None, None),
        };
    }
    (None, None)
}

fn parse_hms(s: &str) -> Option<f64> {
    let mut parts = s.trim().split(':');
    let (h, m, sec) = (parts.next()?, parts.next()?, parts.next()?);
    Some(h.parse::<f64>().ok()? * 3600.0 + m.parse::<f64>().ok()? * 60.0 + sec.parse::<f64>().ok()?)
}

/// Duration by remuxing to null (demux only, fast) when the container header
/// has none (some WebM/MKV, raw streams): the last progress `time=` wins.
fn scan_duration(path: &Path) -> Result<f64> {
    let out = command()
        .args(["-v", "error", "-stats", "-i"])
        .arg(input(path))
        .args(["-map", "0", "-c", "copy", "-f", "null", "-"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()?;
    let text = String::from_utf8_lossy(&out.stderr);
    text.rsplit("time=")
        .next()
        .filter(|_| text.contains("time="))
        .and_then(|s| parse_hms(s.split_whitespace().next().unwrap_or("")))
        .ok_or_else(|| anyhow!("cannot determine duration of {}", path.display()))
}

/// [`Probe`] via ffmpeg.
pub fn probe(path: &Path) -> Result<Probe> {
    let i = info(path)?;
    if !(i.has_audio || i.has_video) {
        bail!("{} has no playable audio or video", path.display());
    }
    let duration_s = match i.duration_s {
        Some(d) => d,
        None => scan_duration(path)?,
    };
    if !duration_s.is_finite() {
        bail!("{} has no playable audio or video", path.display());
    }
    Ok(Probe { duration_s, has_audio: i.has_audio, has_video: i.has_video })
}

/// First audio stream decoded and downmixed by ffmpeg at its native rate,
/// streamed from stdout as mono F32, then resampled to 16 kHz by the same
/// [`Mono16k`] as the Symphonia path (so alignment and length match it). For
/// codecs Symphonia lacks (Opus, HE-AAC, AC-3, E-AC-3, WMA, DTS, …).
pub fn stream_audio(path: &Path, on_chunk: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
    let i = info_cached(path)?;
    if !i.has_audio {
        bail!("{} has no audio track", path.display());
    }
    let rate = i.sample_rate.unwrap_or(48_000);
    let mut out = Mono16k::new(rate, on_chunk)?;
    // ffmpeg's AAC decoder leaves the SBR filterbank delay (481 core = 962
    // output samples) in its HE-AAC output; AVFoundation removes it. Measured
    // 962.0 samples on an Apple-encoded HE-AAC file, and a 22 ms skew drops
    // the audio embedding's cosine to ~0.97, so it's worth matching. The
    // tail is zero-padded to keep the length.
    let mut skip = if i.he_aac { SBR_DELAY } else { 0 };
    let mut skipped = 0usize;
    let mut child = spawn(
        command()
            .args(["-v", "error", "-i"])
            .arg(input(path))
            .args(["-map", "0:a:0", "-vn", "-sn", "-dn"])
            // swresample's default mono downmix already matches
            // AVFoundation's for front/center/LFE ((L + R)/√2 + C); raising
            // the surround level to 1 makes surrounds -3 dB like AVFoundation
            // too (see native::downmix_weights). `-ar` pins the rate in case
            // it changes mid-stream (chained Ogg).
            .args(["-af", "aresample=surround_mix_level=1"])
            .args(["-ac", "1", "-ar", &rate.to_string(), "-f", "f32le", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
    )?;
    let err = stderr_thread(&mut child);
    let mut stdout = child.stdout.take().unwrap();
    let mut bytes = vec![0u8; 64 * 1024];
    let mut filled = 0usize;
    let mut samples: Vec<f32> = Vec::with_capacity(bytes.len() / 4);
    let mut result = Ok(());
    loop {
        let n = match stdout.read(&mut bytes[filled..]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                result = Err(e.into());
                break;
            }
        };
        filled += n;
        let whole = filled / 4 * 4;
        if whole == 0 {
            continue;
        }
        samples.clear();
        samples.extend(bytes[..whole].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        bytes.copy_within(whole..filled, 0);
        filled -= whole;
        let k = skip.min(samples.len());
        skip -= k;
        skipped += k;
        if let Err(e) = out.push(&samples[k..]) {
            result = Err(e);
            break;
        }
    }
    if result.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let err = err.join().unwrap_or_default();
    result?;
    if !status.success() {
        bail!("ffmpeg failed decoding audio of {}: {}", path.display(), err.trim());
    }
    out.push(&vec![0.0; skipped])?;
    out.finish()
}

/// See `stream_audio`.
const SBR_DELAY: usize = 962;

/// A still image ffmpeg can decode but the `image` crate can't (HEIC/HEIF,
/// AVIF), at full size: the primary item, with tile grids stitched and
/// `irot`/`imir` orientation applied (ffmpeg >= 9 builds the grid graph
/// itself, which is also why no `-vf` can be added here). Color is converted
/// with the file's YUV matrix but not color-managed (Display P3 HEICs come
/// out as if sRGB).
pub fn still(path: &Path) -> Result<Rgb8> {
    let mut child = spawn(
        command()
            .args(["-v", "error", "-i"])
            .arg(input(path))
            .args(["-frames:v", "1", "-an", "-pix_fmt", "rgb24", "-c:v", "ppm", "-f", "image2pipe", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
    )?;
    let err = stderr_thread(&mut child);
    let mut r = BufReader::new(child.stdout.take().unwrap());
    let frame = read_ppm(&mut r);
    let status = child.wait()?;
    let err = err.join().unwrap_or_default();
    match frame {
        Ok(Some(f)) => Ok(f),
        _ if !status.success() => bail!("ffmpeg cannot decode {}: {}", path.display(), err.trim()),
        Ok(None) => bail!("ffmpeg produced no image for {}", path.display()),
        Err(e) => Err(e),
    }
}

/// How to fetch frames for one `frames_at` call.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Plan {
    /// One process decodes the whole span once and `select` keeps the frames
    /// displayed at each time: best when times are dense relative to the
    /// keyframe interval.
    Sequential,
    /// One process per time, each seeking to the nearest prior keyframe:
    /// best for sparse times (long videos sampled every 10+ s).
    Seek,
}

/// Mean gap between requested times (s) above which per-time seeking beats
/// one sequential decode (parity `bench`).
const SEEK_GAP_S: f64 = 3.0;

/// Frames at `times_s`, rotated per the stream's display matrix (ffmpeg's
/// autorotate), scaled so the longer side <= `max_side` (never up). Each is
/// the frame displayed at that time, or — when `tolerance_s` allows — the
/// nearest keyframe within ±`tolerance_s` (cheap to decode, like
/// AVAssetImageGenerator with a tolerance).
pub fn frames_at(path: &Path, times_s: &[f64], max_side: u32, tolerance_s: f64) -> Result<Vec<Rgb8>> {
    if times_s.is_empty() {
        return Ok(Vec::new());
    }
    if max_side == 0 {
        bail!("max_side must be positive");
    }
    let i = info_cached(path)?;
    if !i.has_video {
        bail!("{} has no video track", path.display());
    }
    let frame_dur = 1.0 / i.fps.filter(|f| *f > 0.0 && f.is_finite()).unwrap_or(30.0);
    let mut sorted: Vec<f64> = times_s.iter().map(|t| t.max(0.0)).collect();
    sorted.sort_by(f64::total_cmp);
    let span = sorted[sorted.len() - 1] - sorted[0];
    let plan =
        if sorted.len() > 1 && span / (sorted.len() - 1) as f64 <= SEEK_GAP_S { Plan::Sequential } else { Plan::Seek };
    let size = i.display_size().map(|(w, h)| fit_frame(w, h, max_side));
    let fixup = Fixup::new(tone_for(&i));
    let ctx = Ctx { path, max_side, size, matrix: matrix_for(&i), fixup: &fixup, frame_dur, start_s: i.start_s };
    let frames: Vec<Rgb8> = match plan {
        Plan::Sequential => ctx.sequential(times_s)?,
        Plan::Seek => {
            // Processes in parallel: each is mostly a single-threaded demux +
            // a short decode (~0.1 s for 1080p, mostly startup and seeking).
            // One ffmpeg with an input per time was measured slower (its
            // inputs don't decode concurrently).
            let par = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 8);
            let mut out: Vec<Option<Result<Rgb8>>> = (0..times_s.len()).map(|_| None).collect();
            for (idx, chunk) in times_s.chunks(par).enumerate() {
                std::thread::scope(|s| {
                    let handles: Vec<_> =
                        chunk.iter().map(|&t| s.spawn(move || ctx.one(t.max(0.0), tolerance_s))).collect();
                    for (k, h) in handles.into_iter().enumerate() {
                        out[idx * par + k] = Some(h.join().unwrap_or_else(|_| Err(anyhow!("frame thread panicked"))));
                    }
                });
            }
            out.into_iter().map(|r| r.unwrap()).collect::<Result<_>>()?
        }
    };
    Ok(frames)
}

/// YUV -> RGB matrix for the scaler. Tagged streams use their tag. Untagged
/// ones follow AVFoundation, which assumes BT.709 for HD (>= 1280x720) and
/// BT.601 below (swscale would assume BT.601 everywhere).
fn matrix_for(i: &Info) -> &'static str {
    match (i.matrix.as_deref(), i.size) {
        (Some(_), _) => "auto",
        (None, Some((w, h))) if w >= 1280 || h >= 720 => "bt709",
        (None, _) => "bt601",
    }
}

/// The tone conversion AVFoundation applies when it hands frames over as
/// sRGB. Measured per tag combination (parity `color`): video gamma is
/// decoded as a pure 1.961 power — Apple's BT.709 display approximation — and
/// re-encoded as sRGB, which lifts midtones (128 -> 139). It does this for
/// untagged, BT.709 and BT.601/SMPTE-170M video, but not when the matrix is
/// tagged BT.470BG with no transfer (PAL-style files, e.g. the app's sample
/// clip) or the transfer is already sRGB. Skipping it costs ~15 dB PSNR
/// against macOS frames (33 vs 48 dB).
///
/// HDR (HLG, PQ — e.g. iPhone HEVC) has no 8-bit LUT equivalent: the
/// transfer is inverted to linear light, BT.2020 primaries are converted to
/// BT.709/sRGB, then clipped and sRGB-encoded. HLG uses scene light × 1.15,
/// the gain that best matched AVFoundation (34 dB on a test clip, vs 16 dB
/// untreated); PQ maps 203 nits (reference white) to SDR white. These are
/// approximations of Apple's tone mapper, not reproductions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tone {
    None,
    Video,
    Linear,
    Hlg,
    Pq,
}

pub fn tone_for(i: &Info) -> Tone {
    match (i.transfer.as_deref(), i.matrix.as_deref()) {
        (Some("iec61966-2-1"), _) => Tone::None,
        (Some("linear"), _) => Tone::Linear,
        (Some("arib-std-b67"), _) => Tone::Hlg,
        (Some("smpte2084"), _) => Tone::Pq,
        (None, Some("bt470bg")) => Tone::None,
        _ => Tone::Video,
    }
}

fn srgb_encode(l: f64) -> f64 {
    if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// HLG inverse OETF (BT.2100): signal -> scene light in [0, 1].
fn hlg_to_linear(e: f64) -> f64 {
    let (a, b, c) = (0.178_832_77, 0.284_668_92, 0.559_910_73);
    if e <= 0.5 {
        e * e / 3.0
    } else {
        (((e - c) / a).exp() + b) / 12.0
    }
}

/// PQ EOTF (SMPTE ST 2084): signal -> display light in nits.
fn pq_to_nits(e: f64) -> f64 {
    let (m1, m2, c1, c2, c3) = (0.159_301_757_812_5, 78.843_75, 0.835_937_5, 18.851_562_5, 18.687_5);
    let p = e.powf(1.0 / m2);
    10_000.0 * ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1)
}

/// Per-frame color fix-up applied to the scaler's RGB output.
#[derive(Clone)]
enum Fixup {
    None,
    /// Per-channel 8-bit curve (SDR).
    Lut(Box<[u8; 256]>),
    /// Signal -> linear (per channel), BT.2020 -> BT.709 matrix, sRGB encode.
    Hdr {
        linear: Box<[f32; 256]>,
        encode: Box<[u8; 4096]>,
    },
}

impl Fixup {
    fn new(t: Tone) -> Self {
        let lut = |curve: &dyn Fn(f64) -> f64| {
            let mut lut = [0u8; 256];
            for (i, o) in lut.iter_mut().enumerate() {
                *o = (curve(i as f64 / 255.0) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            Fixup::Lut(Box::new(lut))
        };
        let hdr = |to_linear: &dyn Fn(f64) -> f64| {
            let mut linear = [0f32; 256];
            for (i, o) in linear.iter_mut().enumerate() {
                *o = to_linear(i as f64 / 255.0) as f32;
            }
            let mut encode = [0u8; 4096];
            for (i, o) in encode.iter_mut().enumerate() {
                *o = (srgb_encode(i as f64 / 4095.0) * 255.0).round() as u8;
            }
            Fixup::Hdr { linear: Box::new(linear), encode: Box::new(encode) }
        };
        match t {
            Tone::None => Fixup::None,
            Tone::Video => lut(&|v| srgb_encode(v.powf(1.961))),
            Tone::Linear => lut(&srgb_encode),
            Tone::Hlg => hdr(&|e| 1.15 * hlg_to_linear(e)),
            Tone::Pq => hdr(&|e| pq_to_nits(e) / 203.0),
        }
    }

    fn apply(&self, rgb: &mut [u8]) {
        match self {
            Fixup::None => {}
            Fixup::Lut(lut) => rgb.iter_mut().for_each(|v| *v = lut[*v as usize]),
            Fixup::Hdr { linear, encode } => {
                const M: [[f32; 3]; 3] =
                    [[1.660_5, -0.587_6, -0.072_8], [-0.124_6, 1.132_9, -0.008_3], [-0.018_2, -0.100_6, 1.118_7]];
                for px in rgb.as_chunks_mut::<3>().0 {
                    let l = px.map(|v| linear[v as usize]);
                    for (o, row) in px.iter_mut().zip(M) {
                        let v = (row[0] * l[0] + row[1] * l[1] + row[2] * l[2]).clamp(0.0, 1.0);
                        *o = encode[(v * 4095.0).round() as usize];
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Ctx<'a> {
    path: &'a Path,
    max_side: u32,
    /// Output size, when the stream's size is known.
    size: Option<(u32, u32)>,
    /// swscale `in_color_matrix`.
    matrix: &'static str,
    /// Color fix-up applied to the RGB output.
    fixup: &'a Fixup,
    frame_dur: f64,
    start_s: f64,
}

impl Ctx<'_> {
    /// Scale (downscale only, longer side -> max_side; Catmull-Rom, the
    /// still-image filter, as `bicubic` with B = 0, C = 0.5),
    /// then report timestamps, then emit PPM. Autorotation is inserted by
    /// ffmpeg before these filters, so `iw`/`ih` are display dimensions.
    fn tail_filters(&self) -> String {
        let m = self.max_side;
        // Unknown stream size (unparsed probe output): let the scaler fit it,
        // rounding the short side down in double precision.
        let (w, h) = match self.size {
            Some((w, h)) => (w.to_string(), h.to_string()),
            None => (
                format!("'if(lte(max(iw,ih),{m}),iw,if(gte(iw,ih),{m},max(1,floor(iw*({m}/ih)))))'"),
                format!("'if(lte(max(iw,ih),{m}),ih,if(gte(iw,ih),max(1,floor(ih*({m}/iw))),{m}))'"),
            ),
        };
        format!(
            "scale=w={w}:h={h}:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5\
             :in_color_matrix={matrix},format=rgb24,showinfo",
            matrix = self.matrix
        )
    }

    fn base(&self) -> Command {
        let mut c = command();
        c.args(["-loglevel", "info"]);
        c
    }

    /// Run a frame-producing command; returns frames and their `pts_time`s.
    fn run(&self, c: &mut Command) -> Result<Vec<(f64, Rgb8)>> {
        c.args(["-an", "-sn", "-dn", "-fps_mode", "passthrough", "-c:v", "ppm", "-f", "image2pipe", "-"]);
        let mut child = spawn(c.stdout(Stdio::piped()).stderr(Stdio::piped()))?;
        let err = stderr_thread(&mut child);
        let mut r = BufReader::with_capacity(1 << 20, child.stdout.take().unwrap());
        let mut frames = Vec::new();
        let read = (|| -> Result<()> {
            while let Some(mut f) = read_ppm(&mut r)? {
                self.fixup.apply(&mut f.data);
                frames.push(f);
            }
            Ok(())
        })();
        let status = child.wait()?;
        let err = err.join().unwrap_or_default();
        read?;
        if !status.success() && frames.is_empty() {
            let msg = err.lines().rfind(|l| !l.contains("showinfo")).unwrap_or("").trim().to_string();
            bail!("ffmpeg failed reading frames of {}: {msg}", self.path.display());
        }
        let times: Vec<f64> = err
            .lines()
            .filter(|l| l.contains("Parsed_showinfo"))
            .filter_map(|l| l.split("pts_time:").nth(1)?.split_whitespace().next()?.parse().ok())
            .collect();
        // showinfo can see a frame or two more than `-frames:v` lets through
        // to the muxer; written frames are always a prefix, in order.
        if times.len() < frames.len() {
            bail!("ffmpeg reported {} timestamps for {} frames", times.len(), frames.len());
        }
        Ok(times.into_iter().zip(frames).collect())
    }

    /// The frame displayed at `t` (pts <= t < next pts) has pts > t - one
    /// frame duration; selecting the first frame at or after that threshold
    /// picks it for constant-frame-rate video.
    fn threshold(&self, t: f64) -> f64 {
        (t - self.frame_dur * 0.999).max(0.0)
    }

    /// All frames in one pass: seek (accurately) to just before the first
    /// time, then `select` the first frame at/after each threshold.
    fn sequential(&self, times: &[f64]) -> Result<Vec<Rgb8>> {
        let mut sorted: Vec<f64> = times.iter().map(|&t| t.max(0.0)).collect();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        let seek = (self.threshold(sorted[0]) - 0.5).max(0.0);
        // Thresholds relative to the seek point (output timestamps restart
        // at 0 after an input seek).
        let rel: Vec<f64> = sorted.iter().map(|&t| (self.threshold(t) - seek).max(0.0)).collect();
        let terms: Vec<String> =
            rel.iter().map(|r| format!("gte(t,{r:.6})*(isnan(prev_selected_t)+lt(prev_selected_t,{r:.6}))")).collect();
        // `prev_selected_t`, not `prev_t`: once a frame is taken for one
        // threshold, the next threshold's frame must come strictly later.
        let expr = terms.join("+");
        let end = rel[rel.len() - 1] + 2.0 * self.frame_dur + 1.0;
        let script = format!("select='{expr}',{}", self.tail_filters());
        // Long expressions go through a filter script file (command lines
        // are capped at 32 KiB on Windows).
        let tmp = tempfile_path("vf")?;
        std::fs::write(&tmp, &script)?;
        let mut c = self.base();
        c.args(["-ss", &format!("{seek:.6}"), "-t", &format!("{end:.6}"), "-i"])
            .arg(input(self.path))
            .args(["-map", "0:V:0", "-/vf"])
            .arg(&tmp);
        let got = self.run(&mut c);
        let _ = std::fs::remove_file(&tmp);
        let got = got?;
        if got.is_empty() {
            bail!("no frames decoded from {}", self.path.display());
        }
        // Map each requested time to the first emitted frame at/after its
        // threshold (or the last frame, for times past the end).
        times
            .iter()
            .map(|&t| {
                let r = (self.threshold(t.max(0.0)) - seek).max(0.0);
                let k = got.iter().position(|(ft, _)| *ft >= r - 1e-4).unwrap_or(got.len() - 1);
                Ok(got[k].1.clone())
            })
            .collect()
    }

    /// One frame at `t`: a keyframe within ±`tolerance` if there is one
    /// (decodes a single picture), else the exact displayed frame.
    fn one(&self, t: f64, tolerance: f64) -> Result<Rgb8> {
        if tolerance > 0.0 {
            // Seeking to the window's end lands on the latest keyframe at or
            // before it — after `t` if one is close enough, as
            // AVAssetImageGenerator also uses its tolerance-after.
            let mut c = self.base();
            let to = t + tolerance;
            // One thread: with frame threading the decoder only emits its
            // first picture after a packet per thread, i.e. it would decode
            // a keyframe from each of the next N GOPs.
            c.args(["-threads", "1", "-skip_frame", "nokey", "-noaccurate_seek", "-copyts"])
                .args(["-ss", &format!("{to:.6}"), "-i"])
                .arg(input(self.path))
                .args(["-map", "0:V:0", "-frames:v", "1", "-vf", &self.tail_filters()]);
            if let Some((kt, f)) = self.run(&mut c)?.into_iter().next() {
                // -copyts keeps container timestamps, which include the
                // stream's start offset.
                if (kt - self.start_s - t).abs() <= tolerance {
                    return Ok(f);
                }
            }
        }
        let seek = self.threshold(t);
        let mut c = self.base();
        c.args(["-ss", &format!("{seek:.6}"), "-i"]).arg(input(self.path)).args([
            "-map",
            "0:V:0",
            "-frames:v",
            "1",
            "-vf",
            &self.tail_filters(),
        ]);
        let got = self.run(&mut c)?;
        if let Some((_, f)) = got.into_iter().next() {
            return Ok(f);
        }
        // Past the last frame: take the final picture.
        let mut c = self.base();
        c.args(["-sseof", "-1", "-i"]).arg(input(self.path)).args(["-map", "0:V:0", "-vf", &self.tail_filters()]);
        self.run(&mut c)?.pop().map(|(_, f)| f).ok_or_else(|| anyhow!("no frame at {t:.2}s in {}", self.path.display()))
    }
}

fn tempfile_path(tag: &str) -> Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let p = std::env::temp_dir().join(format!(
        "semantra-ffmpeg-{}-{}-{tag}.txt",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(p)
}

/// Read one binary PPM (P6, maxval 255) frame; `None` at a clean EOF.
fn read_ppm(r: &mut impl BufRead) -> Result<Option<Rgb8>> {
    if r.fill_buf()?.is_empty() {
        return Ok(None);
    }
    let mut fields = [0u32; 3];
    let magic = token(r)?;
    if magic != "P6" {
        bail!("unexpected frame format {magic:?}");
    }
    for f in &mut fields {
        *f = token(r)?.parse().context("bad PPM header")?;
    }
    let [width, height, maxval] = fields;
    if maxval != 255 || width == 0 || height == 0 {
        bail!("unsupported PPM {width}x{height} maxval {maxval}");
    }
    let mut data = vec![0u8; width as usize * height as usize * 3];
    r.read_exact(&mut data)?;
    Ok(Some(Rgb8 { width, height, data }))
}

/// Next whitespace-delimited header token; consumes exactly one trailing
/// whitespace byte (PPM's separator before the pixel data).
fn token(r: &mut impl BufRead) -> Result<String> {
    let mut s = String::new();
    let mut byte = [0u8; 1];
    loop {
        r.read_exact(&mut byte)?;
        let c = byte[0];
        if c == b'#' && s.is_empty() {
            let mut line = Vec::new();
            r.read_until(b'\n', &mut line)?;
            continue;
        }
        if c.is_ascii_whitespace() {
            if s.is_empty() {
                continue;
            }
            return Ok(s);
        }
        s.push(c as char);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_input_summary() {
        let text = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'file:x.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:01:02.50, start: 0.023220, bitrate: 1205 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 1000 kb/s, 29.97 fps, 29.97 tbr, 30k tbn (default)
      Side data:
        displaymatrix: rotation of -90.00 degrees
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
At least one output file must be specified
";
        let i = parse_info(text);
        assert_eq!(i.duration_s, Some(62.5));
        assert!((i.start_s - 0.02322).abs() < 1e-9);
        assert!(i.has_audio && i.has_video);
        assert_eq!(i.fps, Some(29.97));
        assert_eq!(i.sample_rate, Some(44_100));
        assert_eq!(i.size, Some((1920, 1080)));
        assert_eq!(i.display_size(), Some((1080, 1920)));
        assert_eq!(i.matrix.as_deref(), Some("bt709"));
        assert_eq!(tone_for(&i), Tone::Video);
        let line = "  Stream #0:0[0x1]: Video: h264 (High) (avc1 / 0x31637661), yuvj420p(pc, bt470bg/unknown/unknown, progressive), 1280x720";
        assert_eq!(parse_colors(line), (Some("bt470bg".into()), None));
        let line = "  Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709/bt709/iec61966-2-1, progressive), 1280x720";
        assert_eq!(parse_colors(line), (Some("bt709".into()), Some("iec61966-2-1".into())));
        let line = "  Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080, 5654 kb/s";
        assert_eq!(parse_colors(line), (None, None));
    }

    #[test]
    fn frame_sizes_match_avfoundation() {
        assert_eq!(fit_frame(1920, 1080, 1344), (1344, 755));
        assert_eq!(fit_frame(1280, 720, 448), (448, 251));
        assert_eq!(fit_frame(3840, 2160, 448), (448, 252));
        assert_eq!(fit_frame(720, 1280, 448), (251, 448));
        assert_eq!(fit_frame(640, 360, 1344), (640, 360));
    }

    #[test]
    fn cover_art_is_not_video() {
        let text = "Input #0, mp3, from 'a.mp3':
  Duration: 00:00:03.00, start: 0.000000, bitrate: 64 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 64 kb/s
  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc), 300x300, 90k tbr, 90k tbn (attached pic)
";
        let i = parse_info(text);
        assert!(i.has_audio && !i.has_video);
    }

    #[test]
    fn reads_ppm_stream() {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.extend([1, 2, 3, 4, 5, 6]);
        bytes.extend(b"P6\n1 1\n255\n");
        bytes.extend([7, 8, 9]);
        let mut r = std::io::Cursor::new(bytes);
        let a = read_ppm(&mut r).unwrap().unwrap();
        assert_eq!((a.width, a.height, a.data.as_slice()), (2, 1, &[1u8, 2, 3, 4, 5, 6][..]));
        let b = read_ppm(&mut r).unwrap().unwrap();
        assert_eq!(b.data, vec![7, 8, 9]);
        assert!(read_ppm(&mut r).unwrap().is_none());
    }
}
