//! Behavioral tests on synthetic media, runnable on any OS. Image and WAV
//! cases are generated in-process; cases that need encoded audio/video are
//! generated with the ffmpeg CLI and skipped (with a note) when it's absent.
//! Parity against the macOS decoders lives in the `parity` crate.

use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::process::Command;

use image::{Rgb, Rgba};
use semantra_media_portable::av;
use semantra_media_portable::image::{decode, decode_bytes, decode_max, encode_jpeg, Rgb8};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("semantra-media-portable-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn px(img: &Rgb8, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * img.width + x) * 3) as usize;
    [img.data[i], img.data[i + 1], img.data[i + 2]]
}

fn near(a: [u8; 3], b: [u8; 3], tol: u8) -> bool {
    a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= tol)
}

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const WHITE: [u8; 3] = [255, 255, 255];

/// 64x32 with solid quadrants: TL red, TR green, BL blue, BR white.
fn quadrants(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        Rgb(match (x < w / 2, y < h / 2) {
            (true, true) => RED,
            (false, true) => GREEN,
            (true, false) => BLUE,
            (false, false) => WHITE,
        })
    })
}

fn jpeg(img: &image::RgbImage, q: u8) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q).encode_image(img).unwrap();
    out
}

/// Insert an EXIF APP1 with one Orientation entry right after SOI.
fn with_orientation(jpeg: &[u8], o: u16) -> Vec<u8> {
    let mut tiff = b"MM\0\x2a\0\0\0\x08".to_vec();
    tiff.extend(1u16.to_be_bytes()); // one entry
    tiff.extend(0x0112u16.to_be_bytes());
    tiff.extend(3u16.to_be_bytes()); // SHORT
    tiff.extend(1u32.to_be_bytes());
    tiff.extend(o.to_be_bytes());
    tiff.extend([0, 0]);
    tiff.extend(0u32.to_be_bytes()); // no next IFD
    let mut app1 = b"Exif\0\0".to_vec();
    app1.extend(tiff);
    let mut out = jpeg[..2].to_vec();
    out.extend([0xFF, 0xE1]);
    out.extend(((app1.len() + 2) as u16).to_be_bytes());
    out.extend(app1);
    out.extend(&jpeg[2..]);
    out
}

#[test]
fn exif_orientations() {
    let base = jpeg(&quadrants(64, 32), 95);
    // (top-left, top-right) after applying each orientation, and whether the
    // sides swap.
    let want = [
        (1, RED, GREEN, false),
        (2, GREEN, RED, false),
        (3, WHITE, BLUE, false),
        (4, BLUE, WHITE, false),
        (5, RED, BLUE, true),
        (6, BLUE, RED, true),
        (7, WHITE, GREEN, true),
        (8, GREEN, WHITE, true),
    ];
    for (o, tl, tr, swap) in want {
        let img = decode_bytes(&with_orientation(&base, o), 2048).unwrap();
        assert_eq!((img.width, img.height), if swap { (32, 64) } else { (64, 32) }, "orientation {o}");
        assert!(near(px(&img, 2, 2), tl, 40), "orientation {o}: top-left {:?}", px(&img, 2, 2));
        assert!(near(px(&img, img.width - 3, 2), tr, 40), "orientation {o}: top-right");
    }
}

#[test]
fn alpha_composites_over_white() {
    let img = image::RgbaImage::from_fn(8, 4, |x, _| match x {
        0..=1 => Rgba([255, 0, 0, 0]),   // fully transparent red
        2..=3 => Rgba([0, 0, 0, 128]),   // half-transparent black
        4..=5 => Rgba([0, 0, 255, 255]), // opaque blue
        _ => Rgba([0, 255, 0, 64]),      // mostly transparent green
    });
    let path = tmp("alpha.png");
    img.save(&path).unwrap();
    let d = decode(&path).unwrap();
    assert_eq!(px(&d, 0, 0), WHITE);
    assert!(near(px(&d, 2, 0), [127, 127, 127], 1));
    assert_eq!(px(&d, 4, 0), BLUE);
    assert!(near(px(&d, 6, 0), [191, 255, 191], 1));
}

#[test]
fn sixteen_bit_and_grayscale() {
    let rgb16 = image::ImageBuffer::<Rgb<u16>, _>::from_fn(4, 4, |_, _| Rgb([65535, 32768, 0]));
    let path = tmp("rgb16.png");
    rgb16.save(&path).unwrap();
    assert!(near(px(&decode(&path).unwrap(), 1, 1), [255, 128, 0], 1));

    let gray = image::GrayImage::from_fn(4, 4, |_, _| image::Luma([77]));
    let path = tmp("gray.png");
    gray.save(&path).unwrap();
    assert_eq!(px(&decode(&path).unwrap(), 0, 0), [77, 77, 77]);

    let ga = image::ImageBuffer::<image::LumaA<u16>, _>::from_fn(4, 4, |_, _| image::LumaA([0, 32768]));
    let path = tmp("gray_alpha16.png");
    ga.save(&path).unwrap();
    assert!(near(px(&decode(&path).unwrap(), 0, 0), [127, 127, 127], 1));
}

#[test]
fn gif_first_frame() {
    let path = tmp("anim.gif");
    {
        let f = std::fs::File::create(&path).unwrap();
        let mut enc = image::codecs::gif::GifEncoder::new(f);
        let red = image::RgbaImage::from_pixel(8, 8, Rgba([255, 0, 0, 255]));
        let blue = image::RgbaImage::from_pixel(8, 8, Rgba([0, 0, 255, 255]));
        enc.encode_frames([image::Frame::new(red), image::Frame::new(blue)]).unwrap();
    }
    assert!(near(px(&decode(&path).unwrap(), 4, 4), RED, 8));
}

#[test]
fn downscale_only_keeps_aspect() {
    let path = tmp("big.png");
    quadrants(1000, 400).save(&path).unwrap();
    let d = decode_max(&path, 250).unwrap();
    assert_eq!((d.width, d.height), (250, 100));
    assert!(near(px(&d, 10, 10), RED, 2) && near(px(&d, 240, 90), WHITE, 2));
    let d = decode_max(&path, 4000).unwrap();
    assert_eq!((d.width, d.height), (1000, 400), "never upscales");
}

#[test]
fn large_jpeg_scaled_decode() {
    // Big enough for the DCT-domain 1/8 path at 256.
    let bytes = jpeg(&quadrants(4096, 2048), 90);
    let d = decode_bytes(&bytes, 256).unwrap();
    assert_eq!((d.width, d.height), (256, 128));
    assert!(near(px(&d, 5, 5), RED, 30) && near(px(&d, 250, 5), GREEN, 30));
    assert!(near(px(&d, 5, 120), BLUE, 30) && near(px(&d, 250, 120), WHITE, 30));
    // And the full-resolution path agrees on size at the default cap.
    let d = decode_bytes(&bytes, 2048).unwrap();
    assert_eq!((d.width, d.height), (2048, 1024));
}

#[test]
fn jpeg_encode_roundtrip() {
    let img = Rgb8 { width: 64, height: 32, data: quadrants(64, 32).into_raw() };
    let bytes = encode_jpeg(&img, 0.9).unwrap();
    assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    let back = decode_bytes(&bytes, 2048).unwrap();
    assert_eq!((back.width, back.height), (64, 32));
    assert!(near(px(&back, 2, 2), RED, 30));
    assert!(encode_jpeg(&Rgb8 { width: 2, height: 2, data: vec![0; 3] }, 0.8).is_err());
}

#[test]
fn garbage_is_an_error() {
    let path = tmp("garbage.jpg");
    std::fs::write(&path, b"definitely not an image").unwrap();
    assert!(decode(&path).is_err());
    assert!(decode(Path::new("/nonexistent/file.png")).is_err());
}

// ------------------------------------------------------------------ audio

/// 16-bit PCM WAV.
fn write_wav(path: &Path, rate: u32, channels: &[Vec<f32>]) {
    let n = channels[0].len();
    let ch = channels.len() as u16;
    let mut b = Vec::new();
    let data_len = (n * ch as usize * 2) as u32;
    b.extend(b"RIFF");
    b.extend((36 + data_len).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(ch.to_le_bytes());
    b.extend(rate.to_le_bytes());
    b.extend((rate * ch as u32 * 2).to_le_bytes());
    b.extend((ch * 2).to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend(data_len.to_le_bytes());
    for i in 0..n {
        for c in channels {
            b.extend(((c[i].clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
        }
    }
    std::fs::write(path, b).unwrap();
}

fn tone(rate: u32, secs: f32, f: f32, amp: f32) -> Vec<f32> {
    (0..(rate as f32 * secs) as usize).map(|i| amp * (2.0 * PI * f * i as f32 / rate as f32).sin()).collect()
}

fn decode_all(path: &Path) -> Vec<f32> {
    let mut v = Vec::new();
    av::stream_audio(path, |c| {
        v.extend_from_slice(c);
        Ok(())
    })
    .unwrap();
    v
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

#[test]
fn wav_stereo_44k_to_16k_mono() {
    let path = tmp("stereo.wav");
    let l = tone(44_100, 3.0, 440.0, 0.25);
    write_wav(&path, 44_100, &[l.clone(), l.clone()]);
    let p = av::probe(&path).unwrap();
    assert!((p.duration_s - 3.0).abs() < 1e-6 && p.has_audio && !p.has_video);
    let out = decode_all(&path);
    assert_eq!(out.len(), 48_000);
    // AVFoundation's stereo downmix is (L + R)/√2: identical channels gain √2.
    let want = 0.25 * std::f32::consts::FRAC_1_SQRT_2 * 2f32.sqrt();
    assert!((rms(&out[1000..47_000]) - want).abs() < 0.003, "rms {}", rms(&out));
}

#[test]
fn audio_streams_in_chunks() {
    let path = tmp("long.wav");
    write_wav(&path, 16_000, &[tone(16_000, 20.0, 300.0, 0.5)]);
    let mut chunks = 0;
    let mut total = 0;
    av::stream_audio(&path, |c| {
        chunks += 1;
        total += c.len();
        assert!(c.len() <= 40_000, "chunks stay small");
        Ok(())
    })
    .unwrap();
    assert_eq!(total, 320_000);
    assert!(chunks >= 10);
    // A callback error stops decoding and is returned.
    let r = av::stream_audio(&path, |_| anyhow::bail!("stop"));
    assert!(r.unwrap_err().to_string().contains("stop"));
}

// ------------------------------------------- ffmpeg-generated (skipped if absent)

/// Run the same ffmpeg the decoders use (`set_ffmpeg_path` /
/// `$SEMANTRA_FFMPEG` / next to the executable / PATH) to make fixtures.
fn ffmpeg(args: &[&str]) -> bool {
    Command::new(av::ffmpeg::ffmpeg_path())
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .status()
        .is_ok_and(|s| s.success())
}

/// An H.264 encoder this ffmpeg has: libx264 (GPL builds), else libopenh264
/// (BSD; in LGPL builds such as the app's sidecar), else the built-in MPEG-4
/// Part 2 encoder.
fn h264_encoder() -> &'static str {
    let out = Command::new(av::ffmpeg::ffmpeg_path())
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    ["libx264", "libopenh264"].into_iter().find(|e| out.contains(e)).unwrap_or("mpeg4")
}

fn have_ffmpeg() -> bool {
    let ok = av::ffmpeg::available();
    if !ok {
        eprintln!("ffmpeg not found: skipping");
    }
    ok
}

/// Lag (in samples) maximizing the cross-correlation of `a` against `b`.
fn best_lag(a: &[f32], b: &[f32], range: i64) -> i64 {
    let n = a.len().min(b.len()) as i64;
    (-range..=range)
        .max_by(|&x, &y| {
            let c = |l: i64| {
                (0..n).filter(|&i| (0..n).contains(&(i + l))).map(|i| a[i as usize] * b[(i + l) as usize]).sum::<f32>()
            };
            c(x).total_cmp(&c(y))
        })
        .unwrap()
}

#[test]
fn lossy_audio_formats_align_with_source() {
    if !have_ffmpeg() {
        return;
    }
    // Speech-like source: a chirp, so misalignment can't hide behind a
    // periodic signal.
    let rate = 44_100;
    let src: Vec<f32> = (0..rate * 4)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.3 * (2.0 * PI * (200.0 * t + 300.0 * t * t)).sin()
        })
        .collect();
    let wav = tmp("chirp.wav");
    write_wav(&wav, rate, &[src.clone(), src]);
    let reference = decode_all(&wav);
    let w = wav.to_str().unwrap();
    for (name, args) in [
        ("chirp.mp3", vec!["-c:a", "libmp3lame", "-b:a", "192k"]),
        ("chirp.m4a", vec!["-c:a", "aac", "-b:a", "192k"]),
        ("chirp.flac", vec!["-c:a", "flac"]),
        ("chirp_opus.webm", vec!["-c:a", "libopus", "-b:a", "128k"]),
    ] {
        let out = tmp(name);
        let mut a = vec!["-i", w];
        a.extend(args);
        a.push(out.to_str().unwrap());
        if !ffmpeg(&a) {
            eprintln!("cannot encode {name}: skipping");
            continue;
        }
        let got = decode_all(&out);
        let diff = (got.len() as f64 - reference.len() as f64).abs() / reference.len() as f64;
        assert!(diff < 0.005, "{name}: {} vs {} samples", got.len(), reference.len());
        let lag = best_lag(&got[8000..24_000], &reference[8000..24_000], 40);
        assert!(lag.abs() <= 1, "{name}: lag {lag}");
        let p = av::probe(&out).unwrap();
        assert!((p.duration_s - 4.0).abs() < 0.1 && p.has_audio && !p.has_video, "{name}: {p:?}");
    }
}

#[test]
fn cover_art_is_not_video_and_video_probe() {
    if !have_ffmpeg() {
        return;
    }
    let mp3 = tmp("cover.mp3");
    let ok = ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "sine=f=440:d=2",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=64x64:d=1",
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-frames:v",
        "1",
        "-c:a",
        "libmp3lame",
        "-c:v",
        "mjpeg",
        "-disposition:v",
        "attached_pic",
        mp3.to_str().unwrap(),
    ]);
    if ok {
        let p = av::probe(&mp3).unwrap();
        assert!(p.has_audio && !p.has_video, "{p:?}");
    }
    let mp4 = tmp("tone.mp4");
    if !ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x240:r=25:d=3",
        "-f",
        "lavfi",
        "-i",
        "sine=f=440:d=3",
        "-c:v",
        h264_encoder(),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        mp4.to_str().unwrap(),
    ]) {
        eprintln!("this ffmpeg cannot encode the fixture (decode-only build?); skipping");
        return;
    }
    let p = av::probe(&mp4).unwrap();
    assert!(p.has_audio && p.has_video && (p.duration_s - 3.0).abs() < 0.1, "{p:?}");
    assert!(decode_all(&mp4).len().abs_diff(48_000) < 200);
}

#[test]
fn frames_rotated_and_scaled() {
    if !have_ffmpeg() {
        return;
    }
    let src = tmp("frames.mp4");
    // Left half red, right half blue; a 1-frame white flash at 1.0 s marks
    // timing.
    if !ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=640x360:r=25:d=3",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x360:r=25:d=3",
        "-filter_complex",
        "[0][1]overlay=x=320,drawbox=enable='between(n,25,25)':c=white:t=fill",
        "-c:v",
        h264_encoder(),
        "-g",
        "50",
        "-pix_fmt",
        "yuv420p",
        src.to_str().unwrap(),
    ]) {
        eprintln!("this ffmpeg cannot encode the fixture (decode-only build?); skipping");
        return;
    }
    let rot = tmp("frames_rot90.mp4");
    if !ffmpeg(&["-display_rotation:v:0", "90", "-i", src.to_str().unwrap(), "-c", "copy", rot.to_str().unwrap()]) {
        eprintln!("this ffmpeg cannot encode the fixture (decode-only build?); skipping");
        return;
    }

    let f = av::frames_at(&src, &[0.5, 1.0, 1.02, 2.5], 1344, 0.0).unwrap();
    assert_eq!(f.len(), 4);
    assert_eq!((f[0].width, f[0].height), (640, 360), "never upscaled");
    assert!(near(px(&f[0], 10, 180), RED, 40) && near(px(&f[0], 630, 180), BLUE, 40));
    assert!(near(px(&f[1], 10, 180), WHITE, 40), "frame displayed at 1.0 s is the flash");
    assert!(near(px(&f[2], 10, 180), WHITE, 40), "still displayed at 1.02 s (frame lasts 40 ms)");
    assert!(near(px(&f[3], 10, 180), RED, 40));

    let small = av::frames_at(&src, &[0.5], 448, 0.0).unwrap();
    // AVFoundation sizing: floor(360 · 0.35f) = 251.
    assert_eq!((small[0].width, small[0].height), (448, 251));

    // ffmpeg's display matrix rotation 90 is counter-clockwise: the right
    // (blue) half ends up on top.
    let r = av::frames_at(&rot, &[0.5, 2.0], 448, 0.5).unwrap();
    assert_eq!((r[0].width, r[0].height), (251, 448));
    assert!(near(px(&r[0], 125, 10), BLUE, 40) && near(px(&r[0], 125, 438), RED, 40));

    // Sparse times take the per-frame seek path; tolerance permits keyframes.
    let sparse = av::frames_at(&src, &[0.1, 2.9], 320, 1.5).unwrap();
    assert_eq!(sparse.len(), 2);
    assert_eq!((sparse[0].width, sparse[0].height), (320, 180));
    assert!(av::frames_at(&src, &[1.0], 0, 0.0).is_err());

    // Past the end: the last picture, in both plans.
    let late = av::frames_at(&src, &[10.0], 320, 0.0).unwrap();
    assert!(near(px(&late[0], 10, 90), RED, 40));
    let late = av::frames_at(&src, &[2.0, 2.5, 3.5, 9.0], 320, 0.0).unwrap();
    assert_eq!(late.len(), 4);
    // Unsorted, duplicated times come back in request order.
    let f = av::frames_at(&src, &[1.0, 0.2, 1.0], 320, 0.0).unwrap();
    assert!(
        near(px(&f[0], 10, 90), WHITE, 40) && near(px(&f[1], 10, 90), RED, 40) && near(px(&f[2], 10, 90), WHITE, 40)
    );
}
