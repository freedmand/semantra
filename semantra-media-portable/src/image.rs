//! Still images, portably: the `image` crate's decoders in place of ImageIO.
//!
//! Same contract as semantra-embed's macOS `media::image::{decode, decode_max,
//! encode_jpeg}`: the first frame/page of the file, EXIF orientation applied,
//! converted to sRGB (embedded ICC profiles honored), transparency composited
//! over white, and the longer side capped at `max_side` — downscale only, like
//! ImageIO's thumbnail-with-transform, which never upscales.
//!
//! Formats: JPEG, PNG, GIF, WebP, TIFF, BMP, at any bit depth and color type
//! (16-bit, grayscale, palette, CMYK JPEG). HEIC/HEIF and AVIF decode through
//! the ffmpeg sidecar (see `av::ffmpeg`). Not available off macOS: camera
//! RAW (ImageIO-only).
//!
//! Big JPEGs are decoded subsampled in the DCT domain (1/2, 1/4, 1/8) to the
//! smallest scale still at least `max_side`, as ImageIO does, so a 48 MP photo
//! never materializes at full size.

use std::io::Cursor;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, Limits};

/// Longest side images are decoded at by [`decode`] (matches
/// semantra-embed's `DECODE_MAX_SIDE`).
pub const DECODE_MAX_SIDE: u32 = 2048;

/// Decoder memory ceiling. The `image` default (512 MiB) rejects e.g. a
/// 12k × 12k 16-bit RGBA TIFF, which ImageIO opens fine.
const MAX_ALLOC: u64 = 2 << 30;

/// Packed 8-bit RGB pixels.
#[derive(Clone)]
pub struct Rgb8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Decode the first image in `path`, oriented and in sRGB, with transparency
/// composited over white (as the reference processor's `convert_rgb` does).
pub fn decode(path: &Path) -> Result<Rgb8> {
    decode_max(path, DECODE_MAX_SIDE)
}

/// [`decode`] with the longer side capped at `max_side` (e.g. thumbnails).
///
/// HEIC/HEIF and AVIF (iPhone photos; no pure-Rust decoder) go to the ffmpeg
/// sidecar, which handles their tile grids and `irot`/`imir` orientation.
pub fn decode_max(path: &Path, max_side: u32) -> Result<Rgb8> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot open {}", path.display()))?;
    match decode_bytes(&bytes, max_side) {
        Err(e) if is_heif(&bytes) => {
            decode_heif(path, &bytes, max_side).with_context(|| format!("cannot decode {} ({e})", path.display()))
        }
        r => r.with_context(|| format!("cannot decode {}", path.display())),
    }
}

/// HEIF/AVIF through the ffmpeg sidecar, then color-managed from the file's
/// `colr` box (ICC profile, or CICP Display P3) and downscaled. ImageIO sizes
/// HEIC thumbnails with the short side rounded to an *even* number (1600x972
/// at 512 is 512x312, where its JPEG path gives 512x311; 6400x3888 at 2048
/// is 2048x1244), so this path does too.
fn decode_heif(path: &Path, bytes: &[u8], max_side: u32) -> Result<Rgb8> {
    let mut img = crate::av::ffmpeg::still(path)?;
    let profile = match heif_colr(bytes) {
        Some(Colr::Icc(icc)) => moxcms::ColorProfile::new_from_slice(icc).ok(),
        Some(Colr::Cicp { primaries: 12, .. }) => Some(moxcms::ColorProfile::new_display_p3()),
        _ => None,
    };
    if let Some(p) = profile.filter(|p| p.color_space == moxcms::DataColorSpace::Rgb && !is_srgb(p)) {
        let srgb = moxcms::ColorProfile::new_srgb();
        let rgb = moxcms::Layout::Rgb;
        if let Ok(t) = p.create_transform_8bit(rgb, &srgb, rgb, moxcms::TransformOptions::default()) {
            let src = img.data.clone();
            let _ = moxcms::TransformExecutor::transform(t.as_ref(), &src, &mut img.data);
        }
    }
    let (w, h) = (img.width, img.height);
    let long = w.max(h);
    if long <= max_side {
        return Ok(img);
    }
    let short = |v: u32| (((v as f64 * max_side as f64 / long as f64 / 2.0).round() * 2.0) as u32).clamp(2, max_side);
    let (tw, th) = if w >= h { (max_side, short(h)) } else { (short(w), max_side) };
    resize(&img, tw, th)
}

enum Colr<'a> {
    Icc(&'a [u8]),
    Cicp { primaries: u16 },
}

/// The first `colr` property box in a HEIF's header (`ipco`): an ICC
/// profile (`prof`/`rICC`) or CICP code points (`nclx`). Found by scanning:
/// the box tree up to it is small and only needed for this.
fn heif_colr(bytes: &[u8]) -> Option<Colr<'_>> {
    let head = &bytes[..bytes.len().min(1 << 20)];
    let at = head.windows(4).position(|w| w == b"colr")?;
    let size = u32::from_be_bytes(head.get(at.checked_sub(4)?..at)?.try_into().ok()?) as usize;
    let body = head.get(at + 4..(at - 4).checked_add(size)?)?;
    match body.get(..4)? {
        b"prof" | b"rICC" => Some(Colr::Icc(&body[4..])),
        b"nclx" => Some(Colr::Cicp { primaries: u16::from_be_bytes(body.get(4..6)?.try_into().ok()?) }),
        _ => None,
    }
}

/// ISO-BMFF still-image brands: HEIC/HEIF (`heic`, `heix`, `mif1`, …) and
/// AVIF.
fn is_heif(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && &bytes[4..8] == b"ftyp"
        && matches!(
            &bytes[8..12],
            b"heic" | b"heix" | b"heim" | b"heis" | b"hevc" | b"hevx" | b"mif1" | b"msf1" | b"avif" | b"avis"
        )
}

/// [`decode_max`] on an in-memory file.
pub fn decode_bytes(bytes: &[u8], max_side: u32) -> Result<Rgb8> {
    if max_side == 0 {
        bail!("max_side must be positive");
    }
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_ALLOC);
    reader.limits(limits);
    let format = reader.format();
    let mut decoder = reader.into_decoder()?;
    // Both must be read before `from_decoder` consumes the decoder. A broken
    // EXIF block shouldn't sink the image: ImageIO ignores it too.
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let icc = decoder.icc_profile().ok().flatten();

    let (img, icc) = match format {
        Some(image::ImageFormat::Jpeg) => match decode_jpeg(bytes, max_side, icc.as_deref()) {
            // CMYK comes back already converted to sRGB.
            Some(Jpeg { img, srgb: true }) => (img, None),
            Some(Jpeg { img, srgb: false }) => (img, icc),
            None => (DynamicImage::from_decoder(decoder)?, icc),
        },
        _ => (DynamicImage::from_decoder(decoder)?, icc),
    };
    if img.width() == 0 || img.height() == 0 {
        bail!("empty image");
    }
    let rgb = to_srgb8_over_white(img, icc.as_deref());
    let rgb = fit(rgb, max_side)?;
    Ok(orient(rgb, orientation))
}

struct Jpeg {
    img: DynamicImage,
    /// Already converted to sRGB (CMYK through its ICC profile).
    srgb: bool,
}

/// JPEG special cases, via jpeg-decoder; `None` means "use the regular
/// decoder" (zune-jpeg, faster at full size).
///
/// - Much larger than `max_side`: decode at 1/2, 1/4 or 1/8 scale in the DCT
///   domain — the smallest scale whose longer side is still >= `max_side`, so
///   the final resize remains a true downscale.
/// - CMYK/YCCK (print workflows, stock photos): convert through the embedded
///   CMYK profile like ImageIO does; zune-jpeg's naive CMYK -> RGB is ~23 dB
///   off.
fn decode_jpeg(bytes: &[u8], max_side: u32, icc: Option<&[u8]>) -> Option<Jpeg> {
    let mut dec = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    dec.read_info().ok()?;
    let info = dec.info()?;
    let long = info.width.max(info.height) as u32;
    let cmyk = info.pixel_format == jpeg_decoder::PixelFormat::CMYK32;
    let scaled = long >= max_side * 2;
    if !scaled && !cmyk {
        return None;
    }
    let (w, h) = if scaled {
        // jpeg-decoder picks the smallest 1/2^k scale covering the request.
        let req = |side: u16| ((side as u64 * max_side as u64).div_ceil(long as u64)) as u16;
        dec.scale(req(info.width), req(info.height)).ok()?
    } else {
        (info.width, info.height)
    };
    let px = dec.decode().ok()?;
    let (w, h) = (w as u32, h as u32);
    use jpeg_decoder::PixelFormat as F;
    let img = match dec.info()?.pixel_format {
        F::RGB24 => image::RgbImage::from_raw(w, h, px).map(DynamicImage::ImageRgb8),
        F::L8 => image::GrayImage::from_raw(w, h, px).map(DynamicImage::ImageLuma8),
        F::L16 => {
            let v = px.as_chunks::<2>().0.iter().map(|b| u16::from_be_bytes(*b)).collect();
            image::ImageBuffer::from_raw(w, h, v).map(DynamicImage::ImageLuma16)
        }
        F::CMYK32 => {
            let rgb = cmyk_to_srgb(&px, icc);
            return image::RgbImage::from_raw(w, h, rgb).map(|i| Jpeg { img: DynamicImage::ImageRgb8(i), srgb: true });
        }
    }?;
    Some(Jpeg { img, srgb: false })
}

/// CMYK (0 = no ink) -> sRGB through the image's CMYK profile, or the naive
/// `(1 - c)(1 - k)` without one.
fn cmyk_to_srgb(cmyk: &[u8], icc: Option<&[u8]>) -> Vec<u8> {
    let profile = icc
        .and_then(|b| moxcms::ColorProfile::new_from_slice(b).ok())
        .filter(|p| p.color_space == moxcms::DataColorSpace::Cmyk);
    if let Some(p) = profile {
        let srgb = moxcms::ColorProfile::new_srgb();
        // moxcms takes 4-channel input through the `Rgba` layout.
        let opts =
            moxcms::TransformOptions { rendering_intent: moxcms::RenderingIntent::Perceptual, ..Default::default() };
        if let Ok(t) = p.create_transform_8bit(moxcms::Layout::Rgba, &srgb, moxcms::Layout::Rgb, opts) {
            let mut rgb = vec![0u8; cmyk.len() / 4 * 3];
            if moxcms::TransformExecutor::transform(t.as_ref(), cmyk, &mut rgb).is_ok() {
                return rgb;
            }
        }
    }
    cmyk.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            let k = 255 - p[3] as u32;
            let ch = |c: u8| (((255 - c as u32) * k + 127) / 255) as u8;
            [ch(p[0]), ch(p[1]), ch(p[2])]
        })
        .collect()
}

/// Any decoded image -> sRGB 8-bit RGB, alpha composited over white.
///
/// Compositing happens on gamma-encoded sRGB values, as CoreGraphics does when
/// drawing into an 8-bit sRGB bitmap pre-filled with white.
fn to_srgb8_over_white(img: DynamicImage, icc: Option<&[u8]>) -> Rgb8 {
    let (width, height) = (img.width(), img.height());
    let has_alpha = img.color().has_alpha();
    let gray = !img.color().has_color();
    let profile = icc.and_then(|b| moxcms::ColorProfile::new_from_slice(b).ok());
    // Grayscale profiles describe one channel; applying them to the expanded
    // RGB is wrong, and the difference from sRGB's curve is small. Skip them,
    // and skip profiles that already are sRGB (most JPEGs from the web).
    let profile = profile.filter(|p| !gray && p.color_space == moxcms::DataColorSpace::Rgb && !is_srgb(p));

    let (mut rgba, mut rgb) =
        if has_alpha { (img.into_rgba8().into_raw(), Vec::new()) } else { (Vec::new(), img.into_rgb8().into_raw()) };
    if let Some(p) = profile {
        let srgb = moxcms::ColorProfile::new_srgb();
        let layout = if has_alpha { moxcms::Layout::Rgba } else { moxcms::Layout::Rgb };
        // An unsupported profile (e.g. exotic LUT types) leaves the pixels
        // as-is: wrong-ish colors beat a failed decode.
        if let Ok(t) = p.create_transform_8bit(layout, &srgb, layout, moxcms::TransformOptions::default()) {
            let buf = if has_alpha { &mut rgba } else { &mut rgb };
            let src = buf.clone();
            let _ = moxcms::TransformExecutor::transform(t.as_ref(), &src, buf);
        }
    }
    let data = if has_alpha {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| {
                let a = p[3] as u32;
                let over = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
                [over(p[0]), over(p[1]), over(p[2])]
            })
            .collect()
    } else {
        rgb
    };
    Rgb8 { width, height, data }
}

/// Whether an ICC profile is (close enough to) sRGB that converting is a no-op.
fn is_srgb(p: &moxcms::ColorProfile) -> bool {
    let s = moxcms::ColorProfile::new_srgb();
    let close = |a: moxcms::Xyzd, b: moxcms::Xyzd| (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs() < 2e-3;
    close(p.red_colorant, s.red_colorant)
        && close(p.green_colorant, s.green_colorant)
        && close(p.blue_colorant, s.blue_colorant)
        && p.red_trc.is_some()
        && format!("{:?}", p.red_trc) == format!("{:?}", s.red_trc)
}

/// Thumbnail size for a `w × h` image capped at `max_side`: the longer side
/// becomes `max_side`, the shorter one is scaled and rounded (never below 1).
/// Images already within the cap keep their size.
pub fn fit_size(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= max_side {
        return (w, h);
    }
    let scale = |s: u32| ((s as f64 * max_side as f64 / long as f64).round() as u32).clamp(1, max_side);
    if w >= h {
        (max_side, scale(h))
    } else {
        (scale(w), max_side)
    }
}

/// Downscale (never upscale) so the longer side is at most `max_side`.
fn fit(img: Rgb8, max_side: u32) -> Result<Rgb8> {
    let (tw, th) = fit_size(img.width, img.height, max_side);
    if (tw, th) == (img.width, img.height) {
        return Ok(img);
    }
    resize(&img, tw, th)
}

/// Antialiased resize. Catmull-Rom measured closest to ImageIO's thumbnail
/// scaler of the 10 filters tried (parity `filters`).
pub(crate) fn resize(img: &Rgb8, tw: u32, th: u32) -> Result<Rgb8> {
    let src = Image::from_vec_u8(img.width, img.height, img.data.clone(), PixelType::U8x3)?;
    let mut dst = Image::new(tw, th, PixelType::U8x3);
    Resizer::new().resize(&src, &mut dst, &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(RESIZE_FILTER)))?;
    Ok(Rgb8 { width: tw, height: th, data: dst.into_vec() })
}

pub(crate) const RESIZE_FILTER: FilterType = FilterType::CatmullRom;

/// Apply an EXIF orientation to packed RGB.
fn orient(img: Rgb8, o: Orientation) -> Rgb8 {
    if o == Orientation::NoTransforms {
        return img;
    }
    let Some(buf) = image::RgbImage::from_raw(img.width, img.height, img.data) else {
        unreachable!("Rgb8 buffer matches its dimensions")
    };
    let mut d = DynamicImage::ImageRgb8(buf);
    d.apply_orientation(o);
    let (width, height) = (d.width(), d.height());
    Rgb8 { width, height, data: d.into_rgb8().into_raw() }
}

/// Encode RGB pixels as a JPEG (`quality` in 0–1), for previews/thumbnails.
pub fn encode_jpeg(img: &Rgb8, quality: f64) -> Result<Vec<u8>> {
    if img.data.len() != img.width as usize * img.height as usize * 3 {
        bail!("RGB buffer does not match {}x{}", img.width, img.height);
    }
    let q = (quality * 100.0).round().clamp(1.0, 100.0) as u8;
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q)
        .encode(&img.data, img.width, img.height, image::ExtendedColorType::Rgb8)
        .map_err(|e| anyhow!("JPEG encoding failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_size_downscales_only() {
        assert_eq!(fit_size(4000, 3000, 2048), (2048, 1536));
        assert_eq!(fit_size(3000, 4000, 2048), (1536, 2048));
        assert_eq!(fit_size(800, 600, 2048), (800, 600));
        assert_eq!(fit_size(10000, 3, 100), (100, 1));
    }

    #[test]
    fn jpeg_roundtrip() {
        let img = Rgb8 { width: 16, height: 8, data: (0..16 * 8 * 3).map(|i| (i % 251) as u8).collect() };
        let jpeg = encode_jpeg(&img, 0.9).unwrap();
        let back = decode_bytes(&jpeg, 2048).unwrap();
        assert_eq!((back.width, back.height), (16, 8));
    }
}
