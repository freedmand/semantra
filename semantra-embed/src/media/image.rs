//! Still images: decode with ImageIO, resize like the reference processor,
//! and stack into [`PatchGrid`]s for the vision tower.
//!
//! ImageIO opens everything Preview does (JPEG, PNG, HEIC/HEIF, WebP, TIFF,
//! GIF, BMP, camera RAW, …), applies EXIF orientation, and color-matches into
//! sRGB. Large photos are decoded subsampled (DCT-domain for JPEG/HEIC) to
//! [`DECODE_MAX_SIDE`], so a 48 MP photo never materializes at full size.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use mlx_rs::Array;
use objc2_core_foundation::{
    CFBoolean, CFDictionary, CFMutableData, CFNumber, CFRetained, CFString, CFType, CFURL, CGPoint, CGRect, CGSize,
};
use objc2_core_graphics::{
    kCGColorSpaceSRGB, CGBitmapContextCreate, CGBitmapContextCreateImage, CGColorSpace, CGContext, CGImage,
    CGImageAlphaInfo,
};
use objc2_image_io::{
    kCGImageDestinationLossyCompressionQuality, kCGImageSourceCreateThumbnailFromImageAlways,
    kCGImageSourceCreateThumbnailWithTransform, kCGImageSourceThumbnailMaxPixelSize, CGImageDestination,
    CGImageSource,
};

use crate::vision::PatchGrid;

/// Longest side images are decoded at before the final resize. Comfortably
/// above the largest resize target (~1.3k px at 280 soft tokens), so the final
/// Catmull-Rom pass is always a genuine, antialiased downscale.
pub const DECODE_MAX_SIDE: u32 = 2048;

/// Soft-token budget per still image / PDF page (the model's default).
pub const IMAGE_SOFT_TOKENS: u32 = 280;
/// Soft-token budget per video frame (the model's default video budget).
pub const FRAME_SOFT_TOKENS: u32 = 140;

const PATCH: u32 = 16;
const POOL: u32 = 3;

/// Packed 8-bit RGB pixels.
#[derive(Clone)]
pub struct Rgb8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// One image resized for the vision tower: `(rows·16) × (cols·16)` 8-bit RGB,
/// row-major HWC. Kept as bytes (4x smaller than F32) until [`stack`] uploads
/// it; scaling to [0, 1] happens on the GPU.
pub struct Prepared {
    pub rows: u32,
    pub cols: u32,
    pub pixels: Vec<u8>,
}

/// Decode the first image in `path`, oriented and in sRGB, with transparency
/// composited over white (as the reference processor's `convert_rgb` does).
pub fn decode(path: &Path) -> Result<Rgb8> {
    decode_max(path, DECODE_MAX_SIDE)
}

/// [`decode`] with the longer side capped at `max_side` (e.g. thumbnails).
pub fn decode_max(path: &Path, max_side: u32) -> Result<Rgb8> {
    let url = CFURL::from_file_path(path).ok_or_else(|| anyhow!("bad path {}", path.display()))?;
    let src = unsafe { CGImageSource::with_url(&url, None) }
        .ok_or_else(|| anyhow!("ImageIO cannot open {}", path.display()))?;
    let max = CFNumber::new_i32(max_side as i32);
    let (k_always, k_transform, k_max) = unsafe {
        (
            kCGImageSourceCreateThumbnailFromImageAlways,
            kCGImageSourceCreateThumbnailWithTransform,
            kCGImageSourceThumbnailMaxPixelSize,
        )
    };
    let keys: [&CFString; 3] = [k_always, k_transform, k_max];
    let values: [&CFType; 3] = [CFBoolean::new(true).as_ref(), CFBoolean::new(true).as_ref(), max.as_ref()];
    let opts = CFDictionary::<CFString, CFType>::from_slices(&keys, &values);
    let image = unsafe { src.thumbnail_at_index(0, Some(opts.as_opaque())) }
        .ok_or_else(|| anyhow!("ImageIO cannot decode {}", path.display()))?;
    rgb_from_cgimage(&image)
}

/// Draw a CGImage into an opaque sRGB RGBA8 bitmap (white background) and
/// drop the alpha channel.
pub fn rgb_from_cgimage(image: &CGImage) -> Result<Rgb8> {
    let (w, h) = (CGImage::width(Some(image)), CGImage::height(Some(image)));
    if w == 0 || h == 0 {
        bail!("empty image");
    }
    let mut rgba = vec![0u8; w * h * 4];
    let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
        .ok_or_else(|| anyhow!("no sRGB color space"))?;
    let ctx: CFRetained<CGContext> = unsafe {
        CGBitmapContextCreate(
            rgba.as_mut_ptr().cast(),
            w,
            h,
            8,
            w * 4,
            Some(&space),
            CGImageAlphaInfo::PremultipliedLast.0,
        )
    }
    .ok_or_else(|| anyhow!("cannot create bitmap context"))?;
    let rect = CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(w as f64, h as f64));
    CGContext::set_rgb_fill_color(Some(&ctx), 1.0, 1.0, 1.0, 1.0);
    CGContext::fill_rect(Some(&ctx), rect);
    CGContext::draw_image(Some(&ctx), rect, Some(image));
    drop(ctx);
    let data = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    Ok(Rgb8 { width: w as u32, height: h as u32, data })
}

/// Encode RGB pixels as a JPEG (`quality` in 0–1), for previews/thumbnails.
pub fn encode_jpeg(img: &Rgb8, quality: f64) -> Result<Vec<u8>> {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut rgba: Vec<u8> = img.data.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
        .ok_or_else(|| anyhow!("no sRGB color space"))?;
    let ctx = unsafe {
        CGBitmapContextCreate(
            rgba.as_mut_ptr().cast(),
            w,
            h,
            8,
            w * 4,
            Some(&space),
            CGImageAlphaInfo::NoneSkipLast.0,
        )
    }
    .ok_or_else(|| anyhow!("cannot create bitmap context"))?;
    let image = CGBitmapContextCreateImage(Some(&ctx)).ok_or_else(|| anyhow!("cannot snapshot bitmap"))?;
    let data = CFMutableData::new(None, 0).ok_or_else(|| anyhow!("cannot allocate CFData"))?;
    let jpeg = CFString::from_static_str("public.jpeg");
    let dest = unsafe { CGImageDestination::with_data(&data, &jpeg, 1, None) }
        .ok_or_else(|| anyhow!("cannot create JPEG encoder"))?;
    let q = CFNumber::new_f64(quality);
    let keys: [&CFString; 1] = [unsafe { kCGImageDestinationLossyCompressionQuality }];
    let values: [&CFType; 1] = [q.as_ref()];
    let props = CFDictionary::<CFString, CFType>::from_slices(&keys, &values);
    unsafe {
        dest.add_image(&image, Some(props.as_opaque()));
        if !dest.finalize() {
            bail!("JPEG encoding failed");
        }
    }
    Ok(data.to_vec())
}

/// Largest `(height, width)` that keeps the aspect ratio, fits
/// `soft_tokens × 9` patches, and has both sides divisible by 48 px — the
/// reference processor's `get_aspect_ratio_preserving_size`, verbatim.
pub fn target_size(height: u32, width: u32, soft_tokens: u32) -> Result<(u32, u32)> {
    let max_patches = soft_tokens * POOL * POOL;
    let side = POOL * PATCH;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / (height as f64 * width as f64)).sqrt();
    let mut th = ((factor * height as f64) / side as f64).floor() as u32 * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as u32 * side;
    if th == 0 && tw == 0 {
        bail!("image too small to resize ({width}x{height})");
    }
    let max_side = (max_patches / (POOL * POOL)) * side;
    if th == 0 {
        th = side;
        tw = ((width / height) * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height / width) * side).min(max_side);
    }
    Ok((th, tw))
}

/// Resize `img` for the vision tower at `soft_tokens` budget (Catmull-Rom,
/// i.e. antialiased bicubic a = -0.5 — what the reference's torchvision
/// `resize(antialias=True)` computes).
pub fn prepare(img: &Rgb8, soft_tokens: u32) -> Result<Prepared> {
    let (th, tw) = target_size(img.height, img.width, soft_tokens)?;
    let rgb = if (th, tw) == (img.height, img.width) {
        img.data.clone()
    } else {
        let src = Image::from_vec_u8(img.width, img.height, img.data.clone(), PixelType::U8x3)?;
        let mut dst = Image::new(tw, th, PixelType::U8x3);
        Resizer::new().resize(
            &src,
            &mut dst,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
        )?;
        dst.into_vec()
    };
    Ok(Prepared {
        rows: th / PATCH,
        cols: tw / PATCH,
        pixels: rgb,
    })
}

/// Stack same-grid images into one batch. All inputs must share `rows`/`cols`
/// (group by [`Prepared::grid`] first).
pub fn stack(images: &[&Prepared]) -> Result<PatchGrid> {
    let first = images.first().ok_or_else(|| anyhow!("no images to stack"))?;
    if images.iter().any(|p| p.grid() != first.grid()) {
        bail!("cannot stack images with different patch grids");
    }
    let (h, w) = ((first.rows * PATCH) as i32, (first.cols * PATCH) as i32);
    let data: Vec<u8> = images.iter().flat_map(|p| p.pixels.iter().copied()).collect();
    let pixels = Array::from_slice(&data, &[images.len() as i32, h, w, 3])
        .as_dtype(mlx_rs::Dtype::Float32)?
        .divide(Array::from_f32(255.0))?;
    Ok(PatchGrid {
        pixels,
        rows: first.rows as i32,
        cols: first.cols as i32,
    })
}

impl Prepared {
    /// `(rows, cols)` patch grid — the batching key.
    pub fn grid(&self) -> (u32, u32) {
        (self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_matches_reference() {
        // Values from the reference processor (Python) at 280 soft tokens.
        assert_eq!(target_size(972, 1600, 280).unwrap(), (624, 1008)); // 39x63 patches
        assert_eq!(target_size(720, 1280, 140).unwrap(), (384, 720)); // 24x45 patches
        let (h, w) = target_size(4000, 3000, 280).unwrap();
        assert!(h % 48 == 0 && w % 48 == 0 && (h / 16) * (w / 16) <= 2520);
        // Extreme panorama: the short side clamps to one 48 px band.
        assert_eq!(target_size(10, 4000, 280).unwrap().0, 48);
    }
}
