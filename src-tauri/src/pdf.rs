//! PDF text extraction via PDFium, deliberately Tauri-agnostic (like `embed.rs`
//! and `store.rs`).
//!
//! PDFium is loaded as a **bundled dynamic library** (fetched per-target by
//! `build.rs`, shipped as a Tauri resource) and bound at runtime — we never
//! statically link it and never render in Rust (PDF pages are rendered by
//! PDF.js in the webview); here we only pull **text** and per-character
//! **bounding boxes** for highlighting. Because `pdfium-render` keeps a
//! process-global binding (and the underlying C library is a singleton),
//! exactly one [`Pdfium`] instance may exist per process: [`load_library`] is
//! called once at startup and the instance is reused for every document.

use std::path::Path;

use pdfium_render::prelude::{PdfRenderConfig, Pdfium, PdfiumError};
use semantra_embed::media::image::Rgb8;

/// Serializes every PDFium call in the process. PDFium is not re-entrant:
/// concurrent use from several threads (indexing renders pages while the UI
/// asks for highlights, thumbnails or page text) corrupts its global state —
/// observed as spurious `FormatError`s on valid files, and as outright crashes.
/// pdfium-render's `thread_safe` feature does not cover this, so each helper
/// below holds this lock for the whole open-document-and-read operation.
static PDFIUM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn pdfium_lock() -> std::sync::MutexGuard<'static, ()> {
    // A panic while holding the lock leaves PDFium usable; ignore poisoning.
    PDFIUM_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Re-exported so callers (and integration tests) can name the shared PDFium
/// handle type without depending on `pdfium-render` directly.
pub use pdfium_render::prelude::Pdfium as PdfiumLib;

/// Whether the bundled PDFium library exists in `lib_dir` (checks the correct
/// platform-specific filename). Lets callers pick between a resource dir and a
/// dev-tree fallback without depending on `pdfium-render` types.
pub fn library_present(lib_dir: &Path) -> bool {
    Pdfium::pdfium_platform_library_name_at_path(lib_dir).exists()
}

/// Bind to the bundled PDFium dynamic library living in `lib_dir`.
///
/// `lib_dir` is the *directory* containing the library; the correct
/// platform-specific filename (`libpdfium.dylib` / `libpdfium.so` /
/// `pdfium.dll`) is appended automatically. Must be called **once** per process
/// — `pdfium-render` panics if a second [`Pdfium`] is constructed.
pub fn load_library(lib_dir: &Path) -> Result<Pdfium, String> {
    let lib_path = Pdfium::pdfium_platform_library_name_at_path(lib_dir);
    let bindings = Pdfium::bind_to_library(&lib_path).map_err(|e| {
        format!("failed to load PDFium library at {}: {e}", lib_path.display())
    })?;
    Ok(Pdfium::new(bindings))
}

/// Extract the verbatim text of every page in `pdf_path`, in document order.
///
/// Pages are processed one at a time and released before the next is loaded, so
/// memory stays flat regardless of document size. The returned `Vec` has one
/// entry per page (empty string for a page with no extractable text, e.g. a
/// scanned image-only page — there is no OCR). The chunker downstream turns
/// these into windows; keeping raw per-page text here lets each chunk record
/// which page it came from.
pub fn extract_pdf_pages(pdfium: &Pdfium, pdf_path: &str) -> Result<Vec<String>, String> {
    let _guard = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| format!("Couldn't open this PDF: {}", describe(e)))?;

    let pages = document.pages();
    let mut out = Vec::with_capacity(pages.len() as usize);
    for (i, page) in pages.iter().enumerate() {
        let text = page
            .text()
            .map_err(|e| format!("read text from page {}: {}", i + 1, describe(e)))?;
        // Build the page string from the per-character iteration (NOT `text.all()`)
        // so a chunk's page-relative char offset indexes the very same character
        // sequence `page_chars` later returns boxes for. This 1:1 alignment is what
        // makes highlight overlays land on the right glyphs.
        out.push(page_string_from_chars(&text));
        // `page`/`text` drop here, releasing native resources before the next.
    }
    Ok(out)
}

/// The canonical page string: one `char` per PDFium text character, in order.
/// Shared by extraction (for chunk offsets) and [`page_chars`] (for box
/// indexing) so the two are guaranteed to agree.
fn page_string_from_chars(text: &pdfium_render::prelude::PdfPageText) -> String {
    text.chars()
        .iter()
        .map(|c| c.unicode_char().unwrap_or('\u{FFFD}'))
        .collect()
}

/// A single character's bounding box on a page, in **PDF user-space points**
/// (origin bottom-left, the coordinate space PDF.js viewports consume directly).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct CharBox {
    pub left: f32,
    pub bottom: f32,
    pub right: f32,
    pub top: f32,
}

/// The text and per-character boxes of one page, plus the page's dimensions in
/// points. `chars[i]` is the box of the i-th character of `text` (by char, not
/// byte) — the indexing the highlighter maps chunk offsets onto.
pub struct PageChars {
    pub width: f32,
    pub height: f32,
    pub text: String,
    pub chars: Vec<CharBox>,
}

/// Read one page's text and per-character boxes. `page_index` is 0-based.
///
/// Used on demand by the highlight command: a chunk's page-relative char range
/// selects a slice of `chars`, whose union (per text line) becomes the highlight
/// rectangles overlaid on the PDF.js-rendered page.
pub fn page_chars(pdfium: &Pdfium, pdf_path: &str, page_index: usize) -> Result<PageChars, String> {
    let _guard = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| format!("Couldn't open this PDF: {}", describe(e)))?;
    let pages = document.pages();
    let page = pages
        .get(page_index as i32)
        .map_err(|e| format!("load page {}: {}", page_index + 1, describe(e)))?;
    let width = page.width().value;
    let height = page.height().value;
    let text = page
        .text()
        .map_err(|e| format!("read text from page {}: {}", page_index + 1, describe(e)))?;

    let mut chars = Vec::new();
    let mut s = String::new();
    for ch in text.chars().iter() {
        // `tight_bounds` hugs the glyph; fall back to a zero box for chars
        // without geometry (e.g. some control chars) so indexing stays aligned
        // 1:1 with the page text.
        let (left, bottom, right, top) = match ch.tight_bounds() {
            Ok(r) => (r.left().value, r.bottom().value, r.right().value, r.top().value),
            Err(_) => (0.0, 0.0, 0.0, 0.0),
        };
        chars.push(CharBox { left, bottom, right, top });
        s.push(ch.unicode_char().unwrap_or('\u{FFFD}'));
    }

    Ok(PageChars {
        width,
        height,
        text: s, // built char-by-char above, aligned 1:1 with `chars`
        chars,
    })
}

/// A user-facing description of a PDFium error (shown in the manage view, so
/// no internal paths), with the raw variant kept for bug reports.
fn describe(e: PdfiumError) -> String {
    use pdfium_render::prelude::PdfiumInternalError as I;
    let raw = format!("{e:?}");
    let friendly = match &e {
        PdfiumError::PdfiumLibraryInternalError(I::PasswordError) => "it is password-protected",
        PdfiumError::PdfiumLibraryInternalError(I::FormatError) => "the file is damaged or not a valid PDF",
        PdfiumError::PdfiumLibraryInternalError(I::FileError) => "the file could not be read",
        PdfiumError::PdfiumLibraryInternalError(I::SecurityError) => "it uses unsupported security settings",
        _ => return raw,
    };
    format!("{friendly} ({raw})")
}

/// Render page `page_index` to RGB with its longer side at `max_side` pixels.
pub fn render_page(pdfium: &Pdfium, pdf_path: &str, page_index: usize, max_side: u32) -> Result<Rgb8, String> {
    render_pages(pdfium, pdf_path, &[page_index], max_side)?
        .pop()
        .ok_or_else(|| format!("render page {}: no output", page_index + 1))
}

/// Render several pages with the document opened once (opening a large PDF
/// costs ~100+ ms, so per-page reopening dominates indexing otherwise). Each
/// page is scaled so its longer side is `max_side` pixels — up or down;
/// PDFium's default is 72 dpi (a letter page would be only 612×792).
pub fn render_pages(pdfium: &Pdfium, pdf_path: &str, pages: &[usize], max_side: u32) -> Result<Vec<Rgb8>, String> {
    let _guard = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| format!("Couldn't open this PDF: {}", describe(e)))?;
    let all = document.pages();
    pages
        .iter()
        .map(|&page_index| {
            let page = all
                .get(page_index as i32)
                .map_err(|e| format!("load page {}: {}", page_index + 1, describe(e)))?;
            let longest = page.width().value.max(page.height().value).max(1.0);
            let config = PdfRenderConfig::new().scale_page_by_factor(max_side as f32 / longest);
            let bitmap = page
                .render_with_config(&config)
                .map_err(|e| format!("render page {}: {}", page_index + 1, describe(e)))?;
            let (width, height) = (bitmap.width() as u32, bitmap.height() as u32);
            let data = bitmap.as_rgba_bytes().chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            Ok(Rgb8 { width, height, data })
        })
        .collect()
}
