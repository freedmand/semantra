//! File → canonical text + segments.
//!
//! The single entry point [`extract`] turns a path into an [`Extracted`]: the
//! file's kind, its canonical full text, and the ordered [`Segment`]s a
//! [`Chunker`](crate::chunk::Chunker) consumes. PDFs become one segment per page
//! (so chunks stay page-local and highlightable); plain text becomes a single
//! flat segment. New file types add an arm here without touching chunking,
//! embedding, or storage.

use pdfium_render::prelude::Pdfium;

use crate::chunk::Segment;
use crate::pdf;

/// Kind of an indexed file, as understood by the reader UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileType {
    Text,
    Pdf,
}

impl FileType {
    pub fn as_str(self) -> &'static str {
        match self {
            FileType::Text => "text",
            FileType::Pdf => "pdf",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "text" => Ok(FileType::Text),
            "pdf" => Ok(FileType::Pdf),
            other => Err(format!("unknown file type {other:?}")),
        }
    }
}

/// The result of reading a file: its kind, canonical full text (the string the
/// flat reader displays and offsets index into), the segments to chunk, and a
/// page count for paginated sources.
pub struct Extracted {
    pub filetype: FileType,
    pub full_text: String,
    pub segments: Vec<Segment>,
    pub page_count: Option<usize>,
}

/// Separator inserted between PDF pages in the canonical full text. Page-relative
/// offsets (used for PDF highlighting) are unaffected by it; it only spaces pages
/// apart in the flat text reader.
const PAGE_SEPARATOR: char = '\n';

/// Read and classify the file at `path`. PDF detection is by content signature
/// (see [`is_pdf`]); a PDF is extracted page-by-page via PDFium, anything else is
/// decoded leniently as text.
pub fn extract(pdfium: &Pdfium, path: &str) -> Result<Extracted, String> {
    if is_pdf(path)? {
        extract_pdf(pdfium, path)
    } else {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path} as text: {e}"))?;
        let full_text = decode_text(&bytes);
        let segments = vec![Segment::flat(full_text.clone())];
        Ok(Extracted {
            filetype: FileType::Text,
            full_text,
            segments,
            page_count: None,
        })
    }
}

/// Build the canonical full text + per-page segments for a PDF. Each page's
/// `base_offset` is the char index where that page's text begins in the full
/// text (page text + a one-char separator between pages).
fn extract_pdf(pdfium: &Pdfium, path: &str) -> Result<Extracted, String> {
    let pages = pdf::extract_pdf_pages(pdfium, path)?;
    let page_count = pages.len();

    let mut full_text = String::new();
    let mut segments = Vec::with_capacity(page_count);
    let mut base_offset = 0usize; // in chars
    for (i, page_text) in pages.iter().enumerate() {
        segments.push(Segment {
            text: page_text.clone(),
            page: Some(i),
            base_offset,
        });
        full_text.push_str(page_text);
        let mut page_chars = page_text.chars().count();
        if i + 1 < page_count {
            full_text.push(PAGE_SEPARATOR);
            page_chars += 1;
        }
        base_offset += page_chars;
    }

    Ok(Extracted {
        filetype: FileType::Pdf,
        full_text,
        segments,
        page_count: Some(page_count),
    })
}

/// Whether the file at `path` is a PDF.
///
/// The content signature is authoritative — a conforming PDF begins with the
/// `%PDF-` magic bytes — so a file is correctly classified regardless of its
/// name. A `.pdf` extension is honored as a fallback for the (rare) PDF that
/// hides its header behind leading junk, which PDFium still opens. Reads only the
/// first few bytes.
pub fn is_pdf(path: &str) -> Result<bool, String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut head = [0u8; 5];
    let n = file
        .read(&mut head)
        .map_err(|e| format!("read {path}: {e}"))?;
    if head[..n] == *b"%PDF-" {
        return Ok(true);
    }
    Ok(std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf")))
}

/// Decode `bytes` from a text file into a `String`, leniently.
///
/// Tries UTF-8 first (modern files, and what our PDF path produces). Otherwise it
/// runs `chardetng` — Firefox's charset detector — over the bytes to guess the
/// encoding (Windows-1252, ISO-8859-*, Shift-JIS, GBK, …) and decodes with that.
/// The legacy encodings map essentially every byte, so this never fails; a
/// genuinely binary file just yields gibberish chunks rather than an error.
pub fn decode_text(bytes: &[u8]) -> String {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_owned();
    }
    let mut detector = chardetng::EncodingDetector::new();
    detector.feed(bytes, true); // true: these are the last (only) bytes
    let encoding = detector.guess(None, true); // no TLD hint; allow UTF-8
    encoding.decode(bytes).0.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_utf8_directly() {
        assert_eq!(decode_text("héllo".as_bytes()), "héllo");
    }

    #[test]
    fn decodes_windows_1252_fallback() {
        // 0x92 is a curly apostrophe in Windows-1252 and invalid UTF-8.
        let decoded = decode_text(&[b'i', b't', 0x92, b's']);
        assert!(decoded.starts_with("it"));
        assert!(decoded.chars().count() == 4);
    }

    #[test]
    fn is_pdf_by_extension_for_missing_file_errors() {
        // Non-existent path surfaces an open error rather than a bool.
        assert!(is_pdf("/no/such/file.pdf").is_err());
    }
}
