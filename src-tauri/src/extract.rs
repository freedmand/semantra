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
    Csv,
}

impl FileType {
    pub fn as_str(self) -> &'static str {
        match self {
            FileType::Text => "text",
            FileType::Pdf => "pdf",
            FileType::Csv => "csv",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "text" => Ok(FileType::Text),
            "pdf" => Ok(FileType::Pdf),
            "csv" => Ok(FileType::Csv),
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
    } else if is_csv(path) {
        extract_csv(path)
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
            col: None,
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

/// A parsed CSV: the header row plus the data rows, all cells verbatim. Built by
/// [`read_csv_grid`] and returned to the canvas-grid reader (serialized camelCase).
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CsvGrid {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Parse the CSV at `path` into a [`CsvGrid`]. The first record is the header row;
/// the rest are data rows. Quoted fields (embedded commas, quotes, newlines) are
/// handled per RFC 4180 and kept **verbatim** — no trimming. `flexible` tolerates
/// rows with a different field count than the header rather than erroring.
pub fn read_csv_grid(path: &str) -> Result<CsvGrid, String> {
    // Decode leniently to UTF-8 first (same path as plain text), then parse: this
    // lets non-UTF-8 CSVs (Windows-1252, …) through instead of failing on bytes.
    let bytes = std::fs::read(path).map_err(|e| format!("read {path} as csv: {e}"))?;
    let text = decode_text(&bytes);
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // take the header row ourselves so it's explicit
        .flexible(true)
        .from_reader(text.as_bytes());
    let mut records = rdr.records();
    let headers = match records.next() {
        Some(r) => r
            .map_err(|e| format!("parse csv header: {e}"))?
            .iter()
            .map(str::to_string)
            .collect(),
        None => Vec::new(), // empty file → no headers, no rows
    };
    let mut rows = Vec::new();
    for rec in records {
        let rec = rec.map_err(|e| format!("parse csv row: {e}"))?;
        rows.push(rec.iter().map(str::to_string).collect());
    }
    Ok(CsvGrid { headers, rows })
}

/// The column label used to give a cell semantic context when embedded. The raw
/// header is used when present; blank or missing headers fall back to `Column N`
/// (1-based) so every field still carries an identifiable column.
fn header_label(headers: &[String], col: usize) -> String {
    match headers.get(col) {
        Some(h) if !h.is_empty() => h.clone(),
        _ => format!("Column {}", col + 1),
    }
}

/// Build the canonical full text + per-cell segments for a CSV. Every non-empty
/// cell becomes one segment (→ one chunk via [`CellChunker`](crate::chunk::CellChunker))
/// whose embedded text is `"{header}: {value}"` with the value **verbatim**. The
/// canonical full text is those composed strings joined by `\n`, so each
/// segment's `[base_offset, base_offset+len)` slices back to its own text exactly.
/// `page` carries the 0-based data-row index and `col` the 0-based column index,
/// which the grid reader navigates by. The header row is not itself indexed.
fn extract_csv(path: &str) -> Result<Extracted, String> {
    let grid = read_csv_grid(path)?;
    let mut full_text = String::new();
    let mut segments = Vec::new();
    let mut base_offset = 0usize; // in chars
    let mut first = true;
    for (row_idx, row) in grid.rows.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            if value.is_empty() {
                continue; // empty field: nothing to embed
            }
            // Separator between cells in the canonical text (not part of any cell).
            if !first {
                full_text.push('\n');
                base_offset += 1;
            }
            first = false;
            let composed = format!("{}: {}", header_label(&grid.headers, col_idx), value);
            let char_len = composed.chars().count();
            full_text.push_str(&composed);
            segments.push(Segment {
                text: composed,
                page: Some(row_idx),
                col: Some(col_idx),
                base_offset,
            });
            base_offset += char_len;
        }
    }
    Ok(Extracted {
        filetype: FileType::Csv,
        full_text,
        segments,
        // CSVs have no pages; row count is exposed via the grid, not page_count.
        page_count: None,
    })
}

/// Whether the file at `path` is a CSV. Detected by the `.csv` extension only —
/// CSV has no content signature, and sniffing it from bytes is unreliable, so a
/// CSV without the extension is indexed as plain text (still searchable, just no
/// grid reader).
fn is_csv(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
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
    fn extract_csv_indexes_each_nonempty_cell_verbatim() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.csv");
        // A quoted field with an embedded comma + newline, and a trailing empty cell.
        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, "city,note\nParis,\"a, b\nc\"\nTokyo,\n").unwrap();
        drop(f);
        let p = path.to_str().unwrap();

        // The grid keeps every cell verbatim, including the embedded comma+newline.
        let grid = read_csv_grid(p).unwrap();
        assert_eq!(grid.headers, vec!["city", "note"]);
        assert_eq!(grid.rows, vec![vec!["Paris", "a, b\nc"], vec!["Tokyo", ""]]);

        let ex = extract_csv(p).unwrap();
        assert_eq!(ex.filetype, FileType::Csv);
        assert_eq!(ex.page_count, None);
        // Tokyo's empty "note" is skipped; values (incl. the newline) stay verbatim.
        let texts: Vec<&str> = ex.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["city: Paris", "note: a, b\nc", "city: Tokyo"]);
        // (row, col) ride along on each segment.
        assert_eq!((ex.segments[1].page, ex.segments[1].col), (Some(0), Some(1)));
        assert_eq!((ex.segments[2].page, ex.segments[2].col), (Some(1), Some(0)));
        // Each segment's offsets slice its exact composed text out of full_text.
        for s in &ex.segments {
            let got: String = ex
                .full_text
                .chars()
                .skip(s.base_offset)
                .take(s.text.chars().count())
                .collect();
            assert_eq!(got, s.text);
        }
    }

    #[test]
    fn is_pdf_by_extension_for_missing_file_errors() {
        // Non-existent path surfaces an open error rather than a bool.
        assert!(is_pdf("/no/such/file.pdf").is_err());
    }
}
