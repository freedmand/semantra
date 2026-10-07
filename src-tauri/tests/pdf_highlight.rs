//! End-to-end check of the PDF highlight pipeline's core invariant: the page
//! text the chunker indexes is **char-for-char identical** to the page text
//! `page_chars` returns boxes for, so a chunk's page-relative offset maps onto
//! the right characters. Also sanity-checks that the resulting boxes sit within
//! the page bounds.
//!
//! Uses a real PDF fixture if present (semantra-web's hamlet.pdf); skips cleanly
//! when it isn't, so the suite stays green on machines without it.
//!
//! Run with: cargo test --test pdf_highlight

use std::path::PathBuf;

use semantra_lib::chunk::{Chunker, TokenWindowChunker};
use semantra_lib::{extract, pdf};

/// A text-bearing PDF to test against, from `PDF_FIXTURE`; skips when unset.
fn fixture() -> Option<String> {
    let p = PathBuf::from(std::env::var("PDF_FIXTURE").ok()?);
    p.exists().then(|| p.to_string_lossy().to_string())
}

fn pdfium() -> pdf::PdfiumLib {
    let lib_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("libpdfium");
    pdf::load_library(&lib_dir).expect("bundled PDFium library should load")
}

#[test]
fn chunk_offsets_align_with_pdfium_char_boxes() {
    let Some(path) = fixture() else {
        eprintln!("skipping: hamlet.pdf fixture not present");
        return;
    };
    let pdfium = pdfium();

    // Extract → segments (one per page) → chunks.
    let extracted = extract::extract(&pdfium, &path).expect("extract hamlet.pdf");
    assert_eq!(extracted.filetype, extract::FileType::Pdf);
    let segments = &extracted.segments;
    assert!(!segments.is_empty(), "PDF should have pages");

    // Offsets are what's under test, not tokenization: one "token" per
    // whitespace-delimited word gives the old ~70-word windows.
    let word_starts = |t: &str| -> Vec<usize> {
        let mut prev_ws = true;
        let mut out = Vec::new();
        for (b, ch) in t.char_indices() {
            if !ch.is_whitespace() && prev_ws {
                out.push(b);
            }
            prev_ws = ch.is_whitespace();
        }
        out
    };
    let chunks = TokenWindowChunker::new(70, 8, word_starts).chunk(segments);
    assert!(!chunks.is_empty(), "PDF should produce chunks");

    // For each of the first few chunks, the page the chunker saw must match the
    // page `page_chars` returns, char-for-char — otherwise offsets would drift.
    let mut checked_boxes = false;
    for chunk in chunks.iter().take(8) {
        let page = chunk.page.expect("PDF chunk carries a page");
        let seg = segments.iter().find(|s| s.page == Some(page)).unwrap();
        let pc = pdf::page_chars(&pdfium, &path, page).expect("page_chars");

        // The two independent code paths must agree on the page string.
        assert_eq!(
            pc.text, seg.text,
            "page {page}: extracted text and page_chars text must be identical"
        );
        assert_eq!(
            pc.chars.len(),
            seg.text.chars().count(),
            "page {page}: one char box per character"
        );

        // The chunk's page-relative span indexes real characters.
        let start = chunk.page_char_start;
        let len = chunk.char_end - chunk.char_start;
        let end = (start + len).min(pc.chars.len());
        assert!(start <= end, "valid slice");

        // Every box in the span sits within the page; at least one has geometry.
        let mut any_geom = false;
        for b in &pc.chars[start..end] {
            assert!(
                b.left >= -1.0
                    && b.right <= pc.width + 1.0
                    && b.bottom >= -1.0
                    && b.top <= pc.height + 1.0,
                "page {page}: char box {b:?} must be within {}x{}",
                pc.width,
                pc.height
            );
            if b.right > b.left && b.top > b.bottom {
                any_geom = true;
            }
        }
        assert!(any_geom, "page {page}: span should contain visible glyphs");
        checked_boxes = true;
    }
    assert!(checked_boxes);
}
