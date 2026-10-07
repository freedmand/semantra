//! PDFium is not re-entrant; `pdf.rs` serializes access. This hammers one PDF
//! from several threads at once (indexing renders + UI highlight/text calls)
//! and requires every call to succeed. Without the lock this test crashes the
//! process or fails with spurious `FormatError`s.
//!
//! Uses `PDF_FIXTURE` if set, else the first PDF in the app-data files dir.

use std::path::PathBuf;
use std::sync::Arc;

fn fixture() -> Option<String> {
    if let Ok(p) = std::env::var("PDF_FIXTURE") {
        return Some(p);
    }
    let dir = PathBuf::from(std::env::var("HOME").ok()?).join("Library/Application Support/com.semantra.app/files");
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "pdf"))
        .map(|p| p.to_string_lossy().into_owned())
}

#[test]
fn concurrent_pdfium_calls_all_succeed() {
    let Some(path) = fixture() else {
        eprintln!("no PDF fixture; skipping");
        return;
    };
    let lib = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("libpdfium");
    let pdfium = Arc::new(semantra_lib::pdf::load_library(&lib).unwrap());
    let pages = semantra_lib::pdf::extract_pdf_pages(&pdfium, &path).unwrap().len();
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let (pdfium, path) = (Arc::clone(&pdfium), path.clone());
            std::thread::spawn(move || {
                for i in 0..10 {
                    let page = i % pages;
                    let r = match t {
                        0 => semantra_lib::pdf::render_pages(&pdfium, &path, &[page, (page + 1) % pages], 1344).map(|v| {
                            assert!(v[0].width.max(v[0].height) == 1344, "page scaled to target");
                        }),
                        1 => semantra_lib::pdf::extract_pdf_pages(&pdfium, &path).map(|_| ()),
                        2 => semantra_lib::pdf::page_chars(&pdfium, &path, page).map(|_| ()),
                        _ => semantra_lib::pdf::render_page(&pdfium, &path, page, 320).map(|_| ()),
                    };
                    r.unwrap_or_else(|e| panic!("thread {t} iter {i}: {e}"));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker thread");
    }
}
