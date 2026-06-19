//! Build script.
//!
//! Beyond the usual `tauri_build::build()`, this fetches a prebuilt **dynamic**
//! PDFium library for the *target* platform into `libpdfium/` so it can be
//! bundled as a Tauri resource (see `bundle.resources` in tauri.conf.json) and
//! loaded at runtime by `pdfium-render` (see `src/pdf.rs`).
//!
//! We deliberately bundle the dynamic lib rather than statically linking:
//! there is no canonical prebuilt static `libpdfium.a` covering Linux/Windows,
//! and static linking drags in C++-runtime/Windows complications. The non-V8
//! build is ~7.4 MB — tiny next to the bundled model weights — and we only ever
//! extract text, never render, so no image deps are pulled in.
//!
//! The download is cached: if the lib already exists in `libpdfium/`, nothing is
//! fetched. Binaries come from the canonical bblanchon/pdfium-binaries release
//! pinned below; bump it together with `pdfium-render`'s `pdfium_latest`.

use std::path::PathBuf;
use std::process::Command;

/// Pinned bblanchon/pdfium-binaries release tag. The literal tag is
/// `chromium/<build>`; the `/` is percent-encoded in the download URL.
const PDFIUM_RELEASE: &str = "chromium/7881";

fn main() {
    fetch_pdfium();
    tauri_build::build();
}

/// Ensure the target's `libpdfium` is present under `libpdfium/`, downloading it
/// from the pinned release if missing. Panics with a clear message on failure —
/// the app cannot index PDFs without it.
fn fetch_pdfium() {
    // Only re-run when this script changes; otherwise the existence check below
    // makes repeated builds a no-op.
    println!("cargo:rerun-if-changed=build.rs");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // (bblanchon archive name, path of the lib inside the archive, output filename)
    let (archive, member, out_name) = match (os.as_str(), arch.as_str()) {
        ("macos", "aarch64") => ("pdfium-mac-arm64.tgz", "lib/libpdfium.dylib", "libpdfium.dylib"),
        ("macos", "x86_64") => ("pdfium-mac-x64.tgz", "lib/libpdfium.dylib", "libpdfium.dylib"),
        ("linux", "x86_64") => ("pdfium-linux-x64.tgz", "lib/libpdfium.so", "libpdfium.so"),
        ("linux", "aarch64") => ("pdfium-linux-arm64.tgz", "lib/libpdfium.so", "libpdfium.so"),
        ("windows", "x86_64") => ("pdfium-win-x64.tgz", "bin/pdfium.dll", "pdfium.dll"),
        ("windows", "aarch64") => ("pdfium-win-arm64.tgz", "bin/pdfium.dll", "pdfium.dll"),
        (o, a) => panic!("no PDFium binary mapping for target {o}/{a}; add it to build.rs"),
    };

    let lib_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("libpdfium");
    let out_lib = lib_dir.join(out_name);
    if out_lib.exists() {
        return; // cached
    }
    std::fs::create_dir_all(&lib_dir).expect("create libpdfium dir");

    // `chromium/7881` -> `chromium%2F7881` for the download URL.
    let tag = PDFIUM_RELEASE.replace('/', "%2F");
    let url =
        format!("https://github.com/bblanchon/pdfium-binaries/releases/download/{tag}/{archive}");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let archive_path = out_dir.join(archive);

    // Download (curl) then extract just the lib member (tar). Both ship with
    // modern macOS, Linux, and Windows 10+.
    run(
        Command::new("curl")
            .args(["-fSL", "--retry", "3", "-o"])
            .arg(&archive_path)
            .arg(&url),
        &format!("download {url}"),
    );
    run(
        Command::new("tar")
            .arg("xzf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&out_dir)
            .arg(member),
        &format!("extract {member} from {archive}"),
    );

    std::fs::copy(out_dir.join(member), &out_lib)
        .unwrap_or_else(|e| panic!("copy {member} -> {}: {e}", out_lib.display()));
}

/// Run `cmd`, panicking with `what` as context if it cannot launch or exits non-zero.
fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run command to {what}: {e}"));
    if !status.success() {
        panic!("command to {what} failed with status {status}");
    }
}
