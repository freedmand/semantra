//! Pick the inference backend: MLX on Apple Silicon (fastest there), ONNX
//! Runtime everywhere else (Windows, Linux, Intel Macs). The `onnx` feature
//! forces ONNX on Apple Silicon too, to test that backend locally.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(backend_mlx)");
    println!("cargo::rustc-check-cfg=cfg(backend_onnx)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let force_onnx = std::env::var_os("CARGO_FEATURE_ONNX").is_some();
    if os == "macos" && arch == "aarch64" && !force_onnx {
        println!("cargo::rustc-cfg=backend_mlx");
    } else {
        println!("cargo::rustc-cfg=backend_onnx");
    }
}
