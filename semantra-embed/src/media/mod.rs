//! Media decoding into model-ready inputs.
//!
//! - [`image`]: still images -> resized RGB -> patch grid.
//! - [`av`]: audio/video -> 16 kHz mono PCM windows and 1 fps frame windows.
//! - [`mel`]: log-mel features for the audio tower.
//!
//! macOS (Apple Silicon and Intel) decodes natively with ImageIO and
//! AVFoundation; other platforms use the portable decoders in `portable`.

#[cfg(target_os = "macos")]
pub mod av;
#[cfg(not(target_os = "macos"))]
pub use portable::av;
pub mod image;
pub mod mel;
#[cfg(not(target_os = "macos"))]
mod portable;
