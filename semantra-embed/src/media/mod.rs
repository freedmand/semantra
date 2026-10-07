//! Native (macOS) media decoding into model-ready inputs.
//!
//! - [`image`]: any still image ImageIO can open -> resized RGB -> patch grid.
//! - [`av`]: audio/video via AVFoundation -> 16 kHz mono PCM windows and
//!   1 fps frame windows.

pub mod av;
pub mod image;
pub mod mel;
