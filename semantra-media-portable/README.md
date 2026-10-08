# semantra-media-portable

Portable (Windows/Linux) media decoding with the same API and semantics as
semantra-embed's macOS decoders (ImageIO / AVFoundation). semantra-embed
re-exports it on non-macOS targets (`media::image::{Rgb8, decode, decode_max,
encode_jpeg}`, `media::av::{Probe, probe, stream_audio, frames_at}`).

## Design

**Stills** — `image` 0.25 decodes and reads EXIF orientation; moxcms converts
embedded ICC profiles (incl. CMYK) to sRGB; big JPEGs decode DCT-scaled so a
48 MP photo never materializes at full size; transparency composites over white
in gamma-encoded sRGB (as CoreGraphics does). HEIC/HEIF/AVIF go through the
ffmpeg sidecar (needs ffmpeg ≥ 9 for full tile grids; dav1d for AVIF).

**Audio** — Symphonia decodes; Rubato resamples to 16 kHz with a filter
matched to AVFoundation's converter (32-tap Blackman sinc, cutoff 0.9) and its
alignment. The downmix copies AVFoundation's measured weights (front L/R and
surrounds 0.7071, center 1, LFE 0, 7.1 sides 0.5) — not a plain average. MP4
edit lists (AAC priming) and the MP3 decoder delay are applied. Opus, HE-AAC
(SBR), AC-3, WMA and DTS fall back to the ffmpeg sidecar.

**Video frames** — an ffmpeg CLI sidecar behind the `VideoBackend` trait (so
Media Foundation / GStreamer can be added later): every codec and container
behaves the same on Windows and Linux, there is no C build, LGPL compliance is
trivial (separate unmodified executable), and a decoder crash kills a child
process, not the app. Frames return as PPM over a pipe with timestamps from
`showinfo`; dense time lists use one process, sparse lists up to 8 in
parallel. AVFoundation's quirks are mirrored (single-precision scale with a
floored short side, BT.709/BT.601 for untagged HD/SD, its 1.961 power tone
curve). The ffmpeg path is `set_ffmpeg_path`, else `$SEMANTRA_FFMPEG`, else
`ffmpeg[.exe]` next to the executable (where Tauri installs sidecars), else
`PATH`.

## Parity with the macOS decoders (Apple M5)

- Images (36 fixtures): dimensions/orientation identical; lossless bit-exact,
  lossy 45–69 dB PSNR at decode size; embedding cosine 0.9987–1.0000.
- Audio: correlation with AVFoundation's 16 kHz PCM ≥ 0.9999 across WAV, AIFF,
  FLAC, ALAC, MP3, AAC, HE-AAC, Opus, Vorbis, 48 kHz and 5.1; embedding cosine
  0.9989–1.0000.
- Video frames (1344 px): sizes/rotation identical; 40–47 dB PSNR; video-window
  embedding cosine 0.9978–0.9993 (SDR), HLG 0.9991, PQ 0.978.

## Shipping ffmpeg (LGPL)

Decode-only LGPL builds suffice (no `--enable-gpl`, no nonfree). Ship the LGPL
notice, version and configure flags, and the source (or an offer). CI uses
BtbN/FFmpeg-Builds `-lgpl` n9.0 builds (`src-tauri/scripts/fetch-ffmpeg.sh`);
`scripts/build_ffmpeg_min.sh` builds an ~8 MB decode-only binary instead. The
H.264/HEVC/AAC decoders carry patent considerations worth a legal review.

## Known gaps

No camera RAW off macOS; HEIC/AVIF need the sidecar; gray ICC profiles and
video color primaries aren't converted; HDR→SDR is approximate; non-square
pixels aren't corrected.

## Tests

`cargo test` (21 tests; ffmpeg-dependent ones skip without ffmpeg). Fixtures:
`scripts/make_fixtures.sh`.
