#!/usr/bin/env bash
# Fetch the ffmpeg sidecar the portable media decoders (semantra-media-portable)
# use on Windows and Linux for video frames, HEIC/AVIF stills and the audio
# codecs Symphonia lacks. macOS decodes natively (AVFoundation/ImageIO) and
# needs none.
#
#   scripts/fetch-ffmpeg.sh [windows-x64 | linux-x64]
#
# Bundled via `bundle.externalBin` as binaries/ffmpeg-<target-triple>[.exe];
# Tauri installs it next to the app, where the decoders look first.
#
# LGPL builds only (no --enable-gpl / nonfree): BtbN/FFmpeg-Builds n9.0
# (9.x is needed for full HEIC tile grids). It is a separate, unmodified
# executable; ship its license notice with the app. A smaller decode-only build
# is semantra-media-portable/scripts/build_ffmpeg_min.sh.
set -euo pipefail
RELEASE="ffmpeg-n9.0-latest"
BASE="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
DEST="$(cd "$(dirname "$0")/.." && pwd)/binaries"
TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) TARGET=linux-x64 ;;
    MINGW*|MSYS*|CYGWIN*) TARGET=windows-x64 ;;
    *) echo "no ffmpeg sidecar needed/available for $(uname -s)-$(uname -m)" >&2; exit 0 ;;
  esac
fi
mkdir -p "${DEST}"
tmp="$(mktemp -d)"
case "${TARGET}" in
  windows-x64)
    curl -fL --retry 3 -o "${tmp}/ff.zip" "${BASE}/${RELEASE}-win64-lgpl-9.0.zip"
    unzip -j -q "${tmp}/ff.zip" "*/bin/ffmpeg.exe" -d "${tmp}"
    mv -f "${tmp}/ffmpeg.exe" "${DEST}/ffmpeg-x86_64-pc-windows-msvc.exe"
    for f in LICENSE.txt; do unzip -j -q -o "${tmp}/ff.zip" "*/${f}" -d "${DEST}" 2>/dev/null || true; done ;;
  linux-x64)
    curl -fL --retry 3 -o "${tmp}/ff.tar.xz" "${BASE}/${RELEASE}-linux64-lgpl-9.0.tar.xz"
    tar -xJf "${tmp}/ff.tar.xz" -C "${tmp}" --wildcards "*/bin/ffmpeg" "*/LICENSE.txt"
    mv -f "${tmp}"/*/bin/ffmpeg "${DEST}/ffmpeg-x86_64-unknown-linux-gnu"
    mv -f "${tmp}"/*/LICENSE.txt "${DEST}/FFMPEG-LICENSE.txt" || true ;;
  *) echo "unknown target ${TARGET}" >&2; exit 1 ;;
esac
rm -rf "${tmp}"
ls -la "${DEST}"
