#!/usr/bin/env bash
# Fetch the ONNX Runtime shared library the ONNX backend loads at runtime into
# onnxruntime/ (bundled as a resource; see tauri.onnx.conf.json).
#
#   scripts/fetch-onnxruntime.sh [windows-x64 | linux-x64 | macos-x64 | macos-arm64]
#
# Version 1.27.0 everywhere (the `ort` crate is built against API 27):
# - windows-x64, linux-x64, macos-arm64: from the `onnxruntime-webgpu` wheel,
#   whose library has the WebGPU execution provider built in (GPU via D3D12,
#   Vulkan or Metal; the Windows build also needs the DirectX shader compiler
#   DLLs shipped next to it). CPU fallback is automatic.
# - macos-x64: Microsoft no longer publishes Intel Mac builds (last: 1.16), so
#   it is built from source (CPU) — set ORT_X86_64_DYLIB to a built
#   libonnxruntime.dylib, or let scripts/build-onnxruntime-macos-x64.sh build it.
set -euo pipefail

VERSION="1.27.0"
DEST="$(cd "$(dirname "$0")/.." && pwd)/onnxruntime"
TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) TARGET=macos-arm64 ;;
    Darwin-x86_64) TARGET=macos-x64 ;;
    Linux-x86_64) TARGET=linux-x64 ;;
    MINGW*|MSYS*|CYGWIN*) TARGET=windows-x64 ;;
    *) echo "unsupported host $(uname -s)-$(uname -m); pass a target" >&2; exit 1 ;;
  esac
fi
mkdir -p "${DEST}"

wheel() { # $1 = wheel platform tag fragment; prints the wheel URL
  curl -fsSL "https://pypi.org/pypi/onnxruntime-webgpu/${VERSION}/json" | python3 -c "
import json, sys
for f in json.load(sys.stdin)['urls']:
    if 'cp312' in f['filename'] and '$1' in f['filename']:
        print(f['url']); break"
}

extract() { # $1 = wheel URL; remaining = member basenames to extract into DEST
  local url="$1"; shift
  local tmp; tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "${tmp}/ort.whl" "${url}"
  for m in "$@"; do
    unzip -o -j -q "${tmp}/ort.whl" "onnxruntime/capi/${m}" -d "${DEST}"
  done
  rm -rf "${tmp}"
}

case "${TARGET}" in
  windows-x64)
    extract "$(wheel win_amd64)" onnxruntime.dll onnxruntime_providers_shared.dll dxcompiler.dll dxil.dll ;;
  linux-x64)
    extract "$(wheel manylinux)" "libonnxruntime.so.${VERSION}" libonnxruntime_providers_shared.so
    mv -f "${DEST}/libonnxruntime.so.${VERSION}" "${DEST}/libonnxruntime.so" ;;
  macos-arm64)
    extract "$(wheel macosx_14_0_arm64)" "libonnxruntime.${VERSION}.dylib"
    mv -f "${DEST}/libonnxruntime.${VERSION}.dylib" "${DEST}/libonnxruntime.dylib" ;;
  macos-x64)
    SRC="${ORT_X86_64_DYLIB:-}"
    if [[ -z "${SRC}" ]]; then
      SRC="$("$(dirname "$0")/build-onnxruntime-macos-x64.sh")"
    fi
    cp -f "${SRC}" "${DEST}/libonnxruntime.dylib" ;;
  *) echo "unknown target ${TARGET}" >&2; exit 1 ;;
esac
ls -la "${DEST}"
