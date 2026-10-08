#!/usr/bin/env bash
# Build ONNX Runtime 1.27.0 (CPU, shared) for Intel Macs — Microsoft stopped
# publishing x86_64 macOS binaries after 1.16. Works on Intel or Apple Silicon
# hosts (cross-compiles). ~30 min on a laptop. Prints the built dylib's path.
#
# Homebrew's protobuf must be kept out of CMake's search path: ORT builds its
# own, and mixing in a newer protoc breaks the generated onnx-ml.pb.h.
set -euo pipefail
VERSION="1.27.0"
WORK="${ORT_BUILD_DIR:-${HOME}/.cache/semantra/onnxruntime-${VERSION}}"
OUT="${WORK}/onnxruntime/build/x86_64/Release/libonnxruntime.${VERSION}.dylib"
if [[ -s "${OUT}" ]]; then
  echo "${OUT}"
  exit 0
fi
mkdir -p "${WORK}"
if [[ ! -d "${WORK}/onnxruntime" ]]; then
  git clone -q --depth 1 --branch "v${VERSION}" --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "${WORK}/onnxruntime" >&2
fi
cd "${WORK}/onnxruntime"
python3 tools/ci_build/build.py --build_dir build/x86_64 --config Release --build_shared_lib --parallel \
  --skip_tests --osx_arch x86_64 --apple_deploy_target 13.0 --compile_no_warning_as_error \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_IGNORE_PREFIX_PATH=/opt/homebrew:/usr/local >&2
echo "${OUT}"
