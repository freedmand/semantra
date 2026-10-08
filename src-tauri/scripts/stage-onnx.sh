#!/usr/bin/env bash
# Run by Tauri right before bundling ONNX-backend builds (the
# build.beforeBundleCommand in tauri.onnx.conf.json). Fails the build if the
# model export or the ONNX Runtime library would not be bundled.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
missing=0
for f in config.json tokenizer.json onnx/model_fp16.onnx onnx/model_fp16.onnx_data \
         onnx/vision_encoder_fp16.onnx onnx/vision_encoder_fp16.onnx_data \
         onnx/audio_encoder_fp16.onnx onnx/audio_encoder_fp16.onnx_data; do
  if [[ ! -s "${ROOT}/models/embeddinggemma-2-onnx/${f}" ]]; then
    echo "error: models/embeddinggemma-2-onnx/${f} is missing (run scripts/fetch-model-onnx.sh)" >&2
    missing=1
  fi
done
if ! ls "${ROOT}"/onnxruntime/*onnxruntime.* >/dev/null 2>&1; then
  echo "error: onnxruntime/ has no ONNX Runtime library (run scripts/fetch-onnxruntime.sh)" >&2
  missing=1
fi
[[ ${missing} -eq 0 ]] || exit 1
echo "ONNX model + runtime staged ($(du -sh "${ROOT}/models/embeddinggemma-2-onnx" | cut -f1) model)"
