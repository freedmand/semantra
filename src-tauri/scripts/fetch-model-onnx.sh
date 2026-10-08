#!/usr/bin/env bash
# Fetch the EmbeddingGemma 2 ONNX export (onnx-community, fp16) that the ONNX
# backend (Windows, Linux, Intel Macs) loads, into models/embeddinggemma-2-onnx/:
#
#   config.json, tokenizer.json
#   onnx/{model,vision_encoder,audio_encoder}_fp16.onnx (+ .onnx_data weights)
#
# ~1.5 GB. fp16 matches the BF16 MLX model to cos >= 0.9997 on every modality
# (the export keeps overflow-prone ops in F32) — unlike converting the BF16
# weights to F16 ourselves, which overflows. HF_TOKEN is optional.
set -euo pipefail

REPO="onnx-community/embeddinggemma-2-ONNX"
BASE="https://huggingface.co/${REPO}/resolve/main"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/embeddinggemma-2-onnx"
FILES=(
  config.json
  tokenizer.json
  onnx/model_fp16.onnx onnx/model_fp16.onnx_data
  onnx/vision_encoder_fp16.onnx onnx/vision_encoder_fp16.onnx_data
  onnx/audio_encoder_fp16.onnx onnx/audio_encoder_fp16.onnx_data
)

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

echo "Fetching ${REPO} -> ${DEST}"
mkdir -p "${DEST}/onnx"
for f in "${FILES[@]}"; do
  if [[ -s "${DEST}/${f}" ]]; then
    echo "  ${f} (cached)"
    continue
  fi
  echo "  ${f}"
  curl -fL --retry 3 --retry-delay 5 "${AUTH[@]}" -o "${DEST}/${f}.part" "${BASE}/${f}"
  mv "${DEST}/${f}.part" "${DEST}/${f}"
done
du -sh "${DEST}"
