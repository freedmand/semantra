#!/usr/bin/env bash
# Fetch a MongoDB "leaf" embedding model's files from Hugging Face into
# models/<name>/, the per-model layout the app loads (see MODEL_NAME in
# src/lib.rs). These weights (~90MB) are gitignored and fetched out-of-band.
#
# Usage:
#   ./fetch-model.sh                 # defaults to mdbr-leaf-ir
#   ./fetch-model.sh mdbr-leaf-mt    # fetch a different leaf model
#
# After fetching, set `MODEL_NAME` in src/lib.rs to the model you want active.
set -euo pipefail

MODEL="${1:-mdbr-leaf-ir}"
REPO="MongoDB/${MODEL}"
BASE="https://huggingface.co/${REPO}/resolve/main"

# Files the candle pipeline needs (BERT -> Dense 384->768 -> L2). Mirrors the
# layout in leaf-ir-candle's setup_model.
FILES=(
  "config.json"
  "tokenizer.json"
  "model.safetensors"
  "2_Dense/model.safetensors"
)

DEST="$(cd "$(dirname "$0")" && pwd)/models/${MODEL}"
echo "Fetching ${REPO} -> ${DEST}"
mkdir -p "${DEST}/2_Dense"

for f in "${FILES[@]}"; do
  echo "  ${f}"
  curl -fSL --retry 3 "${BASE}/${f}" -o "${DEST}/${f}"
done

# Quantize the BERT weights F32 -> F16 to halve the bundled size. Inference
# stays F32 (candle upcasts on load); see quantize-model.py for why this is
# lossless in practice. Skipped (with a warning) if python3 is unavailable.
QUANT="$(dirname "$0")/quantize-model.py"
if command -v python3 >/dev/null 2>&1; then
  echo "Quantizing weights -> F16"
  python3 "${QUANT}" "${DEST}"
else
  echo "WARNING: python3 not found; skipping F16 quantization (model stays F32, ~2x larger)"
fi

echo "Done. Set MODEL_NAME = \"${MODEL}\" in src/lib.rs to use it."
