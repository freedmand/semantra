#!/usr/bin/env bash
# Fetch the EmbeddingGemma 2 files the app loads into models/<name>/ (the
# layout `MODEL_NAME` in src/lib.rs points at). The weights (~1.5 GB, BF16) are
# gitignored and fetched out-of-band.
#
# Usage:
#   ./fetch-model.sh                          # google/embeddinggemma-2
#   ./fetch-model.sh <org>/<repo> [<name>]    # another checkpoint of the same architecture
#
# The weights are used exactly as published (BF16). Do NOT convert them to F16:
# this model's activations exceed F16's range and it silently produces NaN /
# degraded embeddings in F16.
set -euo pipefail

REPO="${1:-google/embeddinggemma-2}"
NAME="${2:-$(basename "${REPO}")}"
BASE="https://huggingface.co/${REPO}/resolve/main"

# What semantra-embed reads: model config, tokenizer, weights, and the
# processor configs (kept for reference/debugging of media preprocessing).
FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "processor_config.json"
  "preprocessor_config.json"
  "config_sentence_transformers.json"
  "model.safetensors"
)

DEST="$(cd "$(dirname "$0")" && pwd)/models/${NAME}"
echo "Fetching ${REPO} -> ${DEST}"
mkdir -p "${DEST}"

AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

for f in "${FILES[@]}"; do
  echo "  ${f}"
  curl -fSL --retry 3 ${AUTH[@]+"${AUTH[@]}"} "${BASE}/${f}" -o "${DEST}/${f}"
done

echo "Done."
