#!/usr/bin/env bash
# Run by Tauri right before bundling (build.beforeBundleCommand). Ensures MLX's
# compiled Metal kernels are in src-tauri/mlx/ so they ship inside the app; a
# bundle without them aborts at launch on every machine but the build machine.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/mlx/mlx.metallib"
if [[ ! -s "${DEST}" ]]; then
  # Fall back to mlx-sys's default location (a build cached before
  # MLX_RS_METAL_PATH was set writes it there).
  SRC="$(ls -t "${HOME}"/.mlx/lib/*/mlx.metallib 2>/dev/null | head -1 || true)"
  if [[ -n "${SRC}" ]]; then
    cp "${SRC}" "${DEST}"
  fi
fi
if [[ ! -s "${DEST}" ]]; then
  echo "error: ${DEST} is missing — MLX's Metal kernels would not be bundled" >&2
  exit 1
fi
echo "mlx.metallib staged ($(du -h "${DEST}" | cut -f1))"
