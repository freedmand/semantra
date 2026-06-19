#!/usr/bin/env python3
"""Quantize a leaf model's BERT weights from float32 to float16, halving the
on-disk (and bundled-app) size with negligible quality loss.

Why this is safe: the candle pipeline loads `model.safetensors` through
`VarBuilder::from_mmaped_safetensors(.., DTYPE, ..)` with `DTYPE = F32`, and
candle casts each tensor from its on-disk dtype up to F32 at load time. So
storing the weights as F16 leaves *inference* fully F32 — only the stored
weights are rounded to half precision (~3 decimal digits), far inside the
1e-3 tolerance the crate's decomposition test asserts.

We only touch `model.safetensors` (the ~86MB BERT). The tiny `2_Dense/` head is
left F32 on purpose: `load_dense` loads it WITHOUT a dtype cast, so an F16 dense
weight would dtype-mismatch the F32 pooled activations it multiplies.

Only floating tensors are converted; any non-float buffer (e.g. an int
`position_ids`) is passed through untouched. Idempotent: a file whose float
tensors are already F16 is left as-is.

Usage:
  ./quantize-model.py models/mdbr-leaf-ir        # quantize one model dir
"""
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

FLOAT_DOWNCAST = {"F32", "F64"}  # dtypes we shrink to F16


def quantize(model_dir: Path) -> None:
    path = model_dir / "model.safetensors"
    if not path.exists():
        sys.exit(f"no model.safetensors in {model_dir}")

    tensors, metadata, changed = {}, None, False
    with safe_open(path, "numpy") as f:
        metadata = f.metadata()
        for key in f.keys():
            arr = f.get_tensor(key)
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = arr.astype(np.float16)
                changed = True
            tensors[key] = arr

    if not changed:
        print(f"{path}: already F16, nothing to do")
        return

    before = path.stat().st_size
    # Write to a sibling temp file then rename, so an interrupted run never
    # leaves a half-written model behind.
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(tensors, tmp, metadata=metadata)
    tmp.replace(path)
    after = path.stat().st_size
    print(f"{path}: {before/1e6:.1f}MB -> {after/1e6:.1f}MB (F32 -> F16)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    quantize(Path(sys.argv[1]))
