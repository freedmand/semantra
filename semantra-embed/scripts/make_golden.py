"""FP32 PyTorch reference ("golden") embeddings for semantra-embed's parity tests.

Usage:
  pip install torch "sentence-transformers>=6.1" transformers pillow soundfile
  python make_golden.py <work_dir>

<work_dir> must contain:
  models/embeddinggemma-2/   a copy of google/embeddinggemma-2
  media/                     the_beach.jpg, tree.jpg, big_sur_road.jpg (photos),
                             pasta.wav / stocks.wav (16 kHz mono speech),
                             frog.wav (a short sound effect), clip.mp4 (video)
Writes <work_dir>/golden/{embeddings.npz,embeddings.json,texts.json}; then run
  EG2_MODEL_DIR=<work_dir>/models/embeddinggemma-2 EG2_GOLDEN_DIR=<work_dir>/golden \
    cargo test --release
(The processor-patch and mel fixtures used by some tests are written by the
same reference processor; see the tests for the expected file names.)
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer

LAB = Path(sys.argv[1]); MODEL = LAB / "models/embeddinggemma-2"; M = LAB / "media"; OUT = LAB / "golden"; OUT.mkdir(exist_ok=True)
torch.manual_seed(0)
st = SentenceTransformer(str(MODEL), device="cpu", model_kwargs={"torch_dtype": torch.float32})

texts = {
  "q_pasta": "task: search result | query: how do I make homemade noodles",
  "q_beach": "task: search result | query: waves crashing on a sandy shore",
  "q_market": "task: search result | query: why did the stock market drop",
  "d_pasta": "title: none | text: Combine flour and eggs, knead the dough, roll it thin and slice into fettuccine.",
  "d_market": "title: none | text: Equities slid as bond yields climbed on fears the Fed would hike again.",
  "d_unicode": "title: notes.txt | text: Café déjà vu — naïve résumé. 東京は日本の首都です。 🚀 emoji & code: fn main() { println!(\"hi\"); }",
  "d_long": "title: none | text: " + " ".join(["The quick brown fox jumps over the lazy dog near the riverbank at dawn."] * 90),
}
res = {}
for k, t in texts.items():
    res[k] = st.encode(t, convert_to_numpy=True, normalize_embeddings=True)
# batch encode (padding path) must equal single
batch = st.encode(list(texts.values()), convert_to_numpy=True, batch_size=8)
print("batch-vs-single max abs diff:", float(np.abs(batch - np.stack(list(res.values()))).max()))

media = {
  "img_beach": {"image": str(M / "the_beach.jpg")},
  "img_tree": {"image": str(M / "tree.jpg")},
  "img_road": {"image": str(M / "big_sur_road.jpg")},
  "aud_pasta": {"audio": str(M / "pasta.wav")},
  "aud_stocks": {"audio": str(M / "stocks.wav")},
  "aud_frog": {"audio": str(M / "frog.wav")},
  "vid_clip": {"video": str(M / "clip.mp4")},
}
for k, v in media.items():
    t0 = time.time()
    try:
        res[k] = st.encode(v, convert_to_numpy=True, normalize_embeddings=True)
        print(f"{k}: ok {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"{k}: FAILED {type(e).__name__}: {e}")

np.savez(OUT / "embeddings.npz", **res)
json.dump({k: v.astype(float).tolist() for k, v in res.items()}, open(OUT / "embeddings.json", "w"))
json.dump(texts, open(OUT / "texts.json", "w"), ensure_ascii=False, indent=1)
keys = list(res); E = np.stack([res[k] for k in keys])
S = E @ E.T
print("\nsimilarity (rows=queries):")
for q in [k for k in keys if k.startswith("q_")]:
    i = keys.index(q)
    ranked = sorted(((S[i, j], keys[j]) for j in range(len(keys)) if not keys[j].startswith("q_")), reverse=True)
    print(f"  {q}: " + ", ".join(f"{n}={s:.3f}" for s, n in ranked[:5]))
