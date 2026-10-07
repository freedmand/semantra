# Semantra desktop app

Semantra is a multi-tool for semantic search. You can upload images, text/PDF files, videos, audio
and CSVs and index them on your computer entirely locally. You can search the resulting files with
semantic- and keyword-based search.

Semantra 0.2.0 is Apple Silicon Mac-only and uses Google's open-source EmbeddingGemma2 model
which runs entirely locally on your machine.
You can download it [here](https://github.com/freedmand/semantra/releases/latest) to get started.

NOTE: For the legacy Python codebase, see: http://github.com/freedmand/semantra-python. For a previous
version of Semantra desktop that works on other operating systems (and uses a much smaller embedding model),
download pre-0.2.0 releases.

## Development

Semantra 0.2.0 runs on Apple Silicon Mac computers (previous versions work on Windows as well).
Embeddings come from [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), which produces
embeddings combining text, images, audio and video and are run on Apple MLX.

Prerequisites (one time):

```sh
xcodebuild -downloadComponent MetalToolchain   # MLX compiles Metal shaders
brew install cmake protobuf                    # MLX build; LanceDB codegen
./src-tauri/fetch-model.sh                     # ~1.5 GB BF16 weights -> src-tauri/models/
pnpm install
```

Run with `pnpm tauri dev`. The first build compiles MLX (a few minutes).

Tests:

```sh
(cd src-tauri && cargo test)                   # app: chunking, store, pipeline, PDFium
(cd semantra-embed && cargo test --release)    # model
pnpm check                                     # frontend types
```

Fixture-dependent tests skip unless pointed at data: `PDF_FIXTURE` (a PDF),
`EG2_MEDIA_DIR` (sample media), and `EG2_MODEL_DIR` / `EG2_GOLDEN_DIR` for the
model's numerical parity tests against the FP32 PyTorch reference (generate the
golden vectors with `semantra-embed/scripts/make_golden.py`).
