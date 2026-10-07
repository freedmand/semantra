# Semantra desktop app

README coming soon. Download the installer corresponding to your operating system [here](https://github.com/freedmand/semantra/releases/latest) to get started.

NOTE: For the legacy Python codebase, see: http://github.com/freedmand/semantra-python

## Development

Semantra runs on Apple Silicon Macs. Embeddings come from
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) — one
768-d space for text, images (including rendered PDF pages), audio and video —
running on Apple's MLX (`semantra-embed/`).

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
(cd semantra-embed && cargo test --release)    # model; parity tests need EG2_MODEL_DIR / EG2_GOLDEN_DIR
pnpm check                                     # frontend types
```
