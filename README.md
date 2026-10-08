# Semantra desktop app

Semantra is a multi-tool for semantic search. You can upload images, text/PDF files, videos, audio
and CSVs and index them on your computer entirely locally. You can search the resulting files with
semantic- and keyword-based search.

Semantra currently uses Google's
open-source EmbeddingGemma2 model, which runs entirely locally on your machine.

## Download

| Platform | Download |
|---|---|
| macOS — Apple Silicon (M1 or later) | [**Semantra-macOS-AppleSilicon.dmg**](https://github.com/freedmand/semantra/releases/latest/download/Semantra-macOS-AppleSilicon.dmg) |
| macOS — Intel | [**Semantra-macOS-Intel.dmg**](https://github.com/freedmand/semantra/releases/latest/download/Semantra-macOS-Intel.dmg) |
| Windows 10/11 (64-bit) | [**Semantra-Windows-x64-setup.exe**](https://github.com/freedmand/semantra/releases/latest/download/Semantra-Windows-x64-setup.exe) |
| Linux (64-bit) | [**Semantra-Linux-x86_64.AppImage**](https://github.com/freedmand/semantra/releases/latest/download/Semantra-Linux-x86_64.AppImage) · [**.deb**](https://github.com/freedmand/semantra/releases/latest/download/Semantra-Linux-amd64.deb) |

------------

A.I.-written text follows:

Each download is about 1 GB, because the model is bundled so everything works offline. Not sure
which Mac you have? Apple menu → **About This Mac**: "Chip: Apple M…" means Apple Silicon. Once
installed, Semantra updates itself. Release notes and older versions are on the
[releases page](https://github.com/freedmand/semantra/releases).

NOTE: For the legacy Python codebase, see: http://github.com/freedmand/semantra-python.

## Development

Semantra 0.3.0 runs on Apple Silicon (MLX) and on Windows, Linux and Intel Macs (ONNX Runtime; see below).
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

### Other platforms (ONNX Runtime backend)

Windows, Linux and Intel Macs run the same model through ONNX Runtime instead
of MLX (`semantra-embed` picks the backend at build time; see its `build.rs`).
It uses the [onnx-community fp16 export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX),
which matches the MLX model to cosine ≥ 0.9997 on every modality (explanation
tinting included), and the GPU via WebGPU (D3D12 / Vulkan / Metal) when
available, else the CPU.

```sh
bash src-tauri/scripts/fetch-model-onnx.sh     # ~1.5 GB fp16 export -> src-tauri/models/embeddinggemma-2-onnx/
bash src-tauri/scripts/fetch-onnxruntime.sh    # ONNX Runtime 1.27 for this platform -> src-tauri/onnxruntime/
pnpm install
pnpm tauri build                               # Windows / Linux pick up tauri.<os>.conf.json automatically
pnpm tauri build --config src-tauri/tauri.onnx.conf.json --target x86_64-apple-darwin   # Intel Mac
```

Intel Macs build ONNX Runtime from source (Microsoft stopped shipping x86_64
macOS binaries); `fetch-onnxruntime.sh macos-x64` does it via
`build-onnxruntime-macos-x64.sh` (~30 min, cached). On Apple Silicon, test the
ONNX backend with `cargo test --features onnx` (semantra-embed) or
`pnpm tauri dev --features onnx`, pointing `ORT_DYLIB_PATH` at a runtime;
`examples/backend_parity.rs` compares it to MLX through every `Model` method.

Fixture-dependent tests skip unless pointed at data: `PDF_FIXTURE` (a PDF),
`EG2_MEDIA_DIR` (sample media), and `EG2_MODEL_DIR` / `EG2_GOLDEN_DIR` for the
model's numerical parity tests against the FP32 PyTorch reference (generate the
golden vectors with `semantra-embed/scripts/make_golden.py`).
