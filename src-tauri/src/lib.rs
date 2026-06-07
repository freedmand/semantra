// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod embed;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use embed::{embed_chunks_streaming, EmbedProgress, BATCH_SIZE};
use leaf_ir_candle_test::{embed_sentences, setup_model, warmup, ModelCtx, QUERY_PREFIX};
use tauri::ipc::Channel;
use tauri::{Manager, State};

// The embedding model is loaded once at startup and kept in Tauri managed
// state. `ModelCtx` holds candle tensors + the tokenizer and is not `Sync`, so
// we guard it with a `Mutex`. The `Arc` lets a handle be cloned into the
// blocking inference task without borrowing the (non-'static) `State`.
type SharedModel = Arc<Mutex<ModelCtx>>;

/// Resolve the directory that holds the bundled model files.
///
/// In a bundled app the files live under the resource dir (configured via
/// `bundle.resources` in tauri.conf.json). `tauri dev` does not reliably copy
/// resources next to the dev binary, so we fall back to the source tree
/// (`<crate>/models`) when the resource path is missing.
fn resolve_model_dir(app: &tauri::App) -> Result<PathBuf, String> {
    let resource_models = app
        .path()
        .resolve("models", tauri::path::BaseDirectory::Resource)
        .map_err(|e| format!("failed to resolve resource dir: {e}"))?;
    if resource_models.join("config.json").exists() {
        return Ok(resource_models);
    }

    // Dev fallback: models/ sits next to this crate's Cargo.toml.
    let dev_models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    if dev_models.join("config.json").exists() {
        return Ok(dev_models);
    }

    Err(format!(
        "could not find model files in resource dir ({}) or dev path ({})",
        resource_models.display(),
        dev_models.display()
    ))
}

/// Embed `texts` into 768-d unit-norm vectors.
///
/// When `is_query` is true each text is prefixed with the asymmetric-retrieval
/// query prefix; documents are embedded raw. Inference runs on a blocking
/// thread so it never stalls the Tauri event loop.
#[tauri::command]
async fn embed(
    state: State<'_, SharedModel>,
    texts: Vec<String>,
    is_query: bool,
) -> Result<Vec<Vec<f32>>, String> {
    let inputs: Vec<String> = if is_query {
        texts
            .into_iter()
            .map(|t| format!("{QUERY_PREFIX}{t}"))
            .collect()
    } else {
        texts
    };

    let model = Arc::clone(state.inner());
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<Vec<f32>>, String> {
        let ctx = model.lock().map_err(|_| "model mutex poisoned".to_string())?;
        let emb = embed_sentences(&ctx, &inputs).map_err(|e| e.to_string())?;
        Ok(emb.rows)
    })
    .await
    .map_err(|e| format!("embedding task panicked: {e}"))?
}

/// Embed `chunks` in batches, discarding the vectors, while streaming progress
/// updates back to the frontend over `on_progress`.
///
/// Unlike `embed`, this returns no vectors — it exists to *process* a large list
/// (e.g. a chunked file) and report timing as it goes. The whole batched loop
/// runs on one blocking thread; each `EmbedProgress` is forwarded to the channel
/// as the library callback fires.
#[tauri::command]
async fn embed_chunks(
    state: State<'_, SharedModel>,
    chunks: Vec<String>,
    is_query: bool,
    on_progress: Channel<EmbedProgress>,
) -> Result<(), String> {
    let model = Arc::clone(state.inner());
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let ctx = model.lock().map_err(|_| "model mutex poisoned".to_string())?;
        embed_chunks_streaming(&ctx, chunks, is_query, BATCH_SIZE, |progress| {
            // A send only fails if the frontend dropped the channel; ignore it
            // and let the loop finish (there is nothing useful to do here).
            let _ = on_progress.send(progress);
        })
    })
    .await
    .map_err(|e| format!("embedding task panicked: {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let model_dir = resolve_model_dir(app)?;
            let ctx = setup_model(&model_dir).map_err(|e| e.to_string())?;
            let shared: SharedModel = Arc::new(Mutex::new(ctx));
            app.manage(Arc::clone(&shared));

            // Warm up the embedding kernels in the background so the first real
            // embed doesn't pay Metal's one-time pipeline-compilation cost (~2s).
            // We grab the model lock immediately here (before the frontend can
            // issue a command), so the warmup wins the lock and any embed call
            // that arrives mid-warmup simply blocks on `model.lock()` until it
            // finishes — the existing Mutex IS the "wait until ready" mechanism.
            std::thread::spawn(move || match shared.lock() {
                Ok(ctx) => {
                    if let Err(e) = warmup(&ctx) {
                        eprintln!("[semantra] model warmup failed: {e}");
                    }
                }
                Err(_) => eprintln!("[semantra] model mutex poisoned before warmup"),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![embed, embed_chunks])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
