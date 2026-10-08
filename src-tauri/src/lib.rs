// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod catalog;
pub mod chunk;
mod embed;
pub mod extract;
mod explain;
pub mod media;
pub mod pdf;
pub mod query;
pub mod store;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use catalog::{Catalog, FileRecord, Job, JOB_PENDING};
use chunk::{CellChunker, Chunk, Chunker, TokenWindowChunker};
use embed::{plan_batches, BATCH_TOKENS, MAX_BATCH_ROWS};
use pdfium_render::prelude::Pdfium;
use semantra_embed::{document_prompt, EmbedService, Lane, QUERY_PREFIX};
use sha2::{Digest, Sha512};
use store::{ChunkRow, Modality, SearchMode, VectorStore};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::Notify;

// The embedding model lives on its own inference thread (MLX state is
// per-thread); `EmbedService` is a cheap, cloneable handle to it, managed
// directly as Tauri state. Interactive work (search, explain) goes on its
// priority lane, indexing on the background lane.

// The vector store is async (LanceDB) and single-writer, so an async `Mutex`
// guards it directly in managed state.
type SharedStore = AsyncMutex<VectorStore>;

// The metadata catalog shares the same LanceDB connection. Its methods only need
// `&self`, but an async `Mutex` keeps a simple "one op at a time" discipline and
// matches how the store is accessed.
type SharedCatalog = AsyncMutex<Catalog>;

// PDFium keeps a process-global binding, so exactly one `Pdfium` exists for the
// whole app: created once at startup and shared. `thread_safe` makes it
// `Send + Sync`, so an `Arc` (no extra mutex) hands a handle to blocking tasks.
type SharedPdfium = Arc<Pdfium>;

/// Wakes the background indexing worker when new jobs are enqueued. A `tokio`
/// `Notify` stores a single permit, so a `notify_one` that races ahead of the
/// worker's `notified().await` is not lost.
type SharedNotify = Arc<Notify>;

/// The file the background worker is currently indexing (if any), with live
/// chunk progress. Read by `project_status` for the initial UI before any event
/// arrives; written by the worker. A `std` mutex (not async) so the synchronous
/// per-batch progress callback can update it without awaiting.
type SharedActive = Arc<Mutex<Option<ActiveJob>>>;

/// Snapshot of the in-flight indexing job.
#[derive(Clone, Debug, Default)]
struct ActiveJob {
    project_id: String,
    sha512: String,
    basename: String,
    done: usize,
    total: usize,
}

/// On-disk locations the app writes to (under the OS app-data dir). Copied-in
/// source files live in `files_dir`, keyed by their SHA-512.
#[derive(Clone)]
struct AppPaths {
    files_dir: PathBuf,
}

// Vectors buffered to this many rows before a single LanceDB append.
const INSERT_BATCH: usize = 1024;

// New chunks the worker lets accumulate before refreshing the search indexes
// mid-run (see `index_worker`). Index rebuilds are otherwise deferred to when
// the queue drains; this bounds how stale keyword search can get during a very
// large import without paying the rebuild cost on every file.
const MAINTENANCE_ROW_THRESHOLD: usize = 50_000;

/// Tokens per text chunk (excluding the document prompt) and the rewind shared
/// between consecutive windows; windows snap to sentence ends (see
/// [`TokenWindowChunker::snap_to_sentences`]) and rewind by whole sentences.
/// A retrieval eval (SQuAD articles as documents) found quality flat from 64 to
/// 256 tokens; 128 keeps a hit short enough to read in full in the results
/// list. Both are baked into [`text_pipeline_version`], so changing either
/// re-chunks text (only) on the next launch.
const CHUNK_TOKENS: usize = 128;
const CHUNK_OVERLAP_TOKENS: usize = 16;

/// The embedding model directory under `models/` (bundled as a resource and in
/// the dev tree): a copy of `google/embeddinggemma-2` (see fetch-model.sh).
pub const MODEL_NAME: &str = "embeddinggemma-2";

/// Matryoshka output width. 768 is the model's native size; 512/256 trade a
/// little quality for 1.5x/3x smaller vectors.
pub const EMBED_DIM: usize = 768;

/// Identifies the embedding approach behind every stored vector: model, output
/// width, precision and prompt. Changing it invalidates all vectors (text and
/// media), so the whole library is re-indexed. Bump the schema rev by hand only
/// for changes not already captured here (v2: store schema; v3: PDF page
/// renders at full resolution instead of PDFium's 72 dpi default).
pub fn pipeline_version() -> String {
    format!("v3:{MODEL_NAME}@{EMBED_DIM}:bf16:prompt-none")
}

/// [`pipeline_version`] plus the text chunk geometry, stamped on text rows.
/// When only this changes, a background pass re-chunks text and keeps every
/// media vector (see [`reindex_text`]).
pub fn text_pipeline_version() -> String {
    format!(
        "{}:tokenwindow:{CHUNK_TOKENS}-{CHUNK_OVERLAP_TOKENS}:sentences-v3",
        pipeline_version()
    )
}

/// The document prompt every text chunk is embedded with. Titles (file names)
/// measured as a wash for retrieval, and leaving them out keeps a chunk's vector
/// independent of what its file is called.
fn doc_prompt() -> String {
    document_prompt(None)
}

/// `db_meta` key under which the [`pipeline_version`] the on-disk vectors were
/// built with is stored, so a model change is detected at startup.
const DB_MODEL_KEY: &str = "active_pipeline";

/// `db_meta` key for the [`text_pipeline_version`] all text rows are known to
/// match; set once a text-only re-index pass completes.
const DB_TEXT_KEY: &str = "active_text_pipeline";

/// Model folder under `models/` for this build's backend: the BF16
/// safetensors checkout for MLX (Apple Silicon), the fp16 ONNX export
/// elsewhere (see fetch-model.sh / fetch-model-onnx.sh).
fn model_folder() -> String {
    match semantra_embed::BACKEND {
        "onnx" => format!("{MODEL_NAME}-onnx"),
        _ => MODEL_NAME.to_string(),
    }
}

/// Resolve the directory that holds the bundled model files (resource dir, then
/// dev-tree fallback — `tauri dev` does not reliably copy resources).
fn resolve_model_dir(app: &tauri::App) -> Result<PathBuf, String> {
    let rel = format!("models/{}", model_folder());
    let resource_models = app
        .path()
        .resolve(&rel, tauri::path::BaseDirectory::Resource)
        .map_err(|e| format!("failed to resolve resource dir: {e}"))?;
    if resource_models.join("config.json").exists() {
        return Ok(resource_models);
    }
    let dev_models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&rel);
    if dev_models.join("config.json").exists() {
        return Ok(dev_models);
    }
    Err(format!(
        "could not find model files in resource dir ({}) or dev path ({})",
        resource_models.display(),
        dev_models.display()
    ))
}

/// Resolve the directory holding the bundled PDFium dynamic library (resource
/// dir, then dev-tree fallback populated by `build.rs`).
fn resolve_pdfium_dir(app: &tauri::App) -> Result<PathBuf, String> {
    let resource = app
        .path()
        .resolve("libpdfium", tauri::path::BaseDirectory::Resource)
        .map_err(|e| format!("failed to resolve resource dir: {e}"))?;
    if pdf::library_present(&resource) {
        return Ok(resource);
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("libpdfium");
    if pdf::library_present(&dev) {
        return Ok(dev);
    }
    Err(format!(
        "could not find PDFium library in resource dir ({}) or dev path ({})",
        resource.display(),
        dev.display()
    ))
}

/// File name of the ONNX Runtime library bundled for this platform (fetched by
/// scripts/fetch-onnxruntime.sh into `onnxruntime/`).
const ONNXRUNTIME_LIB: &str = if cfg!(target_os = "windows") {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

/// The bundled ONNX Runtime library (resource dir, then dev tree), for the ONNX
/// backend. `None` leaves the backend to `ORT_DYLIB_PATH` / the system library.
fn resolve_onnxruntime(app: &tauri::App) -> Option<PathBuf> {
    let rel = format!("onnxruntime/{ONNXRUNTIME_LIB}");
    let resource = app.path().resolve(&rel, tauri::path::BaseDirectory::Resource).ok();
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&rel);
    resource.into_iter().chain([dev]).find(|p| p.exists())
}

/// Current wall-clock in epoch milliseconds (for `created_at`/`added_at`).
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Lowercase hex of a byte slice (for SHA-512 → string key).
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Embed `chunks` (with their metadata) and insert them into the store.
///
/// Chunks are sorted by token length and packed into token-budgeted batches
/// (see [`plan_batches`]) that run on the inference thread's background lane.
/// Two batches are kept in flight so the GPU never idles while a finished batch
/// is converted and inserted. `report` receives the cumulative number of
/// chunks embedded so far.
async fn run_index_pipeline(
    embedder: &EmbedService,
    store: &SharedStore,
    chunks: Vec<Chunk>,
    sha512: String,
    report: &(dyn Fn(usize) + Send + Sync),
) -> Result<(), String> {
    let prompt = doc_prompt();
    let inputs: Arc<Vec<String>> =
        Arc::new(chunks.iter().map(|c| format!("{prompt}{}", c.text)).collect());
    let tokenizer = Arc::clone(embedder.tokenizer());
    let for_lens = Arc::clone(&inputs);
    let lens = tauri::async_runtime::spawn_blocking(move || -> Result<Vec<usize>, String> {
        let enc = tokenizer
            .encode_batch(for_lens.to_vec(), true)
            .map_err(|e| format!("tokenize chunks: {e}"))?;
        // The shared tokenizer pads batches (BatchLongest), so count real
        // tokens from the attention mask, not `len()` (the padded length).
        Ok(enc.iter().map(|e| e.get_attention_mask().iter().filter(|&&m| m == 1).count()).collect())
    })
    .await
    .map_err(|e| format!("tokenize task panicked: {e}"))??;
    let batches = plan_batches(&lens, BATCH_TOKENS, MAX_BATCH_ROWS);

    let submit = |batch: &[usize]| {
        let rows: Vec<String> = batch.iter().map(|&i| inputs[i].clone()).collect();
        embedder.submit(Lane::Background, move |m| Ok(m.embed_texts(&rows)?.rows))
    };
    let mut queued = batches.iter();
    let mut in_flight = VecDeque::new();
    for b in queued.by_ref().take(2) {
        in_flight.push_back((b, submit(b)));
    }

    let mut buf: Vec<ChunkRow> = Vec::new();
    let mut done = 0usize;
    while let Some((batch, rx)) = in_flight.pop_front() {
        let vectors = rx
            .await
            .map_err(|_| "embedding thread stopped".to_string())?
            .map_err(|e| e.to_string())?;
        if let Some(b) = queued.next() {
            in_flight.push_back((b, submit(b)));
        }
        for (&i, vector) in batch.iter().zip(vectors) {
            let c = &chunks[i];
            buf.push(ChunkRow {
                sha512: sha512.clone(),
                modality: Modality::Text,
                text: c.text.clone(),
                char_start: c.char_start as i64,
                char_end: c.char_end as i64,
                page: c.page.map(|p| p as i64),
                page_char_start: c.page_char_start as i64,
                time_start_ms: None,
                time_end_ms: None,
                pipeline_version: text_pipeline_version(),
                vector,
            });
        }
        // Lock the store only for each insert (not across the embedding waits
        // in between), so indexing doesn't block concurrent searches/reads.
        if buf.len() >= INSERT_BATCH {
            store.lock().await.insert(&buf).await?;
            buf.clear();
        }
        done += batch.len();
        report(done);
    }
    if !buf.is_empty() {
        store.lock().await.insert(&buf).await?;
    }
    // Index (re)builds and compaction are intentionally NOT done here: each one
    // rebuilds over the whole table, so running them per file makes a multi-file
    // import O(files²). The worker refreshes them off this hot path — once when
    // the queue drains and occasionally mid-run (see `index_worker`).
    Ok(())
}

/// Embed a file's media units (PDF page renders, images, audio/video windows;
/// see `media.rs`) and insert them. Decoding runs on a blocking thread that
/// streams rows to this task; `report` receives cumulative units embedded.
async fn run_media_pipeline(
    app: &AppHandle,
    store: &SharedStore,
    plan: media::Plan,
    copied_path: String,
    sha512: String,
    report: Arc<dyn Fn(usize) + Send + Sync>,
) -> Result<(), String> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<ChunkRow>>(4);
    let embedder = (*app.state::<EmbedService>()).clone();
    let pdfium = (*app.state::<SharedPdfium>()).clone();
    let producer = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let mut done = 0usize;
        media::run(plan, &copied_path, &pdfium, &embedder, |rows, units| {
            let rows = rows
                .into_iter()
                .map(|r| ChunkRow {
                    sha512: sha512.clone(),
                    modality: r.modality,
                    text: String::new(),
                    char_start: 0,
                    char_end: 0,
                    page: r.page,
                    page_char_start: 0,
                    time_start_ms: r.time_ms.map(|t| t.0),
                    time_end_ms: r.time_ms.map(|t| t.1),
                    pipeline_version: pipeline_version(),
                    vector: r.vector,
                })
                .collect();
            tx.blocking_send(rows).map_err(|_| "media inserter stopped".to_string())?;
            done += units;
            report(done);
            Ok(())
        })
    });
    while let Some(rows) = rx.recv().await {
        store.lock().await.insert(&rows).await?;
    }
    producer
        .await
        .map_err(|e| format!("media task panicked: {e}"))??;
    Ok(())
}

// === Project CRUD =====================================================

/// A project row for the listing, enriched with document + indexing counts.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectListItem {
    project_id: String,
    name: String,
    created_at: i64,
    /// Files committed (fully indexed) into the project.
    doc_count: usize,
    /// Files still queued for indexing.
    pending_count: usize,
    /// Files whose indexing failed.
    error_count: usize,
}

/// List all projects with their document + indexing-queue counts.
#[tauri::command]
async fn list_projects(catalog: State<'_, SharedCatalog>) -> Result<Vec<ProjectListItem>, String> {
    let cat = catalog.lock().await;
    let projects = cat.list_projects().await?;
    let mut out = Vec::with_capacity(projects.len());
    for p in projects {
        let doc_count = cat.project_file_shas(&p.project_id).await?.len();
        let (pending_count, error_count) = cat.job_counts(&p.project_id).await?;
        out.push(ProjectListItem {
            project_id: p.project_id,
            name: p.name,
            created_at: p.created_at,
            doc_count,
            pending_count,
            error_count,
        });
    }
    Ok(out)
}

/// Create a project. `project_id` is a client-generated UUID; idempotent.
#[tauri::command]
async fn create_project(
    catalog: State<'_, SharedCatalog>,
    project_id: String,
    name: String,
) -> Result<(), String> {
    catalog
        .lock()
        .await
        .ensure_project(&project_id, &name, now_ms())
        .await
}

/// Rename a project (CRUD update), preserving its creation time.
#[tauri::command]
async fn rename_project(
    catalog: State<'_, SharedCatalog>,
    project_id: String,
    name: String,
) -> Result<(), String> {
    catalog.lock().await.set_project_name(&project_id, &name).await
}

/// Queue one or more files for background indexing into a project.
///
/// Each file's bytes are read + SHA-512'd + copied into the app-data `files/`
/// dir up front (so a resume after a crash never needs the user's original
/// path). Already-indexed bytes are referenced instantly (content-addressed
/// dedup, no job); genuinely new files get a `pending` row in the durable job
/// queue. Indexing itself happens on the background worker — this returns as
/// soon as the work is enqueued so the project can be launched while it runs.
#[tauri::command]
async fn add_files_to_project(
    catalog: State<'_, SharedCatalog>,
    app_paths: State<'_, AppPaths>,
    notify: State<'_, SharedNotify>,
    project_id: String,
    paths: Vec<String>,
) -> Result<(), String> {
    let files_dir = app_paths.files_dir.clone();
    let mut enqueued = false;
    for path in paths {
        let src = path.clone();
        let dir = files_dir.clone();
        // Hash + copy into the content-addressed store on a blocking thread.
        // The copy target name is deterministic (`<sha>[.ext]`), so re-copying
        // identical bytes is harmless.
        let (sha512, basename, ext, copied_path) =
            tauri::async_runtime::spawn_blocking(move || -> Result<(String, String, String, String), String> {
                // Stream the hash (multi-GB videos are accepted, so never read
                // the whole file into memory).
                let mut hasher = Sha512::new();
                let mut file = std::fs::File::open(&src).map_err(|e| format!("read {src}: {e}"))?;
                std::io::copy(&mut file, &mut hasher).map_err(|e| format!("read {src}: {e}"))?;
                let sha512 = hex(&hasher.finalize());
                let p = std::path::Path::new(&src);
                let basename = p
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| src.clone());
                let ext = p
                    .extension()
                    .map(|s| s.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let copied = dir.join(if ext.is_empty() {
                    sha512.clone()
                } else {
                    format!("{sha512}.{ext}")
                });
                // `fs::copy` clones on APFS (copy-on-write: instant, no extra
                // disk); skip it entirely if these bytes are already stored.
                // Copy to a temp name and rename, so an interrupted copy never
                // leaves a truncated file that a later import would trust.
                if !copied.exists() {
                    let tmp = copied.with_extension("partial");
                    std::fs::copy(&src, &tmp).map_err(|e| format!("copy into app dir: {e}"))?;
                    std::fs::rename(&tmp, &copied).map_err(|e| format!("copy into app dir: {e}"))?;
                }
                Ok((sha512, basename, ext, copied.to_string_lossy().to_string()))
            })
            .await
            .map_err(|e| format!("file read task panicked: {e}"))??;

        // Already fully indexed → reference instantly, no job needed.
        if catalog.lock().await.get_file(&sha512).await?.is_some() {
            catalog
                .lock()
                .await
                .add_file_ref(&project_id, &sha512, &path, now_ms())
                .await?;
            continue;
        }

        catalog
            .lock()
            .await
            .enqueue_job(&Job {
                project_id: project_id.clone(),
                sha512,
                source_path: path,
                copied_path,
                basename,
                ext,
                status: JOB_PENDING.to_string(),
                error: String::new(),
                added_at: now_ms(),
            })
            .await?;
        enqueued = true;
    }
    if enqueued {
        notify.notify_one();
    }
    Ok(())
}

/// Remove one file from a project: cancel any queued/failed job for it, drop the
/// project's reference, and GC its chunks + copied bytes if nothing else
/// references it. If the file is mid-index, the worker's pre-commit re-check
/// notices the cancellation and discards its work.
#[tauri::command]
async fn delete_file_from_project(
    store: State<'_, SharedStore>,
    catalog: State<'_, SharedCatalog>,
    app_paths: State<'_, AppPaths>,
    project_id: String,
    sha512: String,
) -> Result<(), String> {
    catalog.lock().await.delete_job(&project_id, &sha512).await?;
    let orphaned = catalog.lock().await.remove_file_ref(&project_id, &sha512).await?;
    if orphaned {
        store.lock().await.delete_file(&sha512).await?;
        remove_file_bytes(&app_paths.files_dir, &sha512);
    }
    Ok(())
}

/// Retry a file whose indexing failed: flip its job back to pending and wake
/// the worker.
#[tauri::command]
async fn retry_file(
    catalog: State<'_, SharedCatalog>,
    notify: State<'_, SharedNotify>,
    project_id: String,
    sha512: String,
) -> Result<(), String> {
    catalog.lock().await.set_job_pending(&project_id, &sha512).await?;
    notify.notify_one();
    Ok(())
}

/// A queued/failed file's status for the manage view.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JobStatus {
    sha512: String,
    basename: String,
    status: String,
    error: String,
}

/// The file currently being indexed, with live chunk progress.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ActiveStatus {
    sha512: String,
    basename: String,
    done: usize,
    total: usize,
}

/// A project's indexing state: queued/failed jobs plus the in-flight file.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectStatus {
    jobs: Vec<JobStatus>,
    active: Option<ActiveStatus>,
}

/// Current indexing state of a project — the initial snapshot the UI renders
/// before live `index://progress` events take over.
#[tauri::command]
async fn project_status(
    catalog: State<'_, SharedCatalog>,
    active: State<'_, SharedActive>,
    project_id: String,
) -> Result<ProjectStatus, String> {
    let jobs = catalog
        .lock()
        .await
        .jobs_for(&project_id)
        .await?
        .into_iter()
        .map(|j| JobStatus {
            sha512: j.sha512,
            basename: j.basename,
            status: j.status,
            error: j.error,
        })
        .collect();
    let active = active
        .lock()
        .unwrap()
        .clone()
        .filter(|a| a.project_id == project_id)
        .map(|a| ActiveStatus {
            sha512: a.sha512,
            basename: a.basename,
            done: a.done,
            total: a.total,
        });
    Ok(ProjectStatus { jobs, active })
}

// === Background indexing worker =======================================

/// Tauri event channel for indexing progress (listened to on the frontend).
const INDEX_EVENT: &str = "index-progress";

/// One indexing-progress update pushed to the frontend. `kind` is one of
/// `started` (a file began), `progress` (a batch finished — `done`/`total` are
/// chunk counts), `fileDone` (a file committed; refresh the doc list), or
/// `error` (indexing failed, see `error`). `queueRemaining` is how many pending
/// files are left in this project.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexEvent {
    project_id: String,
    sha512: String,
    basename: String,
    kind: String,
    done: usize,
    total: usize,
    queue_remaining: usize,
    error: String,
}

fn emit_index(
    app: &AppHandle,
    job: &Job,
    kind: &str,
    done: usize,
    total: usize,
    queue_remaining: usize,
    error: &str,
) {
    let _ = app.emit(
        INDEX_EVENT,
        IndexEvent {
            project_id: job.project_id.clone(),
            sha512: job.sha512.clone(),
            basename: job.basename.clone(),
            kind: kind.to_string(),
            done,
            total,
            queue_remaining,
            error: error.to_string(),
        },
    );
}

/// Set (or clear) the shared "currently indexing" snapshot.
fn set_active(active: &SharedActive, value: Option<ActiveJob>) {
    if let Ok(mut g) = active.lock() {
        *g = value;
    }
}

/// Best-effort removal of a file's copied bytes (named `<sha>[.ext]`).
fn remove_file_bytes(files_dir: &std::path::Path, sha512: &str) {
    if let Ok(entries) = std::fs::read_dir(files_dir) {
        for e in entries.flatten() {
            if e.file_name().to_string_lossy().starts_with(sha512) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

/// Build the progress sink for a file of `total` units (text chunks + media
/// units): it updates the shared active snapshot and emits a `progress` event
/// per batch. Called with the cumulative units done.
fn progress_sink(
    app: AppHandle,
    active: SharedActive,
    job: Job,
    queue_remaining: usize,
    total: usize,
) -> Arc<dyn Fn(usize) + Send + Sync> {
    Arc::new(move |done: usize| {
        let done = done.min(total);
        if let Ok(mut g) = active.lock() {
            if let Some(a) = g.as_mut() {
                if a.sha512 == job.sha512 {
                    a.done = done;
                    a.total = total;
                }
            }
        }
        emit_index(&app, &job, "progress", done, total, queue_remaining, "");
    })
}

/// Refresh the store's search indexes + compaction when at least `threshold`
/// new chunks have been inserted since the last pass. Holds the store guard
/// exclusively (required by `compact`'s version-prune). Best-effort: logs and
/// returns on error so the indexing worker keeps running.
async fn maybe_maintain_store(app: &AppHandle, threshold: usize) {
    let store = app.state::<SharedStore>();
    let pending = match store.lock().await.rows_since_maintenance().await {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[semantra] indexing worker: rows_since_maintenance failed: {e}");
            return;
        }
    };
    if pending < threshold {
        return;
    }
    let result = store.lock().await.maintain().await;
    if let Err(e) = result {
        eprintln!("[semantra] indexing worker: index maintenance failed: {e}");
    }
}

/// Background worker loop: drain the durable job queue one file at a time,
/// sleeping on the notifier when it is empty. Runs for the life of the app.
async fn index_worker(app: AppHandle) {
    loop {
        let next = app.state::<SharedCatalog>().lock().await.next_pending_job().await;
        match next {
            Ok(Some(job)) => {
                if let Err(e) = process_job(&app, &job).await {
                    let catalog = app.state::<SharedCatalog>();
                    let _ = catalog.lock().await.mark_job_error(&job.project_id, &job.sha512, &e).await;
                    set_active(&app.state::<SharedActive>(), None);
                    let remaining = catalog
                        .lock()
                        .await
                        .job_counts(&job.project_id)
                        .await
                        .map(|c| c.0)
                        .unwrap_or(0);
                    emit_index(&app, &job, "error", 0, 0, remaining, &e);
                } else {
                    // Mid-run refresh once enough new chunks have piled up, so
                    // keyword search doesn't stay stale through a huge import.
                    // Best-effort: on failure the indexes just stay stale until
                    // the queue drains — don't fail the file or stop the worker.
                    maybe_maintain_store(&app, MAINTENANCE_ROW_THRESHOLD).await;
                }
            }
            // Queue drained — refresh indexes once for everything added since the
            // last pass, then wait to be woken.
            Ok(None) => {
                maybe_maintain_store(&app, 1).await;
                (*app.state::<SharedNotify>()).clone().notified().await
            }
            Err(e) => {
                eprintln!("[semantra] indexing worker: read queue failed: {e}");
                (*app.state::<SharedNotify>()).clone().notified().await;
            }
        }
    }
}

/// Process one job: extract → chunk → embed → insert, then commit the file and
/// its project reference. Idempotent and crash-safe:
/// - if the bytes are already committed (dedup, or a crash after commit but
///   before the job row was deleted), just (re)reference and finish;
/// - otherwise clear any partial chunks left by a prior crash before redoing;
/// - and re-check the job still exists before committing, so a file deleted
///   from the project mid-index is discarded rather than resurrected.
async fn process_job(app: &AppHandle, job: &Job) -> Result<(), String> {
    let catalog = app.state::<SharedCatalog>();
    let store = app.state::<SharedStore>();
    let active = app.state::<SharedActive>();

    if catalog.lock().await.get_file(&job.sha512).await?.is_some() {
        catalog
            .lock()
            .await
            .add_file_ref(&job.project_id, &job.sha512, &job.source_path, now_ms())
            .await?;
        catalog.lock().await.delete_job(&job.project_id, &job.sha512).await?;
        let remaining = catalog.lock().await.job_counts(&job.project_id).await?.0;
        emit_index(app, job, "fileDone", 0, 0, remaining, "");
        return Ok(());
    }

    // Clear any partial chunks from a crash mid-embed before reprocessing.
    store.lock().await.delete_file(&job.sha512).await?;

    set_active(
        &active,
        Some(ActiveJob {
            project_id: job.project_id.clone(),
            sha512: job.sha512.clone(),
            basename: job.basename.clone(),
            done: 0,
            total: 0,
        }),
    );
    let queue_remaining = catalog.lock().await.job_counts(&job.project_id).await?.0;
    emit_index(app, job, "started", 0, 0, queue_remaining, "");

    // Extract from the copied bytes on a blocking thread.
    let pdfium = (*app.state::<SharedPdfium>()).clone();
    let copied = job.copied_path.clone();
    let extracted = tauri::async_runtime::spawn_blocking(move || extract::extract(&pdfium, &copied))
        .await
        .map_err(|e| format!("extract task panicked: {e}"))??;

    let char_len = extracted.full_text.chars().count() as i64;
    let word_count = extracted.full_text.split_whitespace().count() as i64;
    let filetype = extracted.filetype;
    let page_count = extracted.page_count.map(|p| p as i64);
    let byte_len = std::fs::metadata(&job.copied_path)
        .map(|m| m.len() as i64)
        .unwrap_or(0);

    let embedder = (*app.state::<EmbedService>()).clone();
    // The media planner only needs page texts (to budget PDF page renders).
    let extracted_meta = extract::Extracted {
        filetype,
        full_text: String::new(),
        segments: extracted.segments.clone(),
        page_count: extracted.page_count,
    };
    let for_chunking = embedder.clone();
    let chunks = tauri::async_runtime::spawn_blocking(move || chunk_segments(&for_chunking, filetype, &extracted.segments))
        .await
        .map_err(|e| format!("chunk task panicked: {e}"))?;

    // Media pass (PDF page renders, images, audio/video windows): planned up
    // front so one progress total covers text + media.
    let copied = job.copied_path.clone();
    let plan = tauri::async_runtime::spawn_blocking(move || media::plan(filetype, &copied, &extracted_meta))
        .await
        .map_err(|e| format!("media plan task panicked: {e}"))??;
    let n_text = chunks.len();
    let total = n_text + plan.units();
    let sink = progress_sink((*app).clone(), (*active).clone(), job.clone(), queue_remaining, total);
    sink(0);
    run_index_pipeline(&embedder, &store, chunks, job.sha512.clone(), &*sink).await?;
    let media_sink: Arc<dyn Fn(usize) + Send + Sync> = {
        let sink = Arc::clone(&sink);
        Arc::new(move |done| sink(n_text + done))
    };
    let media = run_media_pipeline(app, &store, plan, job.copied_path.clone(), job.sha512.clone(), media_sink).await;
    match media {
        // A PDF's page renders are a bonus on top of its text: if they fail
        // (an unrenderable page, a checkpoint without the vision tower), keep
        // the text index rather than failing the whole file.
        Err(e) if filetype == extract::FileType::Pdf => {
            eprintln!("[semantra] {}: page images skipped: {e}", job.basename);
        }
        other => other?,
    }

    // Cancelled (file deleted from the project) while embedding? Discard the work.
    if !catalog.lock().await.job_exists(&job.project_id, &job.sha512).await? {
        store.lock().await.delete_file(&job.sha512).await?;
        remove_file_bytes(&app_paths_dir(app), &job.sha512);
        set_active(&active, None);
        return Ok(());
    }

    {
        let cat = catalog.lock().await;
        cat.insert_file(&FileRecord {
            sha512: job.sha512.clone(),
            basename: job.basename.clone(),
            ext: job.ext.clone(),
            copied_path: job.copied_path.clone(),
            filetype: filetype.as_str().to_string(),
            byte_len,
            page_count,
            char_len,
            word_count,
            pipeline_version: pipeline_version(),
            created_at: now_ms(),
        })
        .await?;
        cat.add_file_ref(&job.project_id, &job.sha512, &job.source_path, now_ms()).await?;
    }
    catalog.lock().await.delete_job(&job.project_id, &job.sha512).await?;
    set_active(&active, None);
    let remaining = catalog.lock().await.job_counts(&job.project_id).await?.0;
    emit_index(app, job, "fileDone", 0, 0, remaining, "");
    Ok(())
}

/// Chunk extracted segments for embedding. CSVs index one chunk per cell (the
/// segments are already cells); everything else uses sentence-snapped token
/// windows sized by the model's own tokenizer, minus windows with no words.
pub fn chunk_segments(embedder: &EmbedService, filetype: extract::FileType, segments: &[chunk::Segment]) -> Vec<Chunk> {
    // The shared tokenizer truncates at the model's input cap (2048 tokens),
    // which would leave the rest of a long segment (a whole plain-text file,
    // a dense page) without token offsets — costed ~1 token per word, so its
    // windows overran the budget. Count every token of the segment instead.
    let mut tokenizer = (**embedder.tokenizer()).clone();
    if let Err(e) = tokenizer.with_truncation(None) {
        eprintln!("[semantra] disable tokenizer truncation for chunking failed: {e}");
    }
    match filetype {
        extract::FileType::Csv => CellChunker.chunk(segments),
        _ => TokenWindowChunker::new(CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, |text: &str| {
            match tokenizer.encode(text, false) {
                Ok(e) => e.get_offsets().iter().map(|o| o.0).collect(),
                Err(e) => {
                    // Falls back to ~1 token per word; embedding inputs are
                    // still capped by the model's truncation (MAX_INPUT_TOKENS).
                    eprintln!("[semantra] tokenize segment for chunking failed: {e}");
                    Vec::new()
                }
            }
        })
        .snap_to_sentences()
        .chunk(segments)
        .into_iter()
        .filter(|c| chunk::has_words(&c.text))
        .collect(),
    }
}

/// The app-data `files/` dir (worker helper).
fn app_paths_dir(app: &AppHandle) -> PathBuf {
    app.state::<AppPaths>().files_dir.clone()
}

/// Explain why each chunk in `texts` matched `query` (per-token attribution).
#[tauri::command]
async fn explain_matches(
    embedder: State<'_, EmbedService>,
    query: String,
    texts: Vec<String>,
) -> Result<Vec<explain::Explanation>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let prefixed = format!("{QUERY_PREFIX}{query}");
    let prompt = doc_prompt();
    embedder
        .run(Lane::Priority, move |m| {
            let qvec = m
                .embed_texts(&[prefixed])?
                .rows
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("query produced no embedding"))?;
            let docs: Vec<(String, String)> = texts.into_iter().map(|t| (prompt.clone(), t)).collect();
            let core = m.explain_similarity_batch(&qvec, &docs)?;
            Ok(core.into_iter().map(Into::into).collect())
        })
        .await
        .map_err(|e| e.to_string())
}

/// Delete a project entirely: drop its membership + outstanding jobs, then GC the
/// files (chunks + copied bytes) it was the last referencer of.
#[tauri::command]
async fn delete_project(
    store: State<'_, SharedStore>,
    catalog: State<'_, SharedCatalog>,
    app_paths: State<'_, AppPaths>,
    project_id: String,
) -> Result<(), String> {
    let orphaned = catalog.lock().await.delete_project(&project_id).await?;
    {
        let mut s = store.lock().await;
        for sha in &orphaned {
            s.delete_file(sha).await?;
        }
    }
    for sha in &orphaned {
        remove_file_bytes(&app_paths.files_dir, sha);
    }
    Ok(())
}

// === In-project interface commands ====================================

/// A document in the project, for the tab bar + sidebar.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct DocMeta {
    sha512: String,
    basename: String,
    filetype: String,
    page_count: Option<i64>,
    byte_len: i64,
    word_count: i64,
}

/// One search hit enriched for the in-project UI. `index` is the stable chunk id
/// (used as a preference key).
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectHit {
    index: i64,
    sha512: String,
    basename: String,
    filetype: String,
    /// What this hit's vector embeds (text, image/page render, audio, video).
    modality: Modality,
    text: String,
    distance: f32,
    score: f32,
    char_start: i64,
    char_end: i64,
    page: Option<i64>,
    page_char_start: i64,
    /// Audio/video window span in milliseconds.
    time_start_ms: Option<i64>,
    time_end_ms: Option<i64>,
}

/// A highlight overlay for one PDF page: rectangles in PDF user-space points
/// (origin bottom-left), plus the page size so the frontend can map onto the
/// PDF.js viewport.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct PageHighlight {
    page_width: f32,
    page_height: f32,
    /// Each rect is `[left, bottom, right, top]` in points.
    rects: Vec<[f32; 4]>,
}

/// List the documents in a project (committed files only).
#[tauri::command]
async fn list_documents(
    catalog: State<'_, SharedCatalog>,
    project_id: String,
) -> Result<Vec<DocMeta>, String> {
    let files = catalog.lock().await.list_project_files(&project_id).await?;
    Ok(files
        .into_iter()
        .map(|f| DocMeta {
            sha512: f.sha512,
            basename: f.basename,
            filetype: f.filetype,
            page_count: f.page_count,
            byte_len: f.byte_len,
            word_count: f.word_count,
        })
        .collect())
}

/// Canonical full text of a document (for the flat text reader + offset math).
/// Reconstructed on demand from the copied file.
#[tauri::command]
async fn get_document_text(
    catalog: State<'_, SharedCatalog>,
    pdfium: State<'_, SharedPdfium>,
    sha512: String,
) -> Result<String, String> {
    let file = catalog
        .lock()
        .await
        .get_file(&sha512)
        .await?
        .ok_or_else(|| format!("unknown file {sha512}"))?;
    let pdfium = Arc::clone(pdfium.inner());
    tauri::async_runtime::spawn_blocking(move || {
        extract::extract(&pdfium, &file.copied_path).map(|e| e.full_text)
    })
    .await
    .map_err(|e| format!("text task panicked: {e}"))?
}

/// Filesystem path of a document's copied bytes. The frontend wraps it with
/// `convertFileSrc` to hand PDF.js an asset URL.
#[tauri::command]
async fn get_pdf_src(catalog: State<'_, SharedCatalog>, sha512: String) -> Result<String, String> {
    let file = catalog
        .lock()
        .await
        .get_file(&sha512)
        .await?
        .ok_or_else(|| format!("unknown file {sha512}"))?;
    Ok(file.copied_path)
}

/// A small JPEG preview as a `data:` URL: a PDF page (`page`), a video frame
/// (`time_ms`), or the image itself — also how formats the webview can't
/// display (camera RAW, JPEG 2000) are shown. Longer side `max_side` px.
#[tauri::command]
async fn get_thumbnail(
    catalog: State<'_, SharedCatalog>,
    pdfium: State<'_, SharedPdfium>,
    sha512: String,
    page: Option<usize>,
    time_ms: Option<i64>,
    max_side: u32,
) -> Result<String, String> {
    use base64::Engine;
    use semantra_embed::media::{av, image};
    let file = catalog
        .lock()
        .await
        .get_file(&sha512)
        .await?
        .ok_or_else(|| format!("unknown file {sha512}"))?;
    let pdfium = Arc::clone(pdfium.inner());
    let max_side = max_side.clamp(32, 2048);
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let path = std::path::Path::new(&file.copied_path);
        let rgb = match (extract::FileType::parse(&file.filetype)?, page, time_ms) {
            (extract::FileType::Pdf, Some(p), _) => pdf::render_page(&pdfium, &file.copied_path, p, max_side)?,
            (extract::FileType::Video, _, t) => {
                let secs = t.unwrap_or(0) as f64 / 1000.0;
                av::frames_at(path, &[secs], max_side, 0.5)
                    .map_err(|e| e.to_string())?
                    .pop()
                    .ok_or("no frame")?
            }
            (extract::FileType::Image, _, _) => image::decode_max(path, max_side).map_err(|e| e.to_string())?,
            (ft, _, _) => return Err(format!("no thumbnail for {}", ft.as_str())),
        };
        let jpeg = image::encode_jpeg(&rgb, 0.82).map_err(|e| e.to_string())?;
        Ok(format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(jpeg)
        ))
    })
    .await
    .map_err(|e| format!("thumbnail task panicked: {e}"))?
}

/// Peak amplitude (0–1) of the file's first audio track in `buckets` equal
/// time slices, for the player's waveform. Streams the decode, so long
/// recordings never sit in memory.
#[tauri::command]
async fn get_waveform(
    catalog: State<'_, SharedCatalog>,
    sha512: String,
    buckets: usize,
) -> Result<Vec<f32>, String> {
    use semantra_embed::media::{av, mel};
    let file = catalog
        .lock()
        .await
        .get_file(&sha512)
        .await?
        .ok_or_else(|| format!("unknown file {sha512}"))?;
    let buckets = buckets.clamp(1, 8192);
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<f32>, String> {
        let path = std::path::Path::new(&file.copied_path);
        let probe = av::probe(path).map_err(|e| e.to_string())?;
        if !probe.has_audio {
            return Ok(Vec::new());
        }
        let total = (probe.duration_s * mel::SAMPLE_RATE as f64).max(1.0);
        let per = (total / buckets as f64).max(1.0);
        let mut peaks = vec![0f32; buckets];
        let mut i = 0usize;
        av::stream_audio(path, |chunk| {
            for &x in chunk {
                let b = ((i as f64 / per) as usize).min(buckets - 1);
                peaks[b] = peaks[b].max(x.abs());
                i += 1;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok(peaks)
    })
    .await
    .map_err(|e| format!("waveform task panicked: {e}"))?
}

/// Parsed grid (header row + data rows, all cells verbatim) of a CSV document,
/// for the canvas-grid reader. Re-parsed on demand from the copied file — the
/// same parser the indexer used, so rows/columns line up with chunk (row, col).
#[tauri::command]
async fn get_csv_data(
    catalog: State<'_, SharedCatalog>,
    sha512: String,
) -> Result<extract::CsvGrid, String> {
    let file = catalog
        .lock()
        .await
        .get_file(&sha512)
        .await?
        .ok_or_else(|| format!("unknown file {sha512}"))?;
    tauri::async_runtime::spawn_blocking(move || extract::read_csv_grid(&file.copied_path))
        .await
        .map_err(|e| format!("csv task panicked: {e}"))?
}

/// Compute highlight rectangles for a chunk's span on one PDF page, from PDFium's
/// per-character boxes. `page` is 0-based; `page_char_start`/`length` index the
/// page's text (the same indexing chunk offsets were recorded in).
#[tauri::command]
async fn get_highlight_rects(
    catalog: State<'_, SharedCatalog>,
    pdfium: State<'_, SharedPdfium>,
    sha512: String,
    page: usize,
    page_char_start: usize,
    length: usize,
) -> Result<PageHighlight, String> {
    let file = catalog
        .lock()
        .await
        .get_file(&sha512)
        .await?
        .ok_or_else(|| format!("unknown file {sha512}"))?;
    let pdfium = Arc::clone(pdfium.inner());
    tauri::async_runtime::spawn_blocking(move || -> Result<PageHighlight, String> {
        let pc = pdf::page_chars(&pdfium, &file.copied_path, page)?;
        let end = (page_char_start + length).min(pc.chars.len());
        let start = page_char_start.min(end);
        let rects = merge_char_boxes(&pc.chars[start..end]);
        Ok(PageHighlight {
            page_width: pc.width,
            page_height: pc.height,
            rects,
        })
    })
    .await
    .map_err(|e| format!("highlight task panicked: {e}"))?
}

/// Merge a contiguous run of character boxes into one rectangle per text line.
/// Boxes with no geometry (spaces/newlines) are skipped; a new rectangle starts
/// when a box does not vertically overlap the current line.
fn merge_char_boxes(boxes: &[pdf::CharBox]) -> Vec<[f32; 4]> {
    let mut rects: Vec<[f32; 4]> = Vec::new();
    let mut cur: Option<[f32; 4]> = None; // [left, bottom, right, top]
    for b in boxes {
        if b.right <= b.left || b.top <= b.bottom {
            continue; // no geometry (whitespace/control)
        }
        match cur {
            None => cur = Some([b.left, b.bottom, b.right, b.top]),
            Some(mut r) => {
                let overlaps = b.bottom < r[3] && r[1] < b.top; // vertical ranges overlap
                if overlaps {
                    r[0] = r[0].min(b.left);
                    r[1] = r[1].min(b.bottom);
                    r[2] = r[2].max(b.right);
                    r[3] = r[3].max(b.top);
                    cur = Some(r);
                } else {
                    rects.push(r);
                    cur = Some([b.left, b.bottom, b.right, b.top]);
                }
            }
        }
    }
    if let Some(r) = cur {
        rects.push(r);
    }
    rects
}

/// Build a unit-norm weighted-centroid query vector from weighted text terms
/// (embedded with the query prefix) plus already-embedded weighted vectors
/// (relevance-feedback marks). Returns `None` when there is nothing to use.
async fn build_centroid(
    embedder: &EmbedService,
    texts: Vec<(String, f32)>,
    vectors: Vec<(Vec<f32>, f32)>,
) -> Result<Option<Vec<f32>>, String> {
    if texts.is_empty() && vectors.is_empty() {
        return Ok(None);
    }
    let prompts: Vec<String> = texts.iter().map(|(t, _)| format!("{QUERY_PREFIX}{t}")).collect();
    let emb = embedder
        .run(Lane::Priority, move |m| m.embed_texts(&prompts))
        .await
        .map_err(|e| e.to_string())?;
    let weighted: Vec<(f32, &[f32])> = texts
        .iter()
        .zip(emb.rows.iter())
        .map(|((_, w), row)| (*w, row.as_slice()))
        .chain(vectors.iter().map(|(v, w)| (*w, v.as_slice())))
        .collect();
    let dim = weighted.first().map(|(_, r)| r.len()).unwrap_or(0);
    if dim == 0 {
        return Ok(None);
    }
    let mut centroid = vec![0.0f32; dim];
    for (w, row) in weighted {
        for (c, x) in centroid.iter_mut().zip(row.iter()) {
            *c += w * x;
        }
    }
    let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return Ok(None); // weights cancelled out
    }
    for c in &mut centroid {
        *c /= norm;
    }
    Ok(Some(centroid))
}

/// A file attached to a search query: an image or audio file by path, or an
/// in-app voice recording as base64 bytes (with its container extension).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueryAttachment {
    /// "image" or "audio".
    kind: String,
    path: Option<String>,
    data_base64: Option<String>,
    ext: Option<String>,
}

/// Decode a query's attachments (off the async runtime) and embed them with
/// the text as one interleaved query vector.
async fn embed_mixed_query(
    embedder: &EmbedService,
    text: String,
    attachments: Vec<QueryAttachment>,
) -> Result<Vec<f32>, String> {
    use base64::Engine;
    use semantra_embed::media::{av, image};
    let (images, clips) = tauri::async_runtime::spawn_blocking(move || -> Result<_, String> {
        let mut images = Vec::new();
        let mut clips: Vec<Vec<f32>> = Vec::new();
        for a in attachments {
            // A recording arrives as bytes; AVFoundation decodes from a file.
            let tmp = match &a.data_base64 {
                Some(b64) => {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .map_err(|e| format!("bad attachment data: {e}"))?;
                    let ext = a.ext.as_deref().unwrap_or("m4a").trim_start_matches('.');
                    let mut f = tempfile::Builder::new()
                        .suffix(&format!(".{ext}"))
                        .tempfile()
                        .map_err(|e| format!("temp file: {e}"))?;
                    std::io::Write::write_all(&mut f, &bytes).map_err(|e| format!("temp file: {e}"))?;
                    Some(f)
                }
                None => None,
            };
            let path = match (&tmp, &a.path) {
                (Some(f), _) => f.path().to_path_buf(),
                (None, Some(p)) => PathBuf::from(p),
                (None, None) => return Err("attachment has no path or data".into()),
            };
            match a.kind.as_str() {
                "image" => {
                    let rgb = image::decode(&path).map_err(|e| e.to_string())?;
                    images.push(image::prepare(&rgb, image::IMAGE_SOFT_TOKENS).map_err(|e| e.to_string())?);
                }
                "audio" => {
                    let mut pcm = Vec::new();
                    av::stream_audio(&path, |c| {
                        if pcm.len() < semantra_embed::media::mel::MAX_SAMPLES {
                            pcm.extend_from_slice(c);
                        }
                        Ok(())
                    })
                    .map_err(|e| e.to_string())?;
                    clips.push(pcm);
                }
                other => return Err(format!("unsupported attachment kind {other:?}")),
            }
        }
        Ok((images, clips))
    })
    .await
    .map_err(|e| format!("attachment task panicked: {e}"))??;
    embedder
        .run(Lane::Priority, move |m| {
            let refs: Vec<&[f32]> = clips.iter().map(|c| c.as_slice()).collect();
            m.embed_query_mixed(&text, &images, &refs)
        })
        .await
        .map_err(|e| e.to_string())
}

/// A JPEG `data:` URL preview of an arbitrary local image (e.g. one attached
/// to a query, which lives outside the app-data asset scope).
#[tauri::command]
async fn thumbnail_for_path(path: String, max_side: u32) -> Result<String, String> {
    use base64::Engine;
    use semantra_embed::media::image;
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let rgb = image::decode_max(std::path::Path::new(&path), max_side.clamp(32, 1024)).map_err(|e| e.to_string())?;
        let jpeg = image::encode_jpeg(&rgb, 0.8).map_err(|e| e.to_string())?;
        Ok(format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(jpeg)))
    })
    .await
    .map_err(|e| format!("thumbnail task panicked: {e}"))?
}

/// The in-project search: parse the query into weighted semantic terms + quoted
/// keyword literals, fold in relevance-feedback `preferences`, and return ranked
/// hits restricted to the project's files.
#[tauri::command]
async fn search_project(
    embedder: State<'_, EmbedService>,
    store: State<'_, SharedStore>,
    catalog: State<'_, SharedCatalog>,
    project_id: String,
    query: String,
    preferences: Vec<query::Preference>,
    limit: usize,
    mode: String,
    modalities: Option<Vec<String>>,
    attachments: Option<Vec<QueryAttachment>>,
) -> Result<Vec<ProjectHit>, String> {
    let attachments = attachments.unwrap_or_default();
    let mode = SearchMode::parse(&mode)?;
    // Result-type filter (empty/absent = every modality).
    let modalities: Vec<Modality> = modalities.unwrap_or_default().iter().map(|m| Modality::parse(m)).collect();
    let files = catalog.lock().await.list_project_files(&project_id).await?;
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let shas: Vec<String> = files.iter().map(|f| f.sha512.clone()).collect();
    let meta: std::collections::HashMap<String, (String, String)> = files
        .iter()
        .map(|f| (f.sha512.clone(), (f.basename.clone(), f.filetype.clone())))
        .collect();

    // Parse + normalize weights (semantic terms and preferences share the split).
    let parsed = query::parse_query(&query);
    let mut semantic = parsed.semantic;
    // With attached media, the positive text terms and the media embed together
    // as ONE interleaved query (the model composes them natively, e.g. a photo
    // + "but at night"); it takes a single positive slot in the weighting,
    // alongside any negative terms and feedback marks.
    let mut mixed_text: Option<String> = None;
    if !attachments.is_empty() {
        let positive: Vec<String> = semantic.iter().filter(|t| t.weight > 0.0).map(|t| t.text.clone()).collect();
        mixed_text = Some(positive.join(" "));
        semantic.retain(|t| t.weight < 0.0);
        semantic.insert(0, query::WeightedTerm { text: String::new(), weight: 1.0 });
    }
    let mut prefs = preferences;
    query::normalize_weights(&mut semantic, &mut prefs);
    let literals = parsed.literals;

    // Weighted-centroid input = semantic terms + preferences. A preference on a
    // stored chunk uses that chunk's vector; one without (legacy) its text.
    let ids: Vec<i64> = prefs.iter().filter_map(|p| p.id).collect();
    let stored = store.lock().await.vectors_by_id(&ids).await?;
    let mut vectors: Vec<(Vec<f32>, f32)> = Vec::new();
    if let Some(text) = mixed_text {
        let weight = semantic.remove(0).weight;
        vectors.push((embed_mixed_query(embedder.inner(), text, attachments).await?, weight));
    }
    let mut texts: Vec<(String, f32)> = semantic.iter().map(|t| (t.text.clone(), t.weight)).collect();
    for p in &prefs {
        match p.id.and_then(|id| stored.get(&id)) {
            Some(v) => vectors.push((v.clone(), p.weight)),
            None if !p.text.is_empty() => texts.push((p.text.clone(), p.weight)),
            None => {}
        }
    }
    let centroid = build_centroid(embedder.inner(), texts, vectors).await?;

    // Candidate set from quoted literals: chunks where each literal matches as a
    // word *prefix* (so `"green"` matches "greenhouse"). The store's substring
    // scan returns a coarse superset (any occurrence of the text); we narrow it
    // here — the "post-logic in the handler" — to word-prefix matches via
    // `query::text_matches_literal_prefix`. Literals are ANDed (intersection).
    const FTS_FETCH: usize = 4096;
    let mut candidate: Option<std::collections::HashSet<i64>> = None;
    // Keep the matched hits (id → row) so the literals-only branch can rank the
    // prefix-only candidates the BM25 word index can't see, without re-querying.
    let mut candidate_hits: std::collections::HashMap<i64, store::Hit> =
        std::collections::HashMap::new();
    for lit in &literals {
        // Tokens below the trigram floor can't be matched by the index; a literal
        // with none imposes no constraint (no-op), matching the highlight side.
        let tokens = query::literal_prefix_tokens(lit);
        if tokens.is_empty() {
            continue;
        }
        let hits = store.lock().await.substring_search(&tokens, FTS_FETCH, &shas).await?;
        let ids: std::collections::HashSet<i64> = hits
            .iter()
            .filter(|h| query::text_matches_literal_prefix(&h.text, lit))
            .map(|h| h.id)
            .collect();
        for h in hits {
            if ids.contains(&h.id) {
                candidate_hits.entry(h.id).or_insert(h);
            }
        }
        candidate = Some(match candidate {
            None => ids,
            Some(prev) => prev.intersection(&ids).copied().collect(),
        });
        if candidate.as_ref().is_some_and(|c| c.is_empty()) {
            return Ok(Vec::new()); // a keyword matched nothing → no results
        }
    }

    let hits = match (&centroid, candidate) {
        // Semantic (optionally keyword-filtered).
        (Some(vec), cand) => {
            let fetch = if cand.is_some() { (limit * 20).max(200) } else { limit };
            let mut hits = store.lock().await.search(vec, fetch, mode, &shas, &modalities).await?;
            if let Some(cand) = cand {
                hits.retain(|h| cand.contains(&h.id));
            }
            hits.truncate(limit);
            hits
        }
        // Literals only: rank the prefix-confirmed candidates by their
        // trigram-overlap score (already gathered by `substring_search` into
        // `candidate_hits`) — no separate ranking query needed.
        (None, Some(cand)) => {
            let mut out: Vec<store::Hit> = cand
                .iter()
                .filter_map(|id| candidate_hits.get(id).cloned())
                .filter(|h| modalities.is_empty() || modalities.contains(&h.modality))
                .collect();
            out.sort_by(|a, b| b.score.total_cmp(&a.score));
            out.truncate(limit);
            out
        }
        // Nothing to search.
        (None, None) => Vec::new(),
    };

    Ok(hits
        .into_iter()
        .map(|h| {
            let (basename, filetype) = meta
                .get(&h.sha512)
                .cloned()
                .unwrap_or_else(|| (h.sha512.clone(), "text".to_string()));
            ProjectHit {
                index: h.id,
                sha512: h.sha512,
                basename,
                filetype,
                modality: h.modality,
                text: h.text,
                distance: h.distance,
                score: h.score,
                char_start: h.char_start,
                char_end: h.char_end,
                page: h.page,
                page_char_start: h.page_char_start,
                time_start_ms: h.time_start_ms,
                time_end_ms: h.time_end_ms,
            }
        })
        .collect())
}

/// Everything the app needs before the window can work: the model (on its
/// inference thread), PDFium, the LanceDB store + catalog, and the indexing
/// worker.
fn setup_app(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // MLX must find its compiled Metal kernels before any model work. A
    // bundled app ships them as a resource; dev builds use the compiled-in
    // path (src-tauri/mlx/, which exists on the dev machine).
    if let Some(metallib) = metallib_resource(app) {
        semantra_embed::set_metallib_path(&metallib).map_err(|e| e.to_string())?;
    }
    // The ONNX backend (Windows, Linux, Intel Macs) loads the bundled ONNX
    // Runtime library at runtime. (Both calls are no-ops on the other backend.)
    if let Some(ort) = resolve_onnxruntime(app) {
        semantra_embed::set_onnxruntime_path(&ort).map_err(|e| e.to_string())?;
    }

    // Load the model on its inference thread (which then warms up the
    // Metal kernels in the background before taking any work).
    let model_dir = resolve_model_dir(app)?;
    let embedder = EmbedService::spawn(model_dir, EMBED_DIM).map_err(|e| e.to_string())?;
    let embedding_dim = embedder.embedding_dim() as i32;
    app.manage(embedder);

    // Bind the bundled PDFium library once and share the single instance.
    let pdfium_dir = resolve_pdfium_dir(app)?;
    let pdfium = pdf::load_library(&pdfium_dir)?;
    app.manage(SharedPdfium::new(pdfium));

    // Persistent app-data layout: lancedb/ (vectors + metadata) and files/
    // (copied originals, keyed by SHA-512).
    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("resolve app data dir: {e}"))?;
    let lancedb_dir = app_data.join("lancedb");
    let files_dir = app_data.join("files");
    std::fs::create_dir_all(&lancedb_dir).map_err(|e| format!("create lancedb dir: {e}"))?;
    std::fs::create_dir_all(&files_dir).map_err(|e| format!("create files dir: {e}"))?;
    app.manage(AppPaths { files_dir });

    // One LanceDB connection, shared (cloned) by the store and catalog.
    let (store, catalog, text_stale) = tauri::async_runtime::block_on(async {
        let uri = lancedb_dir
            .to_str()
            .ok_or_else(|| "lancedb path is not valid UTF-8".to_string())?;
        let conn = lancedb::connect(uri)
            .execute()
            .await
            .map_err(|e| format!("connect lancedb: {e}"))?;
        let catalog = Catalog::open(conn.clone()).await?;

        // If the active embedding model (or chunker) changed since this DB
        // was last written, every stored vector is from a different model —
        // and likely a different dimension — so it can't be queried as-is.
        // Re-enqueue every file for indexing and drop the stale vectors;
        // querying any project then requires the new model's re-index to
        // finish. `pipeline_version()` is derived from MODEL_NAME, so this
        // triggers automatically the first time the app runs after a switch.
        let active = pipeline_version();
        let stored = catalog.get_meta(DB_MODEL_KEY).await?;
        // Before the text/media split, the stored version also carried the
        // chunk geometry (`<base>:tokenwindow:…`): the same base means the
        // vectors' model is unchanged and only text needs re-chunking.
        let same_base = stored.as_deref().is_some_and(|v| {
            v == active || v.strip_prefix(active.as_str()).is_some_and(|rest| rest.starts_with(":tokenwindow:"))
        });
        if !same_base {
            let n = catalog.requeue_all_for_reindex(now_ms()).await?;
            let retried = catalog.retry_failed_jobs().await?;
            if retried > 0 {
                eprintln!("[semantra] retrying {retried} previously failed file(s)");
            }
            store::drop_chunks(&conn).await?;
            catalog.set_meta(DB_MODEL_KEY, &active).await?;
            catalog.set_meta(DB_TEXT_KEY, &text_pipeline_version()).await?;
            if stored.is_some() {
                eprintln!(
                    "[semantra] embedding pipeline changed to {active}; \
                     re-indexing {n} file reference(s)"
                );
            }
        } else if stored.as_deref() != Some(active.as_str()) {
            catalog.set_meta(DB_MODEL_KEY, &active).await?;
        }
        let text_stale = catalog.get_meta(DB_TEXT_KEY).await?.as_deref() != Some(text_pipeline_version().as_str());

        let store = VectorStore::open(conn, embedding_dim).await?;
        Ok::<_, String>((store, catalog, text_stale))
    })?;
    app.manage(SharedStore::new(store));
    app.manage(SharedCatalog::new(catalog));

    // Indexing-worker plumbing: a notifier to wake it on new work and a
    // slot holding the file it is currently embedding.
    let notify: SharedNotify = Arc::new(Notify::new());
    app.manage(Arc::clone(&notify));
    app.manage(SharedActive::new(Mutex::new(None)));

    // Spawn the background indexing worker. It drains any jobs that
    // survived a prior shutdown (crash-resume) and then any newly
    // enqueued ones; the initial `notify_one` kicks off that first drain.
    let worker_app = app.handle().clone();
    tauri::async_runtime::spawn(async move { index_worker(worker_app).await });
    notify.notify_one();

    // Text chunking changed (but not the model): re-chunk text in the
    // background, keeping every media vector.
    if text_stale {
        let reindex_app = app.handle().clone();
        tauri::async_runtime::spawn(async move {
            if let Err(e) = reindex_text(&reindex_app).await {
                eprintln!("[semantra] text re-index failed (will retry next launch): {e}");
            }
        });
    }
    Ok(())
}

/// Re-chunk and re-embed the text of every committed file whose text rows are
/// from an older [`text_pipeline_version`], leaving media rows untouched; then
/// record the new version. Resumable and crash-safe: a file is redone if any
/// of its text rows are stale, new rows are inserted before the stale ones are
/// deleted, and leftovers from an interrupted attempt are cleared first. While
/// a file is mid-pass, search may briefly see both its old and new chunks.
async fn reindex_text(app: &AppHandle) -> Result<(), String> {
    let active = text_pipeline_version();
    let files = app.state::<SharedCatalog>().lock().await.list_files().await?;
    let embedder = (*app.state::<EmbedService>()).clone();
    let (mut redone, mut failed) = (0usize, 0usize);
    for f in &files {
        match reindex_file_text(app, &embedder, &active, f).await {
            Ok(false) => {}
            Ok(true) => {
                redone += 1;
                eprintln!("[semantra] re-chunked text of {} ({redone} so far)", f.basename);
            }
            Err(e) => {
                failed += 1;
                eprintln!("[semantra] re-chunking {} failed: {e}", f.basename);
            }
        }
    }
    maybe_maintain_store(app, 1).await;
    if failed > 0 {
        // Leave the version unrecorded so the failures are retried next launch.
        return Err(format!("{failed} file(s) failed; {redone} re-chunked"));
    }
    app.state::<SharedCatalog>().lock().await.set_meta(DB_TEXT_KEY, &active).await?;
    if redone > 0 {
        eprintln!("[semantra] text re-index complete: {redone} file(s) re-chunked");
    }
    Ok(())
}

/// [`reindex_text`] for one file; `Ok(true)` if it was re-chunked, `Ok(false)`
/// if its text was already current (or it has none).
async fn reindex_file_text(
    app: &AppHandle,
    embedder: &EmbedService,
    active: &str,
    f: &FileRecord,
) -> Result<bool, String> {
    let store = app.state::<SharedStore>();
    let versions = store.lock().await.text_versions(&f.sha512).await?;
    if versions.iter().all(|v| v == active) {
        return Ok(false); // already current, or no text at all (images, media)
    }
    let sha = &f.sha512;
    let text_rows = format!("sha512 = '{sha}' AND modality = 'text'");
    // Partial new rows from an interrupted attempt.
    store.lock().await.delete_where(&format!("{text_rows} AND pipeline_version = '{active}'")).await?;

    let pdfium = (*app.state::<SharedPdfium>()).clone();
    let copied = f.copied_path.clone();
    let extracted = tauri::async_runtime::spawn_blocking(move || extract::extract(&pdfium, &copied))
        .await
        .map_err(|e| format!("extract task panicked: {e}"))??;
    let for_chunking = embedder.clone();
    let filetype = extracted.filetype;
    let chunks = tauri::async_runtime::spawn_blocking(move || chunk_segments(&for_chunking, filetype, &extracted.segments))
        .await
        .map_err(|e| format!("chunk task panicked: {e}"))?;
    run_index_pipeline(embedder, &store, chunks, sha.clone(), &|_| {}).await?;

    // Swap: drop the stale rows — or everything, if the file was deleted
    // while we were embedding.
    let gone = app.state::<SharedCatalog>().lock().await.get_file(sha).await?.is_none();
    let stale = if gone { text_rows } else { format!("{text_rows} AND pipeline_version != '{active}'") };
    store.lock().await.delete_where(&stale).await?;
    Ok(true)
}

/// The bundled `mlx.metallib` (Resources/mlx/), if this is a bundled app.
fn metallib_resource(app: &tauri::App) -> Option<PathBuf> {
    let p = app
        .path()
        .resolve("mlx/mlx.metallib", tauri::path::BaseDirectory::Resource)
        .ok()?;
    p.exists().then_some(p)
}

/// Where startup-error.log goes: ~/Library/Logs/Semantra (macOS),
/// %LOCALAPPDATA%\Semantra\logs (Windows), $XDG_STATE_HOME/semantra or
/// ~/.local/state/semantra (Linux).
fn log_dir() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
    if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Logs/Semantra"))
    } else if cfg!(target_os = "windows") {
        env("LOCALAPPDATA").map(|d| d.join("Semantra").join("logs"))
    } else {
        env("XDG_STATE_HOME").or_else(|| env("HOME").map(|h| h.join(".local/state"))).map(|d| d.join("semantra"))
    }
}

/// Startup failed: write the error to startup-error.log in [`log_dir`] and
/// show it in a native alert (the window isn't usable yet).
fn report_startup_failure(message: &str) {
    eprintln!("[semantra] startup failed: {message}");
    let log_path = log_dir().map(|d| d.join("startup-error.log"));
    if let Some(dir) = log_dir() {
        if std::fs::create_dir_all(&dir).is_ok() {
            let _ = std::fs::write(
                dir.join("startup-error.log"),
                format!("Semantra {} failed to start:\n{message}\n", env!("CARGO_PKG_VERSION")),
            );
        }
    }
    let _ = rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("Semantra couldn't start")
        .set_description(match &log_path {
            Some(p) => format!("{message}\n\nDetails were saved to {}.", p.display()),
            None => message.to_string(),
        })
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| match setup_app(app) {
            Ok(()) => Ok(()),
            // A failed startup used to surface as a panic -> abort (release
            // builds use panic = "abort"): a silent crash report. Explain it
            // instead, keep a log for bug reports, and exit cleanly.
            Err(e) => {
                report_startup_failure(&e.to_string());
                std::process::exit(1);
            }
        })
        .invoke_handler(tauri::generate_handler![
            list_projects,
            create_project,
            rename_project,
            delete_project,
            add_files_to_project,
            delete_file_from_project,
            retry_file,
            project_status,
            explain_matches,
            list_documents,
            get_document_text,
            get_pdf_src,
            get_csv_data,
            get_thumbnail,
            thumbnail_for_path,
            get_waveform,
            get_highlight_rects,
            search_project
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
