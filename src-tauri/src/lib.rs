// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod catalog;
pub mod chunk;
mod embed;
pub mod extract;
mod explain;
pub mod pdf;
pub mod query;
pub mod store;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use catalog::{Catalog, FileRecord, Job, JOB_PENDING};
use chunk::{CellChunker, Chunk, Chunker, WordWindowChunker};
use embed::{embed_chunks_with_vectors, EmbedProgress, BATCH_SIZE};
use leaf_ir_candle_test::{
    embed_sentences, explain_similarity_batch, setup_model, warmup, ModelCtx, QUERY_PREFIX,
};
use pdfium_render::prelude::Pdfium;
use sha2::{Digest, Sha512};
use store::{ChunkRow, SearchMode, VectorStore};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::Notify;

// The embedding model is loaded once at startup and kept in Tauri managed
// state. `ModelCtx` holds candle tensors + the tokenizer and is not `Sync`, so
// we guard it with a `Mutex`. The `Arc` lets a handle be cloned into the
// blocking inference task without borrowing the (non-'static) `State`.
type SharedModel = Arc<Mutex<ModelCtx>>;

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

/// Words per chunk for background indexing, and the rewind/overlap shared
/// between consecutive windows. Both are baked into [`pipeline_version`], so
/// changing either auto-invalidates on-disk vectors and triggers re-indexing.
const DEFAULT_CHUNK_SIZE: usize = 50;
const DEFAULT_CHUNK_OVERLAP: usize = 10;

/// Which embedding model to load. The model files live under
/// `models/<MODEL_NAME>/` (bundled as a resource and in the dev tree). This is
/// the switch for the active model: flipping it loads the new weights and
/// changes [`pipeline_version`] (so the DB detects the change and re-indexes on
/// next startup). The store sizes itself to the model's embedding dim.
///
/// Note: `tauri.conf.json` only bundles the *active* model dir
/// (`models/mdbr-leaf-mt/**`) to keep the app bundle small — the other model
/// dirs stay in the dev tree but are not shipped. So switching to a different
/// model also requires updating the `resources` glob in `tauri.conf.json`.
///
/// Known good values: `"mdbr-leaf-ir"` (768-d retrieval), `"mdbr-leaf-mt"`
/// (1024-d multi-task).
pub const MODEL_NAME: &str = "mdbr-leaf-mt";

/// Identifies the chunking + embedding approach that produced a file's chunks,
/// stored per file/chunk so a change can be detected and stale data re-indexed.
/// Derived from [`MODEL_NAME`] and the chunk geometry, so switching models OR
/// changing the chunk size/overlap invalidates old vectors automatically. Bump
/// the `v1` schema rev by hand only for changes not already captured here.
pub fn pipeline_version() -> String {
    format!("v1:{MODEL_NAME}:wordwindow:{DEFAULT_CHUNK_SIZE}-{DEFAULT_CHUNK_OVERLAP}")
}

/// `db_meta` key under which the [`pipeline_version`] the on-disk vectors were
/// built with is stored, so a model/chunker change is detected at startup.
const DB_MODEL_KEY: &str = "active_pipeline";

/// Resolve the directory that holds the bundled model files (resource dir, then
/// dev-tree fallback — `tauri dev` does not reliably copy resources).
fn resolve_model_dir(app: &tauri::App) -> Result<PathBuf, String> {
    let rel = format!("models/{MODEL_NAME}");
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

/// Embed `chunks` (with their metadata) and insert them into the store, pipelining
/// blocking inference into async inserts over a bounded channel. Progress is
/// delivered through a plain sink (`Channel::send` in a command, a Tauri event
/// emit in the background worker), so the pipeline stays transport-agnostic.
async fn run_index_pipeline(
    model: SharedModel,
    store: &SharedStore,
    chunks: Vec<Chunk>,
    sha512: String,
    on_progress: Arc<dyn Fn(EmbedProgress) + Send + Sync>,
) -> Result<(), String> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<ChunkRow>>(4);
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let chunks = Arc::new(chunks);
    let chunks_for_producer = Arc::clone(&chunks);

    let producer = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        embed_chunks_with_vectors(
            // Lock the shared model per batch (not once for the whole file), so a
            // concurrent search query can embed between our batches instead of
            // blocking until the entire document finishes indexing.
            |inputs: &[String]| -> Result<Vec<Vec<f32>>, String> {
                let ctx = model.lock().map_err(|_| "model mutex poisoned".to_string())?;
                Ok(embed_sentences(&ctx, inputs).map_err(|e| e.to_string())?.rows)
            },
            texts,
            false, // documents, not queries
            BATCH_SIZE,
            |start, _batch_texts, vectors| {
                let mut rows = Vec::with_capacity(vectors.len());
                for (j, vector) in vectors.into_iter().enumerate() {
                    let c = &chunks_for_producer[start + j];
                    rows.push(ChunkRow {
                        sha512: sha512.clone(),
                        text: c.text.clone(),
                        char_start: c.char_start as i64,
                        char_end: c.char_end as i64,
                        page: c.page.map(|p| p as i64),
                        page_char_start: c.page_char_start as i64,
                        pipeline_version: pipeline_version(),
                        vector,
                    });
                }
                tx.blocking_send(rows).map_err(|_| "index inserter stopped".to_string())
            },
            |p| {
                on_progress(p);
            },
        )
    });

    // Lock the store only for each insert (not across the embedding waits in
    // between), so background indexing doesn't block concurrent searches/reads
    // for the whole file — the project stays usable while it indexes.
    let mut buf: Vec<ChunkRow> = Vec::new();
    while let Some(rows) = rx.recv().await {
        buf.extend(rows);
        if buf.len() >= INSERT_BATCH {
            store.lock().await.insert(&buf).await?;
            buf.clear();
        }
    }
    if !buf.is_empty() {
        store.lock().await.insert(&buf).await?;
    }
    // Index (re)builds and compaction are intentionally NOT done here: each one
    // rebuilds over the whole table, so running them per file makes a multi-file
    // import O(files²). The worker refreshes them off this hot path — once when
    // the queue drains and occasionally mid-run (see `index_worker`).

    producer
        .await
        .map_err(|e| format!("embedding task panicked: {e}"))??;
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
        // Read + hash + copy into the content-addressed store on a blocking thread.
        // The copy target name is deterministic (`<sha>[.ext]`), so re-copying
        // identical bytes is harmless.
        let (sha512, basename, ext, copied_path) =
            tauri::async_runtime::spawn_blocking(move || -> Result<(String, String, String, String), String> {
                let bytes = std::fs::read(&src).map_err(|e| format!("read {src}: {e}"))?;
                let sha512 = hex(&Sha512::digest(&bytes));
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
                std::fs::write(&copied, &bytes).map_err(|e| format!("copy into app dir: {e}"))?;
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

/// Build the progress sink handed to [`run_index_pipeline`]: it updates the
/// shared active snapshot and emits a `progress` event per batch.
fn progress_sink(
    app: AppHandle,
    active: SharedActive,
    job: Job,
    queue_remaining: usize,
) -> Arc<dyn Fn(EmbedProgress) + Send + Sync> {
    Arc::new(move |p: EmbedProgress| {
        let (done, total) = match &p {
            EmbedProgress::Started { total_chunks, .. } => (0, *total_chunks),
            EmbedProgress::Batch { chunks_done, total_chunks, .. } => (*chunks_done, *total_chunks),
            EmbedProgress::Finished { total_chunks, .. } => (*total_chunks, *total_chunks),
        };
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

    // CSVs index one chunk per cell (the segments are already cells); everything
    // else uses fixed word windows.
    let chunks = match filetype {
        extract::FileType::Csv => CellChunker.chunk(&extracted.segments),
        _ => WordWindowChunker::new(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            .chunk(&extracted.segments),
    };

    let model = (*app.state::<SharedModel>()).clone();
    let sink = progress_sink((*app).clone(), (*active).clone(), job.clone(), queue_remaining);
    run_index_pipeline(model, &store, chunks, job.sha512.clone(), sink).await?;

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

/// The app-data `files/` dir (worker helper).
fn app_paths_dir(app: &AppHandle) -> PathBuf {
    app.state::<AppPaths>().files_dir.clone()
}

/// Explain why each chunk in `texts` matched `query` (per-token attribution).
#[tauri::command]
async fn explain_matches(
    model: State<'_, SharedModel>,
    query: String,
    texts: Vec<String>,
) -> Result<Vec<explain::Explanation>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let model = Arc::clone(model.inner());
    let prefixed = format!("{QUERY_PREFIX}{query}");
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<explain::Explanation>, String> {
        let ctx = model.lock().map_err(|_| "model mutex poisoned".to_string())?;
        let qvec = embed_sentences(&ctx, &[prefixed])
            .map_err(|e| e.to_string())?
            .rows
            .into_iter()
            .next()
            .ok_or_else(|| "query produced no embedding".to_string())?;
        let core = explain_similarity_batch(&ctx, &qvec, &texts).map_err(|e| e.to_string())?;
        Ok(core.into_iter().map(Into::into).collect())
    })
    .await
    .map_err(|e| format!("explain task panicked: {e}"))?
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
    text: String,
    distance: f32,
    score: f32,
    char_start: i64,
    char_end: i64,
    page: Option<i64>,
    page_char_start: i64,
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

/// Build a unit-norm weighted-centroid query vector from semantic terms +
/// preferences (all embedded with the query prefix). Returns `None` when there
/// is nothing to embed.
async fn build_centroid(
    model: SharedModel,
    inputs: Vec<(String, f32)>,
) -> Result<Option<Vec<f32>>, String> {
    if inputs.is_empty() {
        return Ok(None);
    }
    tauri::async_runtime::spawn_blocking(move || -> Result<Option<Vec<f32>>, String> {
        let ctx = model.lock().map_err(|_| "model mutex poisoned".to_string())?;
        let texts: Vec<String> = inputs
            .iter()
            .map(|(t, _)| format!("{QUERY_PREFIX}{t}"))
            .collect();
        let emb = embed_sentences(&ctx, &texts).map_err(|e| e.to_string())?;
        let dim = emb.rows.first().map(|r| r.len()).unwrap_or(0);
        if dim == 0 {
            return Ok(None);
        }
        let mut centroid = vec![0.0f32; dim];
        for ((_, w), row) in inputs.iter().zip(emb.rows.iter()) {
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
    })
    .await
    .map_err(|e| format!("centroid task panicked: {e}"))?
}

/// The in-project search: parse the query into weighted semantic terms + quoted
/// keyword literals, fold in relevance-feedback `preferences`, and return ranked
/// hits restricted to the project's files.
#[tauri::command]
async fn search_project(
    model: State<'_, SharedModel>,
    store: State<'_, SharedStore>,
    catalog: State<'_, SharedCatalog>,
    project_id: String,
    query: String,
    preferences: Vec<query::Preference>,
    limit: usize,
    mode: String,
) -> Result<Vec<ProjectHit>, String> {
    let mode = SearchMode::parse(&mode)?;
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
    let mut prefs = preferences;
    query::normalize_weights(&mut semantic, &mut prefs);
    let literals = parsed.literals;

    // Weighted-centroid embedding input = semantic terms + preference texts.
    let mut inputs: Vec<(String, f32)> = semantic.iter().map(|t| (t.text.clone(), t.weight)).collect();
    inputs.extend(prefs.iter().map(|p| (p.text.clone(), p.weight)));
    let centroid = build_centroid(Arc::clone(model.inner()), inputs).await?;

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
            let mut hits = store.lock().await.search(vec, fetch, mode, &shas).await?;
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
                text: h.text,
                distance: h.distance,
                score: h.score,
                char_start: h.char_start,
                char_end: h.char_end,
                page: h.page,
                page_char_start: h.page_char_start,
            }
        })
        .collect())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let model_dir = resolve_model_dir(app)?;
            let ctx = setup_model(&model_dir).map_err(|e| e.to_string())?;
            let embedding_dim = ctx.embedding_dim() as i32;
            let shared: SharedModel = Arc::new(Mutex::new(ctx));
            app.manage(Arc::clone(&shared));

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
            let (store, catalog) = tauri::async_runtime::block_on(async {
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
                if stored.as_deref() != Some(active.as_str()) {
                    let n = catalog.requeue_all_for_reindex(now_ms()).await?;
                    store::drop_chunks(&conn).await?;
                    catalog.set_meta(DB_MODEL_KEY, &active).await?;
                    if stored.is_some() {
                        eprintln!(
                            "[semantra] embedding pipeline changed to {active}; \
                             re-indexing {n} file reference(s)"
                        );
                    }
                }

                let store = VectorStore::open(conn, embedding_dim).await?;
                Ok::<_, String>((store, catalog))
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

            // Warm up embedding kernels in the background (Metal pipeline compile).
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
        .invoke_handler(tauri::generate_handler![
            list_projects,
            create_project,
            rename_project,
            delete_project,
            add_files_to_project,
            delete_file_from_project,
            project_status,
            explain_matches,
            list_documents,
            get_document_text,
            get_pdf_src,
            get_csv_data,
            get_highlight_rects,
            search_project
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
