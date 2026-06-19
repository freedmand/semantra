//! Project / file metadata catalog, backed by LanceDB.
//!
//! Three tables, all living in the same persistent connection as the chunk
//! [`VectorStore`](crate::store) (LanceDB has no joins, so membership and
//! ref-counting are done with small filtered scans in Rust — fine at our scale):
//!
//! - `projects` — one row per project (the project UI is a later pass; until
//!   then everything goes under a single default project).
//! - `files` — **content-addressed** by SHA-512: one row per distinct file's
//!   bytes, regardless of how many projects reference it. Copied-in path,
//!   detected type, sizes, page count, and the `pipeline_version` that produced
//!   its chunks (so we can detect & re-index when chunking/embedding changes).
//! - `project_files` — membership: which project references which file, and the
//!   **original on-disk source path** for that reference.
//!
//! A file's bytes are embedded once; deleting a project ref GCs the file's row
//! (and, via the caller, its chunks + copied bytes) only when no other project
//! still references it.

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};

const PROJECTS: &str = "projects";
const FILES: &str = "files";
const PROJECT_FILES: &str = "project_files";
const INDEX_JOBS: &str = "index_jobs";

/// Job status values stored in the `index_jobs` table.
pub const JOB_PENDING: &str = "pending";
pub const JOB_ERROR: &str = "error";

/// A distinct file's metadata (keyed by `sha512`).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileRecord {
    pub sha512: String,
    pub basename: String,
    pub ext: String,
    pub copied_path: String,
    pub filetype: String,
    pub byte_len: i64,
    /// Page count for paginated files; `None` for flat text.
    pub page_count: Option<i64>,
    pub char_len: i64,
    pub pipeline_version: String,
    pub created_at: i64,
}

/// A project row, for the listing UI.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectRecord {
    pub project_id: String,
    pub name: String,
    pub created_at: i64,
}

/// A unit of pending (or failed) indexing work: a file's bytes are already copied
/// into the app-data `files/` dir (`copied_path`); what remains is extract →
/// chunk → embed → insert, then commit the file + project reference. A row exists
/// only while that work is outstanding (deleted on success), so on startup the
/// presence of a `pending` row *is* the resume signal. `status` is [`JOB_PENDING`]
/// or [`JOB_ERROR`] (with `error` carrying the message for the latter).
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    pub project_id: String,
    pub sha512: String,
    pub source_path: String,
    pub copied_path: String,
    pub basename: String,
    pub ext: String,
    pub status: String,
    pub error: String,
    pub added_at: i64,
}

/// Catalog tables. Cheap to clone-from internally (LanceDB `Table` is an `Arc`).
pub struct Catalog {
    conn: Connection,
    projects: Table,
    files: Table,
    project_files: Table,
    index_jobs: Table,
}

impl Catalog {
    /// Open (creating if absent) the three metadata tables on `conn`.
    pub async fn open(conn: Connection) -> Result<Self, String> {
        let existing = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| format!("list tables: {e}"))?;

        let projects = ensure_table(&conn, &existing, PROJECTS, projects_schema()).await?;
        let files = ensure_table(&conn, &existing, FILES, files_schema()).await?;
        let project_files =
            ensure_table(&conn, &existing, PROJECT_FILES, project_files_schema()).await?;
        let index_jobs = ensure_table(&conn, &existing, INDEX_JOBS, index_jobs_schema()).await?;

        Ok(Self {
            conn,
            projects,
            files,
            project_files,
            index_jobs,
        })
    }

    /// Create the project row if it does not already exist.
    pub async fn ensure_project(&self, project_id: &str, name: &str, now: i64) -> Result<(), String> {
        if !self.rows_exist(&self.projects, &format!("project_id = '{project_id}'")).await? {
            let batch = string_int_batch(
                projects_schema(),
                &[("project_id", project_id), ("name", name)],
                &[("created_at", now)],
            )?;
            append(&self.projects, batch).await?;
        }
        Ok(())
    }

    /// All projects (for the listing), in creation order.
    pub async fn list_projects(&self) -> Result<Vec<ProjectRecord>, String> {
        let mut out = read_project_records(&self.scan(&self.projects, None).await?);
        out.sort_by_key(|p| p.created_at);
        Ok(out)
    }

    /// Rename a project, preserving its `created_at`. No-op if it does not exist.
    pub async fn set_project_name(&self, project_id: &str, name: &str) -> Result<(), String> {
        let existing = read_project_records(
            &self.scan(&self.projects, Some(format!("project_id = '{project_id}'"))).await?,
        );
        let Some(prev) = existing.into_iter().next() else {
            return Ok(());
        };
        self.projects
            .delete(&format!("project_id = '{project_id}'"))
            .await
            .map_err(|e| format!("delete project for rename: {e}"))?;
        let batch = string_int_batch(
            projects_schema(),
            &[("project_id", project_id), ("name", name)],
            &[("created_at", prev.created_at)],
        )?;
        append(&self.projects, batch).await
    }

    /// Whether a file with these bytes is already cataloged (exact-dedup gate).
    pub async fn get_file(&self, sha512: &str) -> Result<Option<FileRecord>, String> {
        let batches = self
            .scan(&self.files, Some(format!("sha512 = '{sha512}'")))
            .await?;
        Ok(read_file_records(&batches).into_iter().next())
    }

    /// Insert a new file row (caller has already confirmed it is absent).
    pub async fn insert_file(&self, f: &FileRecord) -> Result<(), String> {
        let batch = build_file_batch(f)?;
        append(&self.files, batch).await
    }

    /// Record that `project_id` references the file `sha512`, originally read
    /// from `source_path`. Idempotent per (project, file).
    pub async fn add_file_ref(
        &self,
        project_id: &str,
        sha512: &str,
        source_path: &str,
        now: i64,
    ) -> Result<(), String> {
        let pred = format!("project_id = '{project_id}' AND sha512 = '{sha512}'");
        if self.rows_exist(&self.project_files, &pred).await? {
            return Ok(());
        }
        let batch = string_int_batch(
            project_files_schema(),
            &[
                ("project_id", project_id),
                ("sha512", sha512),
                ("source_path", source_path),
            ],
            &[("added_at", now)],
        )?;
        append(&self.project_files, batch).await
    }

    /// Files referenced by a project, in insertion order.
    pub async fn list_project_files(&self, project_id: &str) -> Result<Vec<FileRecord>, String> {
        let shas = self.project_file_shas(project_id).await?;
        let mut out = Vec::with_capacity(shas.len());
        for sha in shas {
            if let Some(f) = self.get_file(&sha).await? {
                out.push(f);
            }
        }
        Ok(out)
    }

    /// SHA-512s referenced by a project.
    pub async fn project_file_shas(&self, project_id: &str) -> Result<Vec<String>, String> {
        let batches = self
            .scan(&self.project_files, Some(format!("project_id = '{project_id}'")))
            .await?;
        Ok(column_strings(&batches, "sha512"))
    }

    /// How many project references point at this file.
    pub async fn ref_count(&self, sha512: &str) -> Result<usize, String> {
        let batches = self
            .scan(&self.project_files, Some(format!("sha512 = '{sha512}'")))
            .await?;
        Ok(batches.iter().map(|b| b.num_rows()).sum())
    }

    /// Remove a project and its membership rows. Returns the SHA-512s that are
    /// now orphaned (no remaining references) and have had their `files` rows
    /// deleted — the caller GCs their chunks and copied bytes.
    pub async fn delete_project(&self, project_id: &str) -> Result<Vec<String>, String> {
        let shas = self.project_file_shas(project_id).await?;
        self.project_files
            .delete(&format!("project_id = '{project_id}'"))
            .await
            .map_err(|e| format!("delete project_files: {e}"))?;
        self.projects
            .delete(&format!("project_id = '{project_id}'"))
            .await
            .map_err(|e| format!("delete project: {e}"))?;

        // Drop any outstanding indexing jobs for the project too.
        self.index_jobs
            .delete(&format!("project_id = '{project_id}'"))
            .await
            .map_err(|e| format!("delete project jobs: {e}"))?;

        let mut orphaned = Vec::new();
        for sha in shas {
            if self.ref_count(&sha).await? == 0 {
                self.files
                    .delete(&format!("sha512 = '{sha}'"))
                    .await
                    .map_err(|e| format!("delete file row: {e}"))?;
                orphaned.push(sha);
            }
        }
        Ok(orphaned)
    }

    /// Remove one file's reference from a project. Returns `true` (and deletes the
    /// `files` row) if the file is now orphaned — the caller GCs its chunks and
    /// copied bytes. Mirrors [`delete_project`](Self::delete_project) for a single
    /// file.
    pub async fn remove_file_ref(&self, project_id: &str, sha512: &str) -> Result<bool, String> {
        self.project_files
            .delete(&format!("project_id = '{project_id}' AND sha512 = '{sha512}'"))
            .await
            .map_err(|e| format!("delete project_files ref: {e}"))?;
        if self.ref_count(sha512).await? == 0 {
            self.files
                .delete(&format!("sha512 = '{sha512}'"))
                .await
                .map_err(|e| format!("delete file row: {e}"))?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // --- indexing job queue ---------------------------------------------

    /// Enqueue a unit of indexing work. Idempotent per `(project, file)`: if any
    /// job row already exists for the pair it is left as-is (callers retry an
    /// errored job via [`set_job_pending`](Self::set_job_pending)).
    pub async fn enqueue_job(&self, job: &Job) -> Result<(), String> {
        let pred = format!(
            "project_id = '{}' AND sha512 = '{}'",
            job.project_id, job.sha512
        );
        if self.rows_exist(&self.index_jobs, &pred).await? {
            return Ok(());
        }
        append(&self.index_jobs, build_job_batch(job)?).await
    }

    /// The oldest pending job across all projects, or `None` when the queue is
    /// drained. The background worker pulls one at a time from here.
    pub async fn next_pending_job(&self) -> Result<Option<Job>, String> {
        let mut jobs = read_jobs(
            &self.scan(&self.index_jobs, Some(format!("status = '{JOB_PENDING}'"))).await?,
        );
        jobs.sort_by_key(|j| j.added_at);
        Ok(jobs.into_iter().next())
    }

    /// All jobs (pending and errored) for a project, oldest first — for the manage
    /// view's per-file status.
    pub async fn jobs_for(&self, project_id: &str) -> Result<Vec<Job>, String> {
        let mut jobs = read_jobs(
            &self.scan(&self.index_jobs, Some(format!("project_id = '{project_id}'"))).await?,
        );
        jobs.sort_by_key(|j| j.added_at);
        Ok(jobs)
    }

    /// `(pending, error)` job counts for a project — for the listing badges.
    pub async fn job_counts(&self, project_id: &str) -> Result<(usize, usize), String> {
        let jobs = self.jobs_for(project_id).await?;
        let pending = jobs.iter().filter(|j| j.status == JOB_PENDING).count();
        let error = jobs.iter().filter(|j| j.status == JOB_ERROR).count();
        Ok((pending, error))
    }

    /// Whether a specific `(project, file)` job row still exists (used by the
    /// worker to detect a delete-during-index cancellation before committing).
    pub async fn job_exists(&self, project_id: &str, sha512: &str) -> Result<bool, String> {
        self.rows_exist(
            &self.index_jobs,
            &format!("project_id = '{project_id}' AND sha512 = '{sha512}'"),
        )
        .await
    }

    /// Remove a job row (on success, or when its file is deleted from the project).
    pub async fn delete_job(&self, project_id: &str, sha512: &str) -> Result<(), String> {
        self.index_jobs
            .delete(&format!("project_id = '{project_id}' AND sha512 = '{sha512}'"))
            .await
            .map(|_| ())
            .map_err(|e| format!("delete job: {e}"))
    }

    /// Mark a job failed, recording the error message (it stays in the table so
    /// the UI can show it and offer a retry, but is no longer picked up).
    pub async fn mark_job_error(
        &self,
        project_id: &str,
        sha512: &str,
        message: &str,
    ) -> Result<(), String> {
        self.update_job_status(project_id, sha512, JOB_ERROR, message).await
    }

    /// Flip an errored job back to pending (retry).
    pub async fn set_job_pending(&self, project_id: &str, sha512: &str) -> Result<(), String> {
        self.update_job_status(project_id, sha512, JOB_PENDING, "").await
    }

    /// Rewrite a job row's `status`/`error` (delete + re-append, preserving the
    /// rest). No-op if the job no longer exists.
    async fn update_job_status(
        &self,
        project_id: &str,
        sha512: &str,
        status: &str,
        error: &str,
    ) -> Result<(), String> {
        let pred = format!("project_id = '{project_id}' AND sha512 = '{sha512}'");
        let Some(mut job) = read_jobs(&self.scan(&self.index_jobs, Some(pred.clone())).await?)
            .into_iter()
            .next()
        else {
            return Ok(());
        };
        self.index_jobs
            .delete(&pred)
            .await
            .map_err(|e| format!("delete job for status update: {e}"))?;
        job.status = status.to_string();
        job.error = error.to_string();
        append(&self.index_jobs, build_job_batch(&job)?).await
    }

    // --- internals -------------------------------------------------------

    async fn rows_exist(&self, table: &Table, predicate: &str) -> Result<bool, String> {
        let batches = self.scan(table, Some(predicate.to_string())).await?;
        Ok(batches.iter().any(|b| b.num_rows() > 0))
    }

    async fn scan(&self, table: &Table, predicate: Option<String>) -> Result<Vec<RecordBatch>, String> {
        let mut q = table.query();
        if let Some(p) = predicate {
            q = q.only_if(p);
        }
        q.execute()
            .await
            .map_err(|e| format!("query: {e}"))?
            .try_collect()
            .await
            .map_err(|e| format!("collect: {e}"))
    }

    /// Expose the underlying connection so the store can share it.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

// --- schemas ------------------------------------------------------------

fn projects_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("project_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

fn files_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sha512", DataType::Utf8, false),
        Field::new("basename", DataType::Utf8, false),
        Field::new("ext", DataType::Utf8, false),
        Field::new("copied_path", DataType::Utf8, false),
        Field::new("filetype", DataType::Utf8, false),
        Field::new("byte_len", DataType::Int64, false),
        Field::new("page_count", DataType::Int64, true),
        Field::new("char_len", DataType::Int64, false),
        Field::new("pipeline_version", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

fn project_files_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("project_id", DataType::Utf8, false),
        Field::new("sha512", DataType::Utf8, false),
        Field::new("source_path", DataType::Utf8, false),
        Field::new("added_at", DataType::Int64, false),
    ]))
}

fn index_jobs_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("project_id", DataType::Utf8, false),
        Field::new("sha512", DataType::Utf8, false),
        Field::new("source_path", DataType::Utf8, false),
        Field::new("copied_path", DataType::Utf8, false),
        Field::new("basename", DataType::Utf8, false),
        Field::new("ext", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("error", DataType::Utf8, false),
        Field::new("added_at", DataType::Int64, false),
    ]))
}

// --- batch helpers ------------------------------------------------------

async fn ensure_table(
    conn: &Connection,
    existing: &[String],
    name: &str,
    schema: SchemaRef,
) -> Result<Table, String> {
    if existing.iter().any(|n| n == name) {
        conn.open_table(name)
            .execute()
            .await
            .map_err(|e| format!("open {name}: {e}"))
    } else {
        conn.create_empty_table(name, schema)
            .execute()
            .await
            .map_err(|e| format!("create {name}: {e}"))
    }
}

async fn append(table: &Table, batch: RecordBatch) -> Result<(), String> {
    let schema = batch.schema();
    let reader: Box<dyn RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
    table
        .add(reader)
        .execute()
        .await
        .map_err(|e| format!("append: {e}"))?;
    Ok(())
}

/// Build a one-row batch from named string and i64 columns, in the schema's
/// field order. Only for the all-non-null projects/project_files schemas.
fn string_int_batch(
    schema: SchemaRef,
    strings: &[(&str, &str)],
    ints: &[(&str, i64)],
) -> Result<RecordBatch, String> {
    let columns: Vec<Arc<dyn Array>> = schema
        .fields()
        .iter()
        .map(|f| -> Arc<dyn Array> {
            if let Some((_, v)) = strings.iter().find(|(k, _)| k == f.name()) {
                Arc::new(StringArray::from(vec![*v]))
            } else if let Some((_, v)) = ints.iter().find(|(k, _)| k == f.name()) {
                Arc::new(Int64Array::from(vec![*v]))
            } else {
                Arc::new(StringArray::from(vec![""]))
            }
        })
        .collect();
    RecordBatch::try_new(schema, columns).map_err(|e| format!("build batch: {e}"))
}

fn build_file_batch(f: &FileRecord) -> Result<RecordBatch, String> {
    let schema = files_schema();
    let page_count = Int64Array::from(vec![f.page_count]); // None -> null
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(StringArray::from(vec![f.sha512.as_str()])),
        Arc::new(StringArray::from(vec![f.basename.as_str()])),
        Arc::new(StringArray::from(vec![f.ext.as_str()])),
        Arc::new(StringArray::from(vec![f.copied_path.as_str()])),
        Arc::new(StringArray::from(vec![f.filetype.as_str()])),
        Arc::new(Int64Array::from(vec![f.byte_len])),
        Arc::new(page_count),
        Arc::new(Int64Array::from(vec![f.char_len])),
        Arc::new(StringArray::from(vec![f.pipeline_version.as_str()])),
        Arc::new(Int64Array::from(vec![f.created_at])),
    ];
    RecordBatch::try_new(schema, columns).map_err(|e| format!("build file batch: {e}"))
}

fn build_job_batch(j: &Job) -> Result<RecordBatch, String> {
    string_int_batch(
        index_jobs_schema(),
        &[
            ("project_id", j.project_id.as_str()),
            ("sha512", j.sha512.as_str()),
            ("source_path", j.source_path.as_str()),
            ("copied_path", j.copied_path.as_str()),
            ("basename", j.basename.as_str()),
            ("ext", j.ext.as_str()),
            ("status", j.status.as_str()),
            ("error", j.error.as_str()),
        ],
        &[("added_at", j.added_at)],
    )
}

// --- readers ------------------------------------------------------------

fn column_strings(batches: &[RecordBatch], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.column_by_name(name) {
            if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..arr.len() {
                    out.push(arr.value(i).to_string());
                }
            }
        }
    }
    out
}

fn read_project_records(batches: &[RecordBatch]) -> Vec<ProjectRecord> {
    let mut out = Vec::new();
    for b in batches {
        let s = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let n = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let (Some(id), Some(name), Some(created)) = (s("project_id"), s("name"), n("created_at"))
        else {
            continue;
        };
        for i in 0..b.num_rows() {
            out.push(ProjectRecord {
                project_id: id.value(i).to_string(),
                name: name.value(i).to_string(),
                created_at: created.value(i),
            });
        }
    }
    out
}

fn read_jobs(batches: &[RecordBatch]) -> Vec<Job> {
    let mut out = Vec::new();
    for b in batches {
        let s = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let n = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let (
            Some(project_id),
            Some(sha512),
            Some(source_path),
            Some(copied_path),
            Some(basename),
            Some(ext),
            Some(status),
            Some(error),
            Some(added_at),
        ) = (
            s("project_id"),
            s("sha512"),
            s("source_path"),
            s("copied_path"),
            s("basename"),
            s("ext"),
            s("status"),
            s("error"),
            n("added_at"),
        )
        else {
            continue;
        };
        for i in 0..b.num_rows() {
            out.push(Job {
                project_id: project_id.value(i).to_string(),
                sha512: sha512.value(i).to_string(),
                source_path: source_path.value(i).to_string(),
                copied_path: copied_path.value(i).to_string(),
                basename: basename.value(i).to_string(),
                ext: ext.value(i).to_string(),
                status: status.value(i).to_string(),
                error: error.value(i).to_string(),
                added_at: added_at.value(i),
            });
        }
    }
    out
}

fn read_file_records(batches: &[RecordBatch]) -> Vec<FileRecord> {
    let mut out = Vec::new();
    for b in batches {
        let s = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let n = |name: &str| b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let (Some(sha), Some(base), Some(ext), Some(path), Some(ft), Some(pv)) = (
            s("sha512"),
            s("basename"),
            s("ext"),
            s("copied_path"),
            s("filetype"),
            s("pipeline_version"),
        ) else {
            continue;
        };
        let (Some(byte_len), Some(page_count), Some(char_len), Some(created)) =
            (n("byte_len"), n("page_count"), n("char_len"), n("created_at"))
        else {
            continue;
        };
        for i in 0..b.num_rows() {
            out.push(FileRecord {
                sha512: sha.value(i).to_string(),
                basename: base.value(i).to_string(),
                ext: ext.value(i).to_string(),
                copied_path: path.value(i).to_string(),
                filetype: ft.value(i).to_string(),
                byte_len: byte_len.value(i),
                page_count: if page_count.is_null(i) {
                    None
                } else {
                    Some(page_count.value(i))
                },
                char_len: char_len.value(i),
                pipeline_version: pv.value(i).to_string(),
                created_at: created.value(i),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_catalog() -> (Catalog, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let conn = lancedb::connect(tmp.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();
        (Catalog::open(conn).await.unwrap(), tmp)
    }

    fn sample_file(sha: &str) -> FileRecord {
        FileRecord {
            sha512: sha.into(),
            basename: "doc.pdf".into(),
            ext: "pdf".into(),
            copied_path: format!("/files/{sha}.pdf"),
            filetype: "pdf".into(),
            byte_len: 1234,
            page_count: Some(3),
            char_len: 999,
            pipeline_version: "v1".into(),
            created_at: 100,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_project_is_idempotent() {
        let (cat, _t) = temp_catalog().await;
        cat.ensure_project("p1", "Default", 1).await.unwrap();
        cat.ensure_project("p1", "Default", 2).await.unwrap();
        // Only one row despite two calls.
        let batches = cat.scan(&cat.projects, None).await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn file_dedup_and_refcount() {
        let (cat, _t) = temp_catalog().await;
        cat.ensure_project("p1", "P1", 1).await.unwrap();
        cat.ensure_project("p2", "P2", 1).await.unwrap();

        assert!(cat.get_file("abc").await.unwrap().is_none());
        cat.insert_file(&sample_file("abc")).await.unwrap();
        assert_eq!(cat.get_file("abc").await.unwrap().unwrap().byte_len, 1234);

        // Two projects reference the same file.
        cat.add_file_ref("p1", "abc", "/orig/a.pdf", 2).await.unwrap();
        cat.add_file_ref("p2", "abc", "/elsewhere/a.pdf", 3).await.unwrap();
        // Duplicate ref is a no-op.
        cat.add_file_ref("p1", "abc", "/orig/a.pdf", 4).await.unwrap();
        assert_eq!(cat.ref_count("abc").await.unwrap(), 2);

        assert_eq!(cat.list_project_files("p1").await.unwrap().len(), 1);
    }

    fn sample_job(project: &str, sha: &str, at: i64) -> Job {
        Job {
            project_id: project.into(),
            sha512: sha.into(),
            source_path: format!("/orig/{sha}.pdf"),
            copied_path: format!("/files/{sha}.pdf"),
            basename: "doc.pdf".into(),
            ext: "pdf".into(),
            status: JOB_PENDING.into(),
            error: String::new(),
            added_at: at,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_projects_and_rename() {
        let (cat, _t) = temp_catalog().await;
        cat.ensure_project("p1", "First", 10).await.unwrap();
        cat.ensure_project("p2", "Second", 20).await.unwrap();

        let projects = cat.list_projects().await.unwrap();
        assert_eq!(projects.len(), 2);
        assert_eq!(projects[0].project_id, "p1"); // sorted by created_at
        assert_eq!(projects[1].name, "Second");

        cat.set_project_name("p1", "Renamed").await.unwrap();
        let projects = cat.list_projects().await.unwrap();
        let p1 = projects.iter().find(|p| p.project_id == "p1").unwrap();
        assert_eq!(p1.name, "Renamed");
        assert_eq!(p1.created_at, 10); // preserved
        assert_eq!(projects.len(), 2); // still exactly one p1 row
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remove_file_ref_orphans_when_last() {
        let (cat, _t) = temp_catalog().await;
        cat.insert_file(&sample_file("shared")).await.unwrap();
        cat.add_file_ref("p1", "shared", "/a", 1).await.unwrap();
        cat.add_file_ref("p2", "shared", "/b", 1).await.unwrap();

        // p1 drops it but p2 still holds a ref → not orphaned, file survives.
        assert!(!cat.remove_file_ref("p1", "shared").await.unwrap());
        assert!(cat.get_file("shared").await.unwrap().is_some());
        // p2 drops the last ref → orphaned, file row deleted.
        assert!(cat.remove_file_ref("p2", "shared").await.unwrap());
        assert!(cat.get_file("shared").await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_queue_lifecycle() {
        let (cat, _t) = temp_catalog().await;
        cat.enqueue_job(&sample_job("p1", "b", 20)).await.unwrap();
        cat.enqueue_job(&sample_job("p1", "a", 10)).await.unwrap();
        // Duplicate (p1, a) is ignored.
        cat.enqueue_job(&sample_job("p1", "a", 99)).await.unwrap();

        assert_eq!(cat.job_counts("p1").await.unwrap(), (2, 0));
        // Oldest pending first.
        assert_eq!(cat.next_pending_job().await.unwrap().unwrap().sha512, "a");
        assert!(cat.job_exists("p1", "a").await.unwrap());

        // Errored jobs are no longer picked up, but counted + retriable.
        cat.mark_job_error("p1", "a", "boom").await.unwrap();
        assert_eq!(cat.job_counts("p1").await.unwrap(), (1, 1));
        assert_eq!(cat.next_pending_job().await.unwrap().unwrap().sha512, "b");
        let errored = cat
            .jobs_for("p1")
            .await
            .unwrap()
            .into_iter()
            .find(|j| j.sha512 == "a")
            .unwrap();
        assert_eq!(errored.status, JOB_ERROR);
        assert_eq!(errored.error, "boom");

        cat.set_job_pending("p1", "a").await.unwrap();
        assert_eq!(cat.job_counts("p1").await.unwrap(), (2, 0));

        cat.delete_job("p1", "a").await.unwrap();
        cat.delete_job("p1", "b").await.unwrap();
        assert_eq!(cat.job_counts("p1").await.unwrap(), (0, 0));
        assert!(cat.next_pending_job().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_project_clears_its_jobs() {
        let (cat, _t) = temp_catalog().await;
        cat.ensure_project("p1", "P1", 1).await.unwrap();
        cat.enqueue_job(&sample_job("p1", "a", 1)).await.unwrap();
        cat.enqueue_job(&sample_job("p2", "a", 1)).await.unwrap();
        cat.delete_project("p1").await.unwrap();
        assert_eq!(cat.job_counts("p1").await.unwrap(), (0, 0));
        assert_eq!(cat.job_counts("p2").await.unwrap(), (1, 0)); // untouched
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_project_orphans_only_unreferenced_files() {
        let (cat, _t) = temp_catalog().await;
        cat.ensure_project("p1", "P1", 1).await.unwrap();
        cat.ensure_project("p2", "P2", 1).await.unwrap();
        cat.insert_file(&sample_file("shared")).await.unwrap();
        cat.insert_file(&sample_file("solo")).await.unwrap();
        cat.add_file_ref("p1", "shared", "/a", 2).await.unwrap();
        cat.add_file_ref("p2", "shared", "/b", 2).await.unwrap();
        cat.add_file_ref("p1", "solo", "/c", 2).await.unwrap();

        // Deleting p1: "solo" is now unreferenced (orphaned), "shared" survives (p2).
        let orphaned = cat.delete_project("p1").await.unwrap();
        assert_eq!(orphaned, vec!["solo".to_string()]);
        assert!(cat.get_file("solo").await.unwrap().is_none());
        assert!(cat.get_file("shared").await.unwrap().is_some());
        assert_eq!(cat.ref_count("shared").await.unwrap(), 1);
    }
}
