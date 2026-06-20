/**
 * Frontend client for the in-project interface commands (see `src-tauri/src/lib.rs`).
 *
 * These wrap the document/search/highlight Tauri commands that operate on the
 * (currently single, default) project. Result highlighting in the sidebar reuses
 * the embedding module's `explainMatches` + `highlight` helpers, so this file
 * only covers documents, search, full text, and PDF highlight geometry.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

/** A project in the listing, with document + indexing-queue counts. */
export interface ProjectListItem {
  projectId: string;
  name: string;
  createdAt: number;
  /** Files fully indexed and searchable. */
  docCount: number;
  /** Files still queued for indexing. */
  pendingCount: number;
  /** Files whose indexing failed. */
  errorCount: number;
}

/** A queued/failed file's status (manage view). */
export interface JobStatus {
  sha512: string;
  basename: string;
  status: "pending" | "error";
  error: string;
}

/** The file currently being indexed, with live chunk progress. */
export interface ActiveStatus {
  sha512: string;
  basename: string;
  done: number;
  total: number;
}

/** A project's indexing state snapshot. */
export interface ProjectStatus {
  jobs: JobStatus[];
  active: ActiveStatus | null;
}

/**
 * A live indexing-progress event (backend `index://progress`). `kind`:
 * - `started`  — a file began indexing
 * - `progress` — a batch finished (`done`/`total` are chunk counts)
 * - `fileDone` — a file committed; refresh the document list
 * - `error`    — indexing failed (`error` carries the message)
 */
export interface IndexEvent {
  projectId: string;
  sha512: string;
  basename: string;
  kind: "started" | "progress" | "fileDone" | "error";
  done: number;
  total: number;
  /** Pending files left in this project. */
  queueRemaining: number;
  error: string;
}

/** A document's kind, as understood by the reader UI. */
export type Filetype = "text" | "pdf" | "csv";

/** A document in the project (tab bar + sidebar grouping). */
export interface DocMeta {
  sha512: string;
  basename: string;
  filetype: Filetype;
  pageCount: number | null;
  byteLen: number;
  wordCount: number;
}

/** One search hit, enriched for navigation + highlighting. Mirrors Rust `ProjectHit`. */
export interface ProjectHit {
  /** Stable chunk id — used as the preference key. */
  index: number;
  sha512: string;
  basename: string;
  filetype: Filetype;
  text: string;
  distance: number;
  score: number;
  /** Char offsets into the document's canonical full text (flat text reader). */
  charStart: number;
  charEnd: number;
  /**
   * Page (0-based) and page-relative char offset for PDF highlighting. For CSV
   * documents these are repurposed: `page` is the 0-based data-row index and
   * `pageCharStart` is the 0-based column index, so the grid reader navigates to
   * the matched cell by (row, col).
   */
  page: number | null;
  pageCharStart: number;
}

/** A parsed CSV document: header row + data rows, all cells verbatim. */
export interface CsvData {
  headers: string[];
  rows: string[][];
}

/** A relevance-feedback preference sent back to refine the next search. */
export interface Preference {
  /** The marked result's text (re-embedded as a query term). */
  text: string;
  /** +1 relevant, -1 not relevant. */
  weight: number;
}

/** Highlight rectangles for one PDF page, in PDF user-space points. */
export interface PageHighlight {
  pageWidth: number;
  pageHeight: number;
  /** Each rect is `[left, bottom, right, top]` in points (bottom-left origin). */
  rects: [number, number, number, number][];
}

// === Project CRUD ===========================================================

/** List all projects with their document + indexing-queue counts. */
export async function listProjects(): Promise<ProjectListItem[]> {
  return await invoke<ProjectListItem[]>("list_projects");
}

/** Create a project. `projectId` is a client-generated UUID. Idempotent. */
export async function createProject(
  projectId: string,
  name: string,
): Promise<void> {
  await invoke("create_project", { projectId, name });
}

/** Rename a project (preserves its creation time). */
export async function renameProject(
  projectId: string,
  name: string,
): Promise<void> {
  await invoke("rename_project", { projectId, name });
}

/**
 * Queue files for background indexing into a project. Returns immediately;
 * progress arrives via {@link onIndexProgress}. Already-indexed bytes are
 * referenced instantly (content-addressed dedup).
 */
export async function addFilesToProject(
  projectId: string,
  paths: string[],
): Promise<void> {
  await invoke("add_files_to_project", { projectId, paths });
}

/** Remove a file from a project (cancels its job; GCs bytes if orphaned). */
export async function deleteFileFromProject(
  projectId: string,
  sha512: string,
): Promise<void> {
  await invoke("delete_file_from_project", { projectId, sha512 });
}

/** A project's current indexing state (initial snapshot before live events). */
export async function projectStatus(projectId: string): Promise<ProjectStatus> {
  return await invoke<ProjectStatus>("project_status", { projectId });
}

/**
 * Subscribe to live indexing progress for all projects. Filter by
 * `event.projectId` in the callback. Returns an unlisten function.
 */
export async function onIndexProgress(
  cb: (event: IndexEvent) => void,
): Promise<UnlistenFn> {
  return await listen<IndexEvent>("index-progress", (e) => cb(e.payload));
}

/** List the documents in a project (committed files only; for tabs + sidebar). */
export async function listDocuments(projectId: string): Promise<DocMeta[]> {
  return await invoke<DocMeta[]>("list_documents", { projectId });
}

/** Canonical full text of a document (flat text reader + offset math). */
export async function getDocumentText(sha512: string): Promise<string> {
  return await invoke<string>("get_document_text", { sha512 });
}

/** Filesystem path of a document's copied bytes (wrap with `convertFileSrc`). */
export async function getPdfSrc(sha512: string): Promise<string> {
  return await invoke<string>("get_pdf_src", { sha512 });
}

/** Parsed grid (headers + rows, verbatim cells) of a CSV document, for the grid reader. */
export async function getCsvData(sha512: string): Promise<CsvData> {
  return await invoke<CsvData>("get_csv_data", { sha512 });
}

/**
 * Highlight rectangles for a chunk's span on one PDF page, computed from PDFium
 * char boxes. `page` is 0-based; `pageCharStart`/`length` index the page text.
 */
export async function getHighlightRects(
  sha512: string,
  page: number,
  pageCharStart: number,
  length: number,
): Promise<PageHighlight> {
  return await invoke<PageHighlight>("get_highlight_rects", {
    sha512,
    page,
    pageCharStart,
    length,
  });
}

/**
 * Run the in-project search. `query` is the raw query box text (the backend
 * parses weighted semantic terms + quoted keyword literals); `preferences` are
 * the active relevance-feedback marks.
 */
export async function searchProject(
  projectId: string,
  query: string,
  preferences: Preference[],
  limit = 20,
  mode: "exact" | "ann" = "exact",
): Promise<ProjectHit[]> {
  return await invoke<ProjectHit[]>("search_project", {
    projectId,
    query,
    preferences,
    limit,
    mode,
  });
}

/** Delete a project, GC'ing files no other project references. */
export async function deleteProject(projectId: string): Promise<void> {
  await invoke("delete_project", { projectId });
}
