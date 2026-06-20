/**
 * Central application state — one reactive source of truth for every subview.
 *
 * A single {@link AppState} class (exported as the {@link appState} singleton) holds
 * per-page sub-views (`manage`, `search`, `sidebar`) as nested `$state` objects, exposes
 * derived "subview" getters (the row merge, result grouping, indexing rollups), and is
 * driven by the standalone loader/mutation functions below. A single
 * {@link initLiveUpdates} subscription fans `index-progress` events out to whichever
 * view is active — replacing the per-component subscriptions that used to live in
 * `ProjectSidebar`, `ManagePanel`, and the search page.
 *
 * Imperative library/DOM handles (PDF.js viewer, ProseMirror EditorView, `bind:this`
 * component refs, the navigation retry loop) are deliberately NOT stored here — only
 * reactive values are. The search page registers its navigation glue via
 * `search.requestNavigate` so deep children can trigger it without prop threading.
 */
import type { Explanation } from "$lib/embedding/search";
import { explainMatches } from "$lib/embedding/search";
import { parseQuery } from "$lib/project/query";
import {
  listProjects,
  listDocuments,
  projectStatus,
  searchProject,
  addFilesToProject,
  deleteFileFromProject,
  renameProject as renameProjectCmd,
  deleteProject as deleteProjectCmd,
  createProject as createProjectCmd,
  onIndexProgress,
  type ProjectListItem,
  type DocMeta,
  type Filetype,
  type ProjectStatus,
  type ProjectHit,
  type Preference,
  type IndexEvent,
} from "$lib/project/projectClient";

export const SEARCH_LIMIT = 30;

/** A relevance-feedback mark in the search view, keyed by chunk id. */
export interface PreferenceMark {
  hit: ProjectHit;
  weight: number;
}

/** A merged manage-view file row: committed documents + queued/failed/in-flight jobs. */
export interface ManageRow {
  sha512: string;
  basename: string;
  state: "indexed" | "indexing" | "queued" | "error";
  done?: number;
  total?: number;
  error?: string;
  /** Stats, present on committed (indexed) rows. */
  filetype?: Filetype;
  pageCount?: number | null;
  wordCount?: number;
  byteLen?: number;
}

/** Manage page ("/project?id="): the file list + indexing + inline-rename state. */
interface ManageView {
  projectId: string;
  docs: DocMeta[];
  status: ProjectStatus;
  loading: boolean;
  adding: boolean;
  error: string;
  renaming: boolean;
  renameText: string;
}

/** Search workspace ("/search?id="): query/results/feedback, active doc, view prefs. */
interface SearchView {
  projectId: string;
  docs: DocMeta[];
  activeIndex: number;
  status: ProjectStatus;
  query: string;
  results: ProjectHit[];
  /** Current result cap; grows by SEARCH_LIMIT each "load more". */
  limit: number;
  explanations: Record<number, Explanation | null>;
  /** Quoted keyword literals from the active query, for extra in-result highlighting. */
  literals: string[];
  preferences: Record<number, PreferenceMark>;
  unsearched: boolean;
  /** A hit awaiting navigation once its viewer mounts (drained by `requestNavigate`). */
  pendingNav: ProjectHit | null;
  /** Monotonic id guarding async explanation results against stale searches. */
  runId: number;
  /** Imperative navigation glue registered by the search page (see module doc). */
  requestNavigate: ((hit: ProjectHit) => void) | null;
  // Results-sidebar view preferences.
  filenameFilter: string;
  filterViewed: boolean;
  excerptView: boolean;
  detailReverse: boolean;
  closed: Record<string, boolean>;
  // PDF reader toolbar state (the PDF.js handles themselves stay in PdfViewer).
  pdf: {
    currentPage: number;
    totalPages: number;
    scale: number;
    loading: boolean;
    error: string | null;
  };
}

/** Sidebar create/rename input buffers + transient error. */
interface SidebarView {
  error: string;
  creatingOpen: boolean;
  newName: string;
  creating: boolean;
  renamingId: string | null;
  renameText: string;
}

function emptyManage(): ManageView {
  return {
    projectId: "",
    docs: [],
    status: { jobs: [], active: null },
    loading: true,
    adding: false,
    error: "",
    renaming: false,
    renameText: "",
  };
}

function emptySearch(): SearchView {
  return {
    projectId: "",
    docs: [],
    activeIndex: 0,
    status: { jobs: [], active: null },
    query: "",
    results: [],
    limit: SEARCH_LIMIT,
    explanations: {},
    literals: [],
    preferences: {},
    unsearched: true,
    pendingNav: null,
    runId: 0,
    requestNavigate: null,
    filenameFilter: "",
    filterViewed: false,
    excerptView: false,
    detailReverse: false,
    closed: {},
    pdf: { currentPage: 1, totalPages: 0, scale: 1, loading: true, error: null },
  };
}

/** A file-grouped bundle of search hits (results sidebar, file view). */
export interface SearchGroup {
  sha512: string;
  basename: string;
  hits: ProjectHit[];
  score: number;
}

class AppState {
  // ---- Shared project list (absorbs the former projectsStore) ----
  projects = $state<ProjectListItem[]>([]);
  projectsLoaded = $state(false);

  // ---- Per-page sub-views ----
  manage = $state<ManageView>(emptyManage());
  search = $state<SearchView>(emptySearch());
  sidebar = $state<SidebarView>({
    error: "",
    creatingOpen: false,
    newName: "",
    creating: false,
    renamingId: null,
    renameText: "",
  });

  // ---- Manage subviews ----
  manageProjectName = $derived(
    this.projects.find((p) => p.projectId === this.manage.projectId)?.name ?? "Project",
  );

  manageRows = $derived.by<ManageRow[]>(() => {
    const { docs, status } = this.manage;
    const out: ManageRow[] = docs.map((d) => ({
      sha512: d.sha512,
      basename: d.basename,
      state: "indexed",
      filetype: d.filetype,
      pageCount: d.pageCount,
      wordCount: d.wordCount,
      byteLen: d.byteLen,
    }));
    const activeSha = status.active?.sha512;
    for (const j of status.jobs) {
      if (j.status === "error") {
        out.push({ sha512: j.sha512, basename: j.basename, state: "error", error: j.error });
      } else if (j.sha512 === activeSha) {
        out.push({
          sha512: j.sha512,
          basename: j.basename,
          state: "indexing",
          done: status.active!.done,
          total: status.active!.total,
        });
      } else {
        out.push({ sha512: j.sha512, basename: j.basename, state: "queued" });
      }
    }
    return out;
  });

  managePending = $derived(
    this.manage.status.jobs.filter((j) => j.status === "pending").length,
  );
  manageIndexing = $derived(this.manage.status.active !== null || this.managePending > 0);
  manageCanSearch = $derived(this.manage.docs.length > 0 || this.manageIndexing);

  // ---- Search subviews ----
  searchProjectName = $derived(
    this.projects.find((p) => p.projectId === this.search.projectId)?.name ?? "Project",
  );
  searchActiveDoc = $derived(this.search.docs[this.search.activeIndex] ?? null);
  searchPending = $derived(
    this.search.status.jobs.filter((j) => j.status === "pending").length,
  );
  searchIndexing = $derived(this.search.status.active !== null || this.searchPending > 0);

  // Group hits by file, score each group by its mean hit score, sort high → low.
  searchGroups = $derived.by<SearchGroup[]>(() => {
    const activeSha = this.searchActiveDoc?.sha512 ?? null;
    const byFile = new Map<string, SearchGroup>();
    for (const hit of this.search.results) {
      let g = byFile.get(hit.sha512);
      if (!g) {
        g = { sha512: hit.sha512, basename: hit.basename, hits: [], score: 0 };
        byFile.set(hit.sha512, g);
      }
      g.hits.push(hit);
    }
    const out = [...byFile.values()];
    for (const g of out) {
      g.hits.sort((a, b) => b.score - a.score);
      g.score = g.hits.reduce((s, h) => s + h.score, 0) / g.hits.length;
    }
    out.sort((a, b) => b.score - a.score);
    return out.filter((g) =>
      this.search.filterViewed && activeSha ? g.sha512 === activeSha : true,
    );
  });

  searchExcerpt = $derived.by<ProjectHit[]>(() => {
    const activeSha = this.searchActiveDoc?.sha512 ?? null;
    return [...this.search.results]
      .sort((a, b) => b.score - a.score)
      .filter((h) =>
        this.search.filterViewed && activeSha ? h.sha512 === activeSha : true,
      );
  });

  /** Active (non-zero) preference marks, for the search bar chips. */
  prefList = $derived(Object.values(this.search.preferences).filter((p) => p.weight !== 0));

  /** Whether the last search likely has more results to fetch (hit the cap). */
  searchHasMore = $derived(
    !this.search.unsearched && this.search.results.length >= this.search.limit,
  );
}

export const appState = new AppState();

// === Pure label helpers =====================================================

/** A similarity score (0–1) as a rounded percent, e.g. "56%". */
export function scorePercent(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/** Short status line for a manage-view file row. */
export function manageRowLabel(row: ManageRow): string {
  switch (row.state) {
    case "indexed":
      // No label once indexed — the word-count stats line signals it's ready to search.
      return "";
    case "indexing":
      return row.total && row.total > 0
        ? `Indexing… ${(row.done ?? 0).toLocaleString()}/${row.total.toLocaleString()}`
        : "Indexing…";
    case "queued":
      return "Queued";
    case "error":
      return `Failed: ${row.error}`;
  }
}

/** Human-readable size, e.g. "1.2 MB". */
function formatBytes(n: number): string {
  if (n < 1024) return `${n.toLocaleString()} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString()} ${units[i]}`;
}

/**
 * Stats for a committed file row (word count, pages for PDFs, size), or null
 * for rows that haven't finished indexing yet.
 */
export function manageRowStats(row: ManageRow): string | null {
  if (row.state !== "indexed" || row.wordCount == null) return null;
  const parts = [
    `${row.wordCount.toLocaleString()} ${row.wordCount === 1 ? "word" : "words"}`,
  ];
  if (row.filetype === "pdf" && row.pageCount) {
    parts.push(`${row.pageCount.toLocaleString()} ${row.pageCount === 1 ? "page" : "pages"}`);
  }
  if (row.byteLen) parts.push(formatBytes(row.byteLen));
  return parts.join(" · ");
}

/** Short status line for a sidebar project row. */
export function projectStatusLabel(p: ProjectListItem): string {
  const parts: string[] = [
    `${p.docCount.toLocaleString()} ${p.docCount === 1 ? "document" : "documents"}`,
  ];
  if (p.pendingCount > 0) parts.push(`${p.pendingCount.toLocaleString()} indexing`);
  if (p.errorCount > 0) parts.push(`${p.errorCount.toLocaleString()} failed`);
  return parts.join(" · ");
}

// === Shared project-list operations =========================================

/** Reload the project list and publish it to every consumer. */
export async function refreshProjects(): Promise<ProjectListItem[]> {
  appState.projects = await listProjects();
  appState.projectsLoaded = true;
  return appState.projects;
}

export async function createProject(id: string, name: string): Promise<void> {
  await createProjectCmd(id, name);
  await refreshProjects();
}

export async function renameProject(projectId: string, name: string): Promise<void> {
  await renameProjectCmd(projectId, name);
  await refreshProjects();
}

export async function deleteProject(projectId: string): Promise<void> {
  await deleteProjectCmd(projectId);
  await refreshProjects();
}

// === Manage view =============================================================

/** Reload the manage view's docs + status (live updates; no project-list refetch). */
async function refreshManage(): Promise<void> {
  const projectId = appState.manage.projectId;
  if (!projectId) return;
  try {
    const [docs, status] = await Promise.all([
      listDocuments(projectId),
      projectStatus(projectId),
    ]);
    appState.manage.docs = docs;
    appState.manage.status = status;
    appState.manage.error = "";
  } catch (e) {
    appState.manage.error = `${e}`;
  }
}

/** Enter the manage view for a project: reset state, then load docs/status/projects. */
export async function loadManage(projectId: string): Promise<void> {
  // Reset in place (not reassign) so component-held aliases stay valid.
  Object.assign(appState.manage, emptyManage(), { projectId });
  await Promise.all([refreshManage(), refreshProjects()]);
  appState.manage.loading = false;
}

export async function addFilesToManage(paths: string[]): Promise<void> {
  if (appState.manage.adding) return;
  appState.manage.adding = true;
  try {
    await addFilesToProject(appState.manage.projectId, paths);
    await refreshManage();
    await refreshProjects();
  } catch (e) {
    appState.manage.error = `${e}`;
  } finally {
    appState.manage.adding = false;
  }
}

export async function removeManageFile(sha512: string): Promise<void> {
  await deleteFileFromProject(appState.manage.projectId, sha512);
  await refreshManage();
  await refreshProjects();
}

/** Commit the manage-title inline rename (no-op if unchanged/empty). */
export async function commitManageRename(): Promise<void> {
  const name = appState.manage.renameText.trim();
  appState.manage.renaming = false;
  if (!name || name === appState.manageProjectName) return;
  await renameProject(appState.manage.projectId, name);
}

// === Search view =============================================================

/** Enter the search workspace for a project: reset state, load docs/status/projects. */
export async function loadSearch(projectId: string): Promise<void> {
  // Reset in place (not reassign) so aliases stay valid; preserve the page's
  // registered navigation glue across the reset.
  const requestNavigate = appState.search.requestNavigate;
  Object.assign(appState.search, emptySearch(), { projectId, requestNavigate });
  const [docs, status] = await Promise.all([
    listDocuments(projectId),
    projectStatus(projectId),
    refreshProjects(),
  ]);
  appState.search.docs = docs;
  appState.search.status = status;
}

/** Reload the search view's docs + status (live updates as files finish indexing). */
async function refreshSearchIndexing(): Promise<void> {
  const projectId = appState.search.projectId;
  if (!projectId) return;
  const [docs, status] = await Promise.all([
    listDocuments(projectId),
    projectStatus(projectId),
  ]);
  appState.search.docs = docs;
  appState.search.status = status;
}

/** Run the in-project search at the current query/limit, then attach explanations. */
async function executeSearch(): Promise<void> {
  const s = appState.search;
  const prefs: Preference[] = Object.values(s.preferences)
    .filter((p) => p.weight !== 0)
    .map((p) => ({ text: p.hit.text, weight: p.weight }));
  if (s.query.trim() === "" && prefs.length === 0) {
    s.results = [];
    s.unsearched = true;
    return;
  }
  const myRun = ++s.runId;
  const parsed = parseQuery(s.query);
  s.literals = parsed.literals;
  s.results = await searchProject(s.projectId, s.query, prefs, s.limit);
  s.unsearched = false;
  s.explanations = {};

  // Best-effort highlighting: explain against the semantic part of the query.
  const semantic = parsed.semantic.map((t) => t.text).join(" ").trim() || s.query;
  const current = s.results;
  explainMatches(semantic, current.map((r) => r.text))
    .then((exps) => {
      if (myRun !== s.runId) return;
      const map: Record<number, Explanation | null> = {};
      current.forEach((r, i) => (map[r.index] = exps[i] ?? null));
      s.explanations = map;
    })
    .catch((e) => console.warn("explain failed", e));
}

/** Run a fresh search for `q` (resets the result cap). */
export async function runSearch(q: string): Promise<void> {
  appState.search.query = q;
  appState.search.limit = SEARCH_LIMIT;
  await executeSearch();
}

/** Fetch the next page of results for the current query (grows the cap). */
export async function loadMoreResults(): Promise<void> {
  appState.search.limit += SEARCH_LIMIT;
  await executeSearch();
}

/** Set (or clear, when `weight === 0`) the relevance-feedback mark for a hit. */
export function setPreference(hit: ProjectHit, weight: number): void {
  const s = appState.search;
  if (weight === 0) {
    const next = { ...s.preferences };
    delete next[hit.index];
    s.preferences = next;
  } else {
    s.preferences = { ...s.preferences, [hit.index]: { hit, weight } };
  }
}

/** Navigate to a hit: switch to its document, then hand off to the page's nav glue. */
export function jumpToResult(hit: ProjectHit): void {
  const idx = appState.search.docs.findIndex((d) => d.sha512 === hit.sha512);
  if (idx < 0) return;
  appState.search.activeIndex = idx;
  appState.search.pendingNav = hit;
  appState.search.requestNavigate?.(hit);
}

// === Live indexing updates (single subscription) ============================

let liveUpdatesStarted = false;

/** Register the single app-wide `index-progress` listener. Idempotent. */
export async function initLiveUpdates(): Promise<void> {
  if (liveUpdatesStarted) return;
  liveUpdatesStarted = true;
  await onIndexProgress(handleIndexEvent);
}

function patchActive(view: ManageView | SearchView, ev: IndexEvent): void {
  view.status = {
    ...view.status,
    active: { sha512: ev.sha512, basename: ev.basename, done: ev.done, total: ev.total },
  };
}

function handleIndexEvent(ev: IndexEvent): void {
  // File set changed → refresh project-list badges for the sidebar.
  if (ev.kind === "started" || ev.kind === "fileDone" || ev.kind === "error") {
    refreshProjects();
  }
  // Route docs/status updates to whichever loaded view owns this project.
  if (ev.projectId === appState.manage.projectId) {
    if (ev.kind === "progress") patchActive(appState.manage, ev);
    else refreshManage();
  }
  if (ev.projectId === appState.search.projectId) {
    if (ev.kind === "progress") patchActive(appState.search, ev);
    else refreshSearchIndexing();
  }
}
