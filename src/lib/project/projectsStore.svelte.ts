/**
 * Shared, reactive project list for the (manage) shell. The sidebar and the
 * manage panel are siblings across the layout/page-slot boundary, so they can't
 * pass props to each other; this module-level rune state lets them stay in sync.
 * Any mutation (create/rename/delete) calls {@link refreshProjects} so both the
 * sidebar row and the manage title update from one source of truth.
 */
import { listProjects, type ProjectListItem } from "./projectClient";

let projects = $state<ProjectListItem[]>([]);
let loaded = $state(false);

/** Reactive accessors — read `.list` / `.loaded` inside `$derived` or markup. */
export const projectsState = {
  get list(): ProjectListItem[] {
    return projects;
  },
  get loaded(): boolean {
    return loaded;
  },
};

/** Reload the project list from the backend and publish it to all consumers. */
export async function refreshProjects(): Promise<ProjectListItem[]> {
  projects = await listProjects();
  loaded = true;
  return projects;
}
