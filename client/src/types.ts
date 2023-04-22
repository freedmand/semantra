export type Offset = [number, number];

export interface File {
  basename: string;
  filename: string;
  filetype: "text" | "pdf";
}

export interface ParsedQuery {
  query: string;
  weight: number;
}

export type SearchResultSet = {
  results: [string, SearchResult[]][];
  sort: "asc" | "desc";
};
export type ScoredSearchResult = [string, SearchResult[], number];

export interface SearchResult {
  distance: number;
  text: string;
  offset: [number, number];
  index: number;
  filename: string;
  queries: ParsedQuery[];
  preferences: Preference[];
}

export interface Preference {
  file: File;
  searchResult: SearchResult;
  weight: -1 | 0 | 1;
}

export function preferenceKey(file: File, searchResult: SearchResult): string {
  return JSON.stringify([file.filename, searchResult.index]);
}

export interface Navigation {
  file: File;
  searchResult: SearchResult;
}

export interface PdfPosition {
  char_index: number;
  page_width: number;
  page_height: number;
}

export type PdfChar = [string, PdfCharInfo];
export interface PdfCharInfo {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  lpad?: number;
  tpad?: number;
  rpad?: number;
  bpad?: number;
}

export interface Highlight {
  text: string;
  type: "highlight" | "normal";
}
