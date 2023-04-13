export type Offset = [number, number];

export interface File {
  basename: string;
  filename: string;
  filetype: "text" | "pdf";
}

export type SearchResultSet = [string, SearchResult[]][];

export interface SearchResult {
  distance: number;
  text: string;
  offset: [number, number];
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
}
