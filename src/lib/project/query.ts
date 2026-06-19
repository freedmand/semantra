/**
 * Client-side query parsing — for display only (chips, "search outdated" state).
 * The backend (`src-tauri/src/query.rs`) is authoritative for the actual search;
 * this mirrors its grammar so the UI can preview how a query splits into weighted
 * semantic terms and quoted keyword literals.
 */

export interface WeightedTerm {
  text: string;
  weight: number;
}

export interface ParsedQuery {
  semantic: WeightedTerm[];
  literals: string[];
}

/** Pull `"quoted"` spans out as literals; returns the remainder + literals. */
function extractLiterals(raw: string): { remainder: string; literals: string[] } {
  let remainder = "";
  const literals: string[] = [];
  let current = "";
  let inQuote = false;
  for (const ch of raw) {
    if (ch === '"') {
      if (inQuote) {
        const lit = current.trim();
        if (lit) literals.push(lit);
        current = "";
        inQuote = false;
      } else {
        inQuote = true;
      }
    } else if (inQuote) {
      current += ch;
    } else {
      remainder += ch;
    }
  }
  if (inQuote) {
    const lit = current.trim();
    if (lit) literals.push(lit);
  }
  return { remainder, literals };
}

/** Split the non-quoted remainder into weighted semantic terms (mirrors Rust). */
function parseWeightedTerms(text: string): WeightedTerm[] {
  const regex = /([+\-]?\d*\.?\d*\s*)?([^+\-]+)/g;
  const out: WeightedTerm[] = [];
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    const prefix = match[1] ?? "";
    const term = (match[2] ?? "").trim();
    if (!term) continue;
    const parsed = parseFloat(prefix);
    const weight = !Number.isNaN(parsed)
      ? prefix.includes("-")
        ? -Math.abs(parsed)
        : parsed
      : prefix.includes("-")
        ? -1
        : 1;
    out.push({ text: term, weight });
  }
  return out;
}

export function parseQuery(raw: string): ParsedQuery {
  const { remainder, literals } = extractLiterals(raw);
  return { semantic: parseWeightedTerms(remainder), literals };
}
