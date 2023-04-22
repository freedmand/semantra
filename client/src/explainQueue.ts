import { get, writable } from "svelte/store";
import type { Highlight, ParsedQuery, Preference } from "./types";

interface ExplainDictionary {
  [key: string]: Highlight[];
}

export const explainDictionary = writable<ExplainDictionary>({});

const explanationQueue: [HTMLElement, string][] = [];

interface ExplainProps {
  filename: string;
  offset: [number, number];
  queries: ParsedQuery[];
  preferences: Preference[];
}
export function requestExplanation(element: HTMLElement, params: ExplainProps) {
  const key = JSON.stringify(params);
  if (get(explainDictionary)[key]) {
    return;
  }
  if (explanationQueue.find(([, k]) => k === key)) {
    return;
  }
  explanationQueue.push([element, key]);
}

const BOTTOM_BOUNDARY = 0;

let explaining = false;

function explainTopOfQueue() {
  if (explanationQueue.length === 0 || explaining) {
    return;
  }
  const sortedQueue = explanationQueue.sort(([e1], [e2]) => {
    if (e1 == null && e2 == null) {
      return 0;
    }
    if (e1 == null) {
      return 1;
    }
    if (e2 == null) {
      return -1;
    }
    const rect1B = e1.getBoundingClientRect().bottom;
    const rect2B = e2.getBoundingClientRect().bottom;
    if (rect1B < BOTTOM_BOUNDARY && rect2B >= BOTTOM_BOUNDARY) {
      return 1;
    }
    if (rect1B >= BOTTOM_BOUNDARY && rect2B < BOTTOM_BOUNDARY) {
      return -1;
    }
    return rect1B - rect2B;
  });

  while (sortedQueue.length > 0) {
    const [element, key] = sortedQueue.shift()!;
    if (
      element == null ||
      element.getBoundingClientRect().bottom < BOTTOM_BOUNDARY
    ) {
      continue;
    }

    explain(key);
    return;
  }
}

export async function explain(body: string) {
  explaining = true;
  const existingResponse = get(explainDictionary)[body];
  if (existingResponse) {
    return existingResponse;
  }

  const request = await fetch("/api/explain", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: body,
  });
  const highlights = await request.json();

  explainDictionary.update((dict) => {
    dict[body] = highlights;
    return dict;
  });
  explaining = false;
  return highlights;
}

// Continuously explain the top of the queue
setInterval(explainTopOfQueue, 20);
