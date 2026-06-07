/**
 * Split free text into fixed-size word chunks.
 *
 * Words are runs of non-whitespace; chunks are `size` words joined by single
 * spaces. The final chunk holds the remainder. Pure and dependency-free so it
 * can be reused by the upcoming file-upload feature (and unit-tested).
 *
 * @param text  the raw document text
 * @param size  words per chunk (defaults to 128)
 * @returns     the chunks, in document order (empty array for empty input)
 */
export function chunkWords(text: string, size = 128): string[] {
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += size) {
    chunks.push(words.slice(i, i + size).join(" "));
  }
  return chunks;
}
