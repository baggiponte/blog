// The ⌘K search: /search.json is built from SearchEntry values and the palette in scripts/site.ts queries it.

export interface SearchEntry {
  title: string;
  description: string;
  date: string;
  url: string;
  // The lowercase post without code blocks, which would match every query for "import".
  text: string;
}

// Every word must appear in the title or the text, in any order. Entries matching more words in their title come
// first; ties keep the index's order, which is newest first.
export function search(index: readonly SearchEntry[], query: string): SearchEntry[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const inTitle = (entry: SearchEntry): number => words.filter(word => entry.title.toLowerCase().includes(word)).length;
  return index
    .filter(entry => words.every(word => entry.title.toLowerCase().includes(word) || entry.text.includes(word)))
    .toSorted((a, b) => inTitle(b) - inTitle(a));
}
