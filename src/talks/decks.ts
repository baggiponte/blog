// Each directory in decks/ is a Slidev deck, served at /talks/<directory>/.
import { readdir } from "node:fs/promises";
import { resolve } from "node:path";

// Directories only, so a stray file such as macOS's .DS_Store is not taken for a deck.
export async function listDecks(root: string): Promise<string[]> {
  const entries = await readdir(resolve(root, "decks"), { withFileTypes: true });
  return entries
    .filter(entry => entry.isDirectory())
    .map(entry => entry.name)
    .sort();
}

export const deckEntry = (root: string, slug: string): string => resolve(root, "decks", slug, "slides.md");
