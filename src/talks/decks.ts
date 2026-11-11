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

// The _redirects rewrites that serve a deck's app shell on each of its routes: every slide, in the normal and the
// presenter view, and the presenter, overview and notes pages. Exact routes rather than a catch-all, so a deck never
// swallows another deck's paths or a missing asset.
export function deckRewrites(slug: string, slides: number): string[] {
  const base = `/talks/${slug}/`;
  const routes: string[] = [];
  for (let slide = 1; slide <= slides; slide++) {
    routes.push(String(slide), `${slide}/`, `presenter/${slide}`, `presenter/${slide}/`);
  }
  routes.push("presenter", "presenter/", "overview", "overview/", "notes", "notes/");
  return routes.map(route => `${base}${route} ${base} 200`);
}
