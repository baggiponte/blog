// Kept apart from decks.ts so the talks page doesn't load Slidev just to list the decks.
import { dirname } from "node:path";
import { parser } from "@slidev/cli";

import { deckEntry } from "./decks.ts";

export async function slideCount(root: string, slug: string): Promise<number> {
  const entry = deckEntry(root, slug);
  const directory = dirname(entry);
  const data = await parser.load({ roots: [directory], userRoot: directory }, entry);
  return data.slides.length;
}
