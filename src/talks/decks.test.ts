import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { deckRewrites, listDecks } from "./decks.ts";

test("only directories are decks, in name order", async () => {
  const root = await mkdtemp(join(tmpdir(), "decks-"));
  for (const slug of ["pymi-polars", "agilelab-2024-polars"]) {
    await mkdir(join(root, "decks", slug), { recursive: true });
  }
  // macOS drops these into every folder Finder opens.
  await writeFile(join(root, "decks", ".DS_Store"), "");
  assert.deepEqual(await listDecks(root), ["agilelab-2024-polars", "pymi-polars"]);
});

test("every slide and view of a deck is rewritten to its app, and nothing else", () => {
  const rewrites = deckRewrites("pymi-polars", 2);
  assert.ok(rewrites.includes("/talks/pymi-polars/2/ /talks/pymi-polars/ 200"));
  assert.ok(rewrites.includes("/talks/pymi-polars/presenter/1 /talks/pymi-polars/ 200"));
  assert.ok(rewrites.includes("/talks/pymi-polars/overview /talks/pymi-polars/ 200"));
  assert.ok(!rewrites.some(rewrite => rewrite.startsWith("/talks/pymi-polars/3")));
  assert.ok(rewrites.every(rewrite => rewrite.endsWith(" /talks/pymi-polars/ 200")));
});
