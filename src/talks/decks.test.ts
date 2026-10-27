import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { listDecks } from "./decks.ts";

test("only directories are decks, in name order", async () => {
  const root = await mkdtemp(join(tmpdir(), "decks-"));
  for (const slug of ["pymi-polars", "agilelab-2024-polars"]) {
    await mkdir(join(root, "decks", slug), { recursive: true });
  }
  // macOS drops these into every folder Finder opens.
  await writeFile(join(root, "decks", ".DS_Store"), "");
  assert.deepEqual(await listDecks(root), ["agilelab-2024-polars", "pymi-polars"]);
});
