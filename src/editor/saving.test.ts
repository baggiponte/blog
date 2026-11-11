import assert from "node:assert/strict";
import { test } from "node:test";

import type { Version } from "./post.ts";
import { opened, saveStep, shouldReload } from "./saving.ts";

const v1 = "v1" as Version;
const v2 = "v2" as Version;
const disk = opened({ markdown: "hello", version: v1 });

test("a save writes new text, and only reports back when the text is what's on disk", () => {
  assert.equal(saveStep(disk, "hello world"), "write");
  // Typed and undone: nothing to write, but the status must say "saved".
  assert.equal(saveStep(disk, "hello"), "unchanged");
});

test("after a conflict nothing is saved until the post is opened again", () => {
  const conflicted = { ...disk, conflicted: true };
  assert.equal(saveStep(conflicted, "hello world"), "blocked");
  assert.equal(saveStep(opened({ markdown: "theirs", version: v2 }), "theirs!"), "write");
});

test("coming back to the tab reloads a changed file only when nothing is unsaved", () => {
  assert.equal(shouldReload(disk, "hello", v2), true);
  assert.equal(shouldReload(disk, "hello", v1), false);
  assert.equal(shouldReload(disk, "hello, unsaved", v2), false);
});
