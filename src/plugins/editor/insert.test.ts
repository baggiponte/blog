import assert from "node:assert/strict";
import { test } from "node:test";

import { imageInsert } from "./insert.ts";

// The document after inserting, with | where the cursor lands.
function drop(doc: string, position: number, names: string[]): string {
  const { text, altFrom } = imageInsert(doc, position, names);
  const result = doc.slice(0, position) + text + doc.slice(position);
  const cursor = position + altFrom;
  return `${result.slice(0, cursor)}|${result.slice(cursor)}`;
}

test("an image dropped mid-paragraph gets a paragraph of its own", () => {
  assert.equal(drop("one two", 3, ["a.png"]), "one\n\n![|](./a.png)\n\n two");
});

test("no extra blank lines where there already are some", () => {
  assert.equal(drop("one\n\ntwo", 5, ["a.png"]), "one\n\n![|](./a.png)\n\ntwo");
  assert.equal(drop("one\n", 4, ["a.png"]), "one\n\n![|](./a.png)\n");
});

test("at the start or end of the file, the image isn't padded with blank lines", () => {
  assert.equal(drop("", 0, ["a.png"]), "![|](./a.png)\n");
  assert.equal(drop("one", 3, ["a.png"]), "one\n\n![|](./a.png)\n");
});

test("several images are one paragraph each, and the cursor goes to the first", () => {
  assert.equal(drop("one", 3, ["a.png", "b.jpg"]), "one\n\n![|](./a.png)\n\n![](./b.jpg)\n");
});
