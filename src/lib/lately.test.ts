import assert from "node:assert/strict";
import { test } from "node:test";

import { parseLately } from "./lately.ts";

test("each top-level bullet is a phrase, in order, and the prose around them is ignored", () => {
  const markdown = [
    "Notes on the list.",
    "",
    "- first one.",
    "* second one.  ",
    "  - nested, not a phrase.",
    "-not a bullet",
    "- ",
    "+ third, with a — dash.",
  ].join("\n");
  assert.deepEqual(parseLately(markdown), ["first one.", "second one.", "third, with a — dash."]);
});

test("a list with no bullets fails the build instead of rendering an empty bio", () => {
  assert.throws(() => parseLately("just prose\n"), /no bullet points/);
});
