import assert from "node:assert/strict";
import { test } from "node:test";

import { type Marker, toggleMarker } from "./format.ts";

// `doc` with the selection marked by [ and ], toggled, then marked the same way.
function press(marked: string, marker: Marker): string {
  const from = marked.indexOf("[");
  const to = marked.indexOf("]") - 1;
  const doc = marked.replace("[", "").replace("]", "");
  const toggled = toggleMarker(doc, from, to, marker);
  let result = doc;
  for (const change of toggled.changes.toReversed()) {
    result = result.slice(0, change.from) + change.insert + result.slice(change.to);
  }
  return `${result.slice(0, toggled.from)}[${result.slice(toggled.from, toggled.to)}]${result.slice(toggled.to)}`;
}

test("wraps the selection and keeps it on the text", () => {
  assert.equal(press("a [word] here", "**"), "a **[word]** here");
  assert.equal(press("a [word] here", "_"), "a _[word]_ here");
  assert.equal(press("a [word] here", "=="), "a ==[word]== here");
  assert.equal(press("a [word] here", "~~"), "a ~~[word]~~ here");
});

test("leaves the spaces at the selection's ends outside the markers", () => {
  assert.equal(press("a [word ]here", "**"), "a **[word]** here");
});

test("takes the markers off, outside or inside the selection", () => {
  assert.equal(press("a **[word]** here", "**"), "a [word] here");
  assert.equal(press("a [**word**] here", "**"), "a [word] here");
  assert.equal(press("a [==word==] here", "=="), "a [word] here");
});

test("italic and bold nest the way oxfmt writes them, and come off either way round", () => {
  assert.equal(press("a **[word]** here", "_"), "a **_[word]_** here");
  assert.equal(press("a **_[word]_** here", "_"), "a **[word]** here");
  assert.equal(press("a **_[word]_** here", "**"), "a _[word]_ here");
  assert.equal(press("a _**[word]**_ here", "_"), "a **[word]** here");
  // Older posts write bold italic as ***word***.
  assert.equal(press("a ***[word]*** here", "**"), "a *[word]* here");
});

test("an empty selection gets an empty pair to type into, and loses it again", () => {
  assert.equal(press("a []here", "**"), "a **[]**here");
  assert.equal(press("a **[]**here", "**"), "a []here");
});
