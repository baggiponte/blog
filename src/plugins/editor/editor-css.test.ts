import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { test } from "node:test";

// The dev server serves editor.css on its own, so a relative @import that points nowhere fails the whole stylesheet
// and the editor loads unstyled; moving the editor into src/plugins/ did exactly that.
test("editor.css imports only stylesheets that exist", async () => {
  const sheet = new URL("./editor.css", import.meta.url);
  const css = await readFile(sheet, "utf8");
  const relative = [...css.matchAll(/@import\s+"(\.[^"]+)"/g)].map(([, path]) => path ?? "");
  assert.notEqual(relative.length, 0);
  for (const path of relative) {
    await assert.doesNotReject(access(new URL(path, sheet)), `${path} is missing`);
  }
});
