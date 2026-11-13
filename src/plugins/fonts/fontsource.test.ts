import assert from "node:assert/strict";
import { test } from "node:test";

import { facesFor, parseFaces } from "./fontsource.ts";

const css = `/* bricolage-grotesque-vietnamese-wght-normal */
@font-face {
  font-family: 'Bricolage Grotesque Variable';
  font-style: normal;
  font-display: swap;
  font-weight: 200 800;
  src: url(./files/bricolage-grotesque-vietnamese-wght-normal.woff2) format('woff2-variations');
  unicode-range: U+0102-0103,U+0110-0111;
}

/* bricolage-grotesque-latin-ext-wght-normal */
@font-face {
  font-family: 'Bricolage Grotesque Variable';
  font-style: normal;
  font-display: swap;
  font-weight: 200 800;
  src: url(./files/bricolage-grotesque-latin-ext-wght-normal.woff2) format('woff2-variations');
  unicode-range: U+0100-02BA,U+02BD-02C5;
}

/* bricolage-grotesque-latin-wght-normal */
@font-face {
  font-family: 'Bricolage Grotesque Variable';
  font-style: normal;
  font-display: swap;
  font-weight: 200 800;
  src: url(./files/bricolage-grotesque-latin-wght-normal.woff2) format('woff2-variations');
  unicode-range: U+0000-00FF,U+0131;
}`;

test("each face keeps its file, subset, weight range and characters", () => {
  const [, latinExt] = parseFaces(css, "bricolage-grotesque");
  assert.deepEqual(latinExt, {
    file: "files/bricolage-grotesque-latin-ext-wght-normal.woff2",
    subset: "latin-ext",
    weight: "200 800",
    style: "normal",
    unicodeRange: ["U+0100-02BA", "U+02BD-02C5"],
  });
});

test("only the subsets asked for are served", () => {
  const subsets = facesFor(parseFaces(css, "bricolage-grotesque"), ["latin", "latin-ext"]).map(face => face.subset);
  assert.deepEqual(subsets, ["latin-ext", "latin"]);
});

test("a rule that isn't a Fontsource variable face stops the build", () => {
  assert.throws(
    () => parseFaces("@font-face { src: url(./font.ttf); }", "bricolage-grotesque"),
    /not a Fontsource variable font face/,
  );
});
