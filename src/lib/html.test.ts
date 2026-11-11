import assert from "node:assert/strict";
import { test } from "node:test";

import { html, join } from "./html.ts";

const escaped = "polars &amp; &lt;script&gt;&quot;friends&quot;&lt;/script&gt;";

test("interpolated text is escaped, so data can't inject markup", () => {
  const title = 'polars & <script>"friends"</script>';
  assert.equal(html`<h2 title="${title}">${title}</h2>`.markup, `<h2 title="${escaped}">${escaped}</h2>`);
});

test("markup built with html`` nests as is, arrays are joined and missing values are left out", () => {
  const items = ["a", "b&c"].map(item => html`<li>${item}</li>`);
  assert.equal(html`<ol>${items}${null}${undefined}${3}</ol>`.markup, "<ol><li>a</li><li>b&amp;c</li>3</ol>");
});

test("join puts the separator between pieces only", () => {
  assert.equal(join([html`<p>a</p>`, html`<p>b</p>`], "\n").markup, "<p>a</p>\n<p>b</p>");
});
