// The post pipeline as astro.config.ts sets it up: highlights, code captions and the plain-text export.
import assert from "node:assert/strict";
import { test } from "node:test";

import { markdown, plainMarkdown } from "./markdown.ts";

const renderer = await markdown.processor.createRenderer({ shikiConfig: markdown.shikiConfig });
const html = async (source: string): Promise<string> => (await renderer.render(source)).code;

test("==text== becomes an escaped <mark>, but not in code or around spaces", async () => {
  assert.equal(
    await html("keep ==a < b== in mind, `==code==`, and x == y == z"),
    "<p>keep <mark>a &#x3C; b</mark> in mind, <code>==code==</code>, and x == y == z</p>",
  );
});

test("several highlights in one paragraph, and inside a link", async () => {
  assert.equal(
    await html("==one== and ==two== [==three==](/x)"),
    '<p><mark>one</mark> and <mark>two</mark> <a href="/x"><mark>three</mark></a></p>',
  );
});

test("code blocks are captioned with the marimo cell name, else the language", async () => {
  const page = await html('```python {.marimo name="setup"}\nx = 1\n```\n\n```js\n1\n```\n\n```\nplain\n```');
  const captions = [...page.matchAll(/<figcaption><span>([^<]*)<\/span><\/figcaption>/g)].map(match => match[1]);
  assert.deepEqual(captions, ["setup", "js", "text"]);
});

test("the plain export drops marimo cell attributes and separators", () => {
  const notebook = '```python {.marimo name="setup"}\nimport polars\n```\n\n<!---->\n\n\n```python {.marimo}\nx = 1\n```';
  assert.equal(plainMarkdown(notebook), "```python\nimport polars\n```\n\n```python\nx = 1\n```");
});
