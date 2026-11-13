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

test("a single newline starts a new paragraph, as a blank line does", async () => {
  assert.equal(await html("one\ntwo **bold**\n\nthree"), "<p>one</p>\n<p>two <strong>bold</strong></p>\n<p>three</p>");
});

test("lines in list items and quotes become paragraphs too, code blocks keep their lines", async () => {
  assert.equal(await html("> one\n> two"), "<blockquote>\n<p>one</p>\n<p>two</p>\n</blockquote>");
  const code = await html("```\na\nb\n```");
  assert.doesNotMatch(code, /<p>/);
  assert.equal(code.match(/class="line"/g)?.length, 2);
});

test("a hard break stays a line break inside the paragraph", async () => {
  assert.equal(await html("one\\\ntwo"), "<p>one<br>\ntwo</p>");
});

test("code blocks are captioned with the marimo cell name, else the language", async () => {
  const page = await html('```python {.marimo name="setup"}\nx = 1\n```\n\n```js\n1\n```\n\n```\nplain\n```');
  const captions = [...page.matchAll(/<figcaption><span>([^<]*)<\/span><\/figcaption>/g)].map(match => match[1]);
  assert.deepEqual(captions, ["setup", "js", "text"]);
});

test("the plain export drops marimo cell attributes and separators", () => {
  const notebook =
    '```python {.marimo name="setup"}\nimport polars\n```\n\n<!---->\n\n\n```python {.marimo}\nx = 1\n```';
  assert.equal(plainMarkdown(notebook), "```python\nimport polars\n```\n\n```python\nx = 1\n```");
});

test("a cover linked to Spotify on its own line becomes a card, a song with a record and a podcast without", async () => {
  const song = await html(
    '[![lose yourself](./cover.jpg "eminem")](https://open.spotify.com/intl-it/track/4cOdK2wGLETKBW3PvgPWqT?si=x)',
  );
  assert.match(
    song,
    /^<div class="spotify" data-kind="song"><a href="https:\/\/open\.spotify\.com\/track\/4cOdK2wGLETKBW3PvgPWqT" class="spotify-card"/,
  );
  assert.match(song, /<span class="vinyl" aria-hidden="true"><\/span>/);
  assert.match(song, /<b class="spotify-title">lose yourself<\/b><span class="spotify-by">eminem<\/span>/);
  const podcast = await html("[![an episode](./ep.jpg)](https://open.spotify.com/episode/4cOdK2wGLETKBW3PvgPWqT)");
  assert.match(podcast, /data-kind="podcast"/);
  assert.doesNotMatch(podcast, /vinyl|spotify-by/);
});

test("a Spotify cover inside a sentence, or a cover linked elsewhere, stays a linked image", async () => {
  const inline = await html("listen [![x](./c.jpg)](https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT) now");
  assert.doesNotMatch(inline, /spotify-card/);
  assert.doesNotMatch(await html("[![x](./c.jpg)](https://example.com/track/4cOdK2wGLETKBW3PvgPWqT)"), /spotify-card/);
});
