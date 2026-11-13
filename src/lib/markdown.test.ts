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

const block = (fields: string): string => `before\n\n\`\`\`player\n${fields}\n\`\`\`\n\nafter`;
const episode = [
  "type: podcast",
  "title: an episode",
  "by: a show",
  "note: from minute 32",
  "url: https://open.spotify.com/intl-it/episode/4cOdK2wGLETKBW3PvgPWqT?si=x",
  "cover: ./episode.jpg",
].join("\n");

test("a player block becomes a card that says where it opens, wherever it sits", async () => {
  const page = await html(block(episode));
  assert.match(page, /^<p>before<\/p>\n<div class="player" data-type="podcast" data-platform="spotify">/);
  assert.match(page, /<a href="https:\/\/open\.spotify\.com\/episode\/4cOdK2wGLETKBW3PvgPWqT" class="player-card"/);
  assert.match(page, /<b class="player-title">an episode<\/b><span class="player-by">a show<\/span>/);
  assert.match(
    page,
    /<span class="player-note">from minute 32<\/span><span class="player-go">listen on spotify ↗<\/span>/,
  );
  assert.match(page, /<\/div>\n<p>after<\/p>$/);
  assert.doesNotMatch(page, /vinyl|<code/);
});

test("music gets a record and needs no type, since it's the default", async () => {
  const song = await html(block(episode.replace("type: podcast\n", "").replace("note: from minute 32\n", "")));
  assert.match(song, /data-type="music"/);
  assert.match(song, /<span class="vinyl" aria-hidden="true"><\/span>/);
  assert.doesNotMatch(song, /player-note/);
});

test("a player block with a missing field, an unknown one or a link no platform takes stops the build", async () => {
  await assert.rejects(html(block(episode.replace("by: a show\n", ""))), /:3: by: /);
  await assert.rejects(html(block(`${episode}\nartist: someone`)), /artist/);
  await assert.rejects(html(block(episode.replace(/url: .*/, "url: https://youtu.be/x"))), /url: a Spotify/);
  await assert.rejects(html(block("title: [unclosed")), /a player block holds YAML fields/);
});

test("other code blocks are left alone", async () => {
  assert.match(await html("```yaml\ntitle: x\n```"), /<figure class="code"/);
});
