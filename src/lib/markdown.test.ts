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

const block = (fields: string): string => `before\n\n\`\`\`card\n${fields}\n\`\`\`\n\nafter`;
const episode = [
  "title: an episode",
  "note: from minute 32",
  "url: https://open.spotify.com/intl-it/episode/4cOdK2wGLETKBW3PvgPWqT?si=x",
  "image: ./episode.jpg",
].join("\n");

test("a card block for a Spotify episode is a podcast card that says where it opens, wherever it sits", async () => {
  const page = await html(block(episode));
  assert.match(page, /^<p>before<\/p>\n<div class="player" data-type="podcast" data-platform="spotify">/);
  assert.match(page, /<a href="https:\/\/open\.spotify\.com\/episode\/4cOdK2wGLETKBW3PvgPWqT" class="player-card"/);
  assert.match(page, /<img src="\.\/episode\.jpg" alt="" width="256" height="256">/);
  assert.match(page, /<b class="player-title">an episode<\/b><span class="player-note">from minute 32<\/span>/);
  assert.match(page, /<span class="player-go">listen on spotify ↗<\/span>/);
  assert.match(page, /<\/div>\n<p>after<\/p>$/);
  assert.doesNotMatch(page, /vinyl|<code/);
});

test("only the link is needed: a track is a song, with a title from its type and a blank sleeve", async () => {
  const song = await html(block("url: https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"));
  assert.match(song, /data-type="song"/);
  assert.match(song, /<span class="vinyl" aria-hidden="true"><\/span><span class="blank" aria-hidden="true"><\/span>/);
  assert.match(song, /<b class="player-title">a song worth a listen<\/b>/);
  assert.doesNotMatch(song, /player-note|<img/);
});

test("a YouTube link is a video unless the block says otherwise, and the call to action follows the platform", async () => {
  const video = await html(block("url: https://youtu.be/dQw4w9WgXcQ\nimage: ./frame.jpg"));
  assert.match(video, /data-type="video" data-platform="youtube"/);
  assert.match(video, /a video worth a watch.*watch on youtube ↗/);
  assert.match(video, /<img src="\.\/frame\.jpg" alt="" width="320" height="180">/);
  const podcast = await html(block("type: podcast\nurl: https://youtu.be/dQw4w9WgXcQ"));
  assert.match(podcast, /data-type="podcast" data-platform="youtube"/);
  assert.match(podcast, /an episode worth a watch/);
});

test("any other address is a text link card with its title, note and site", async () => {
  const page = await html(
    block(
      [
        "url: https://jvns.ca/blog/git-config/?utm_source=rss&page=2",
        "title: popular git config options",
        "note: the options people set",
        "site: julia evans",
      ].join("\n"),
    ),
  );
  assert.match(page, /^<p>before<\/p>\n<div class="link">/);
  assert.match(page, /<a href="https:\/\/jvns\.ca\/blog\/git-config\/\?page=2" class="link-card" target="_blank"/);
  assert.match(page, /<b class="link-title">popular git config options<\/b><span class="link-desc">the options/);
  assert.match(page, /<span class="link-name">julia evans<\/span><span class="link-host">jvns\.ca<\/span> ↗/);
});

test("a link card needs only its address: the address as title, the host as site", async () => {
  const page = await html(block("url: https://www.example.com/notes/caffè/"));
  assert.match(page, /<div class="link">/);
  assert.match(page, /<b class="link-title">example\.com\/notes\/caffè<\/b>/);
  assert.match(page, /<span class="link-name">example\.com<\/span> ↗/);
  assert.doesNotMatch(page, /link-desc|link-host|<img/);
  // A Spotify link can still be a bookmark when the block says so.
  assert.match(
    await html(block("type: link\nurl: https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT")),
    /link-card/,
  );
});

test("a card block with an unknown field, a bad value or a link its type can't take stops the build", async () => {
  await assert.rejects(html(block(`${episode}\nby: someone`)), /:3: .*by/);
  await assert.rejects(html(block(`type: music\n${episode}`)), /type: /);
  await assert.rejects(html(block(episode.replace(/url: .*/, ""))), /url: /);
  await assert.rejects(
    html(block("type: song\nurl: https://example.com/x")),
    /url: a Spotify or YouTube link for a song/,
  );
  await assert.rejects(html(block("url: mailto:me@example.com")), /url: an http or https link/);
  await assert.rejects(html(block(`${episode}\nsite: spotify`)), /site: only a link card/);
  await assert.rejects(
    html(block("url: https://example.com\nimage: https://example.com/og.png")),
    /image: a file next/,
  );
  await assert.rejects(html(block("title: [unclosed")), /a card block holds YAML fields/);
});

test("other code blocks are left alone", async () => {
  assert.match(await html("```yaml\ntitle: x\n```"), /<figure class="code"/);
});
