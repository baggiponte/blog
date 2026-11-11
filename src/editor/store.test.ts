import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { type BundleDir, type ImageName, parseBundleDir } from "./post.ts";
import { fileStore, versionOf } from "./store.ts";

// A content/blog folder in a temporary directory, with these index.md files.
async function blog(posts: Record<string, string>): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "editor-store-"));
  for (const [name, markdown] of Object.entries(posts)) {
    await mkdir(join(root, name));
    await writeFile(join(root, name, "index.md"), markdown);
  }
  return root;
}

const dir = (text: string): BundleDir => parseBundleDir(text);
const post = (fields: string): string => `---\n${fields}\n---\n\nbody\n`;

test("lists hand-written posts newest first, leaving out notebooks and stray folders", async () => {
  const root = await blog({
    "01-first": post('title: "first post"\ndate: 2026-03-01'),
    "02-notebook": post("title: notebook\nnotebook: nb.md"),
    "03-third": post("title: third\ndate: 2026-01-01"),
    "zz-undated": post("title: undated"),
  });
  await mkdir(join(root, "empty-folder"));
  await mkdir(join(root, "Not A Bundle"));
  const listed = await fileStore(root).list();
  assert.deepEqual(
    listed.map(summary => summary.dir),
    ["01-first", "03-third", "zz-undated"],
  );
});

test("a save over a changed file is refused, unless it writes the same text", async () => {
  const root = await blog({ hello: post("title: hello") });
  const store = fileStore(root);
  const { version } = await store.read(dir("hello"));

  const next = await store.write(dir("hello"), "mine", version);
  assert.equal(next, versionOf("mine"));
  // Someone else writes; a save from the old version would overwrite them.
  await writeFile(join(root, "hello", "index.md"), "theirs");
  await assert.rejects(store.write(dir("hello"), "mine again", next), { failure: "conflict" });
  assert.equal(await readFile(join(root, "hello", "index.md"), "utf8"), "theirs");
  // The same text as on disk is not a conflict.
  assert.equal(await store.write(dir("hello"), "theirs", next), versionOf("theirs"));
  // No temporary file is left behind.
  assert.deepEqual(await readdir(join(root, "hello")), ["index.md"]);
});

test("only existing posts are read or written", async () => {
  const store = fileStore(await blog({}));
  await assert.rejects(store.read(dir("missing")), { failure: "missing" });
  await assert.rejects(store.write(dir("missing"), "x"), { failure: "missing" });
});

test("a new post can't take an existing folder", async () => {
  const root = await blog({});
  const store = fileStore(root);
  await store.create(dir("hello-world"), "first");
  assert.equal(await readFile(join(root, "hello-world", "index.md"), "utf8"), "first");
  await assert.rejects(store.create(dir("hello-world"), "second"), { failure: "exists" });
  assert.equal(await readFile(join(root, "hello-world", "index.md"), "utf8"), "first");
});

test("images land next to index.md, never over another file", async () => {
  const root = await blog({ hello: post("title: hello") });
  const store = fileStore(root);
  const bytes = new Uint8Array([1, 2, 3]);
  assert.equal(await store.addImage(dir("hello"), "My Plot.png", bytes), "my-plot.png");
  assert.equal(await store.addImage(dir("hello"), "my plot.png", bytes), "my-plot-2.png");
  const image = await store.readImage(dir("hello"), "my-plot-2.png" as ImageName);
  assert.deepEqual([...image.bytes], [1, 2, 3]);
  assert.equal(image.type, "image/png");
  await assert.rejects(store.readImage(dir("hello"), "nope.png" as ImageName), { failure: "missing" });
  await assert.rejects(store.addImage(dir("missing"), "a.png", bytes), { failure: "missing" });
});
