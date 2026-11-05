import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { createPost, imageFileName, listPosts, readPost, saveImage, writePost } from "./posts.ts";

async function blog(posts: Record<string, string>): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "editor-"));
  for (const [name, markdown] of Object.entries(posts)) {
    await mkdir(join(dir, name));
    await writeFile(join(dir, name, "index.md"), markdown);
  }
  return dir;
}

const frontmatter = (fields: string): string => `---\n${fields}\n---\n\nbody\n`;
const may5 = new Date("2026-05-05T10:00:00Z");

test("lists hand-written posts newest folder first, without notebook posts", async () => {
  const dir = await blog({
    "01-first": frontmatter('title: "first post"\ndraft: false'),
    "02-notebook": frontmatter("title: notebook\nnotebook: nb.md"),
    "03-third": frontmatter("title: 'third: the sequel'\ndraft: true"),
  });
  await mkdir(join(dir, "empty-folder"));
  assert.deepEqual(await listPosts(dir), [
    { dir: "03-third", title: "third: the sequel", draft: true, url: "/blog/third/" },
    { dir: "01-first", title: "first post", draft: false, url: "/blog/first/" },
  ]);
});

test("a new post is a draft bundle named after its lowercased title", async () => {
  const dir = await blog({});
  const post = await createPost(
    dir,
    { title: "Café: The Ground Up!", description: 'say "hi"', format: "tidbits" },
    may5,
  );
  assert.equal(post, "cafe-the-ground-up");
  assert.equal(
    await readPost(dir, post),
    '---\ntitle: "café: the ground up!"\ndescription: "say \\"hi\\""\ndate: 2026-05-05\nformat: tidbits\ndraft: true\n---\n\n',
  );
  await assert.rejects(createPost(dir, { title: "cafe the ground up", description: "", format: "post" }, may5), {
    status: 409,
  });
  await assert.rejects(createPost(dir, { title: "!!!", description: "", format: "post" }, may5), { status: 400 });
});

test("only an existing post's index.md can be read or written", async () => {
  const dir = await blog({ hello: frontmatter("title: hello") });
  await writePost(dir, "hello", "new text");
  assert.equal(await readFile(join(dir, "hello", "index.md"), "utf8"), "new text");
  await assert.rejects(writePost(dir, "missing", "x"), { status: 404 });
  for (const sneaky of ["../hello", "hello/..", "Hello", ".git", ""]) {
    await assert.rejects(readPost(dir, sneaky), { status: 400 }, sneaky);
  }
});

test("dropped images get a web-friendly name next to index.md, never over another file", async () => {
  const dir = await blog({ hello: frontmatter("title: hello") });
  const bytes = new Uint8Array([1, 2, 3]);
  assert.equal(
    await saveImage(dir, "hello", "Screenshot 2026-11-05 at 10.00.PNG", bytes),
    "screenshot-2026-11-05-at-10-00.png",
  );
  assert.equal(
    await saveImage(dir, "hello", "screenshot 2026-11-05 at 10.00.png", bytes),
    "screenshot-2026-11-05-at-10-00-2.png",
  );
  assert.deepEqual((await readdir(join(dir, "hello"))).toSorted(), [
    "index.md",
    "screenshot-2026-11-05-at-10-00-2.png",
    "screenshot-2026-11-05-at-10-00.png",
  ]);
  await assert.rejects(saveImage(dir, "missing", "a.png", bytes), { status: 404 });
});

test("image names keep a known image extension", () => {
  assert.equal(imageFileName("Photo.JPEG"), "photo.jpg");
  assert.equal(imageFileName("日本.webp"), "image.webp");
  assert.throws(() => imageFileName("notes.md"), { status: 415 });
  assert.throws(() => imageFileName("png"), { status: 415 });
});
