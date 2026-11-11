import assert from "node:assert/strict";
import { test } from "node:test";

import {
  type BundleDir,
  checkPost,
  dirForTitle,
  imageName,
  newPostMarkdown,
  numbered,
  parseBundleDir,
  parseImageName,
  parseNewPost,
  summarize,
} from "./post.ts";

const dir = (text: string): BundleDir => parseBundleDir(text);
const post = (fields: string): string => `---\n${fields}\n---\n\nbody\n`;

test("folder names can't climb out of content/blog or hide", () => {
  assert.equal(parseBundleDir("01-groundup-linear-regression"), "01-groundup-linear-regression");
  for (const sneaky of ["../hello", "hello/..", "Hello", ".git", "", "a--b"]) {
    assert.throws(() => parseBundleDir(sneaky), { failure: "invalid" }, sneaky);
  }
});

test("a new post needs a title, a description and a known format", () => {
  assert.deepEqual(parseNewPost({ title: "t", description: "d", format: "tidbits" }), {
    title: "t",
    description: "d",
    format: "tidbits",
  });
  for (const bad of [undefined, { title: "t" }, { title: "t", description: "d", format: "essay" }]) {
    assert.throws(() => parseNewPost(bad), { failure: "invalid" });
  }
});

test("a new post is a draft in a folder named after its title", () => {
  const input = { title: "Café: The Ground Up!", description: 'say "hi"', format: "tidbits" } as const;
  assert.equal(dirForTitle(input.title), "cafe-the-ground-up");
  assert.throws(() => dirForTitle("!!!"), { failure: "invalid" });
  const markdown = newPostMarkdown(input, new Date("2026-05-05T10:00:00Z"));
  assert.equal(
    markdown,
    '---\ntitle: "café: the ground up!"\ndescription: "say \\"hi\\""\ndate: 2026-05-05\nformat: tidbits\ndraft: true\n---\n\n',
  );
  // What the editor creates, it can also save.
  checkPost(markdown);
});

test("a post the site couldn't build is refused, with the reason", () => {
  checkPost(post('title: "a post"\ndescription: "about it"\ndate: 2026-11-10'));
  assert.throws(() => checkPost(post('title: "unclosed\ndate: 2026-11-10')), {
    failure: "invalid",
    message: /isn't valid YAML/,
  });
  assert.throws(() => checkPost(post('title: "A Post"\ndescription: ""\ndate: 2026-11-10')), {
    failure: "invalid",
    message: /^title: .*; description: /,
  });
  assert.throws(() => checkPost(post('title: "x"\ndescription: "y"\ndraft: true')), { message: /^date: / });
});

test("the sidebar row reads the frontmatter as YAML, and skips notebook posts", () => {
  assert.deepEqual(summarize(dir("03-third"), post("title: 'third: the sequel'\ndate: 2026-01-01\ndraft: true")), {
    dir: "03-third",
    title: "third: the sequel",
    date: "2026-01-01",
    draft: true,
    url: "/blog/third/",
  });
  assert.equal(summarize(dir("nb"), post("title: notebook\nnotebook: nb.md")), undefined);
  // A file broken outside the editor is still listed, so it can be opened and fixed.
  assert.deepEqual(summarize(dir("broken"), post('title: "unclosed')), {
    dir: "broken",
    title: "broken",
    date: "",
    draft: false,
    url: "/blog/broken/",
  });
});

test("images get web-friendly names, numbered when taken", () => {
  const name = imageName("Screenshot 2026-11-05 at 10.00.PNG");
  assert.equal(name, "screenshot-2026-11-05-at-10-00.png");
  assert.equal(numbered(name, 1), name);
  assert.equal(numbered(name, 2), "screenshot-2026-11-05-at-10-00-2.png");
  assert.equal(imageName("Photo.JPEG"), "photo.jpg");
  assert.equal(imageName("日本.webp"), "image.webp");
  assert.throws(() => imageName("notes.md"), { failure: "unsupported" });
  assert.throws(() => imageName("png"), { failure: "unsupported" });
});

test("only plain image file names can be read back", () => {
  assert.equal(parseImageName("shot.png"), "shot.png");
  for (const sneaky of ["index.md", "../index.md", "../../secret.png", ".hidden.png", "a/b.png"]) {
    assert.throws(() => parseImageName(sneaky), { failure: "invalid" }, sneaky);
  }
});
