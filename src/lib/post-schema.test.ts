import assert from "node:assert/strict";
import { test } from "node:test";

import { postSchema } from "./post-schema.ts";

const post = { title: "a post", description: "about things", date: "2026-11-10" };
const issues = (data: object): string[] =>
  postSchema.safeParse(data).error?.issues.map(issue => issue.path.join(".")) ?? [];

test("a published post needs a lowercase title and a description", () => {
  assert.deepEqual(issues(post), []);
  assert.deepEqual(issues({ ...post, title: "A Post", description: " " }), ["title", "description"]);
});

test("a draft can have a capitalised title and no description yet", () => {
  assert.deepEqual(issues({ ...post, title: "How I Use Agents", description: "", draft: true }), []);
});

test("every post still needs a title and a date", () => {
  assert.deepEqual(issues({ ...post, title: "", draft: true }), ["title"]);
  assert.deepEqual(issues({ title: "x", description: "", draft: true }), ["date"]);
});

test("a song needs a title, an artist, a cover and a Spotify link, kept without its share id", () => {
  const song = { title: "song", artist: "artist", cover: "cover.jpg" };
  const url = "https://open.spotify.com/intl-it/track/4cOdK2wGLETKBW3PvgPWqT?si=a1b2c3";
  assert.equal(
    postSchema.parse({ ...post, song: { ...song, url } }).song?.url,
    "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT",
  );
  assert.deepEqual(issues({ ...post, song: { ...song, url: "https://youtu.be/dQw4w9WgXcQ" } }), ["song.url"]);
  assert.deepEqual(issues({ ...post, song: { title: "song", url } }), ["song.artist", "song.cover"]);
});
