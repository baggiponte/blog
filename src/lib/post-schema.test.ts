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
