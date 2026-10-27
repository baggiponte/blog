import assert from "node:assert/strict";
import { test } from "node:test";

import { type SearchEntry, search } from "./search.ts";

// Newest first, as /search.json lists them.
const entry = (title: string, text: string): SearchEntry => ({ title, text, description: "", date: "", url: "" });
const index = [
  entry("from the ground up: linear classifier", "a perceptron in numpy"),
  entry("polars tricks", "lazy frames and the linear algebra you skip"),
  entry("from the ground up: linear regression", "least squares in numpy"),
];
const titles = (query: string): string[] => search(index, query).map(hit => hit.title);

test("every word must appear somewhere, in any order", () => {
  assert.deepEqual(titles("NUMPY linear"), [
    "from the ground up: linear classifier",
    "from the ground up: linear regression",
  ]);
  assert.deepEqual(titles("numpy polars"), []);
});

test("posts matching more words in the title come first, then newest", () => {
  assert.deepEqual(titles("linear"), [
    "from the ground up: linear classifier",
    "from the ground up: linear regression",
    "polars tricks",
  ]);
  assert.deepEqual(titles("linear regression"), ["from the ground up: linear regression"]);
  assert.deepEqual(titles("polars linear"), ["polars tricks"]);
});
