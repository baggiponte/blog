import assert from "node:assert/strict";
import { test } from "node:test";

import { slugify, urlOfDir } from "./slugs.ts";

test("slugs are lowercase words joined by single dashes, accents dropped", () => {
  assert.equal(slugify("Café au lait!"), "cafe-au-lait");
  assert.equal(slugify("polars & friends"), "polars-friends");
  assert.equal(slugify("  --LLM inference: arithmetics--  "), "llm-inference-arithmetics");
  assert.equal(slugify("!!!"), "");
});

test("a bundle's URL drops its ordering prefix", () => {
  assert.equal(urlOfDir("01-groundup-linear-regression"), "/blog/groundup-linear-regression/");
  assert.equal(urlOfDir("hello"), "/blog/hello/");
});
