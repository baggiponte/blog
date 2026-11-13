import assert from "node:assert/strict";
import { test } from "node:test";

import { EditorError } from "./failure.ts";
import { changeOf, parseSuggestions, place, type Suggestion, withContext } from "./suggestion.ts";

const suggestion = (fields: Partial<Suggestion>): Suggestion => ({
  id: "s1",
  kind: "typo",
  quote: "teh",
  replacement: "the",
  ...fields,
});

test("a quote that appears once is found wherever it moved", () => {
  assert.deepEqual(place("we train teh model", suggestion({})), { found: true, from: 9, to: 12 });
  assert.deepEqual(place("first, a new line.\nwe train teh model", suggestion({})), { found: true, from: 28, to: 31 });
});

test("a repeated quote is told apart by the words around it, and a tie is not guessed", () => {
  const text = "fit the model. then fit the model again.";
  const second = suggestion({ quote: "the model", prefix: "then fit ", suffix: " again" });
  assert.deepEqual(place(text, second), { found: true, from: 24, to: 33 });
  // Partly edited context still points the right way.
  assert.deepEqual(place(text, { ...second, prefix: "and then fit " }), { found: true, from: 24, to: 33 });
  assert.deepEqual(place(text, suggestion({ quote: "the model" })), { found: false, why: "ambiguous" });
});

test("a quote whose words were edited away is missing", () => {
  assert.deepEqual(place("we train the model", suggestion({})), { found: false, why: "missing" });
});

test("context taken afresh finds the same place again", () => {
  const text = "fit the model. then fit the model again.";
  const fresh = withContext(suggestion({ quote: "the model" }), text, 24, 33);
  assert.deepEqual(place(text, fresh), { found: true, from: 24, to: 33 });
});

test("a file with a bad entry or a repeated id is refused, saying where", () => {
  assert.equal(parseSuggestions({ suggestions: [suggestion({})] }).length, 1);
  assert.throws(
    () => parseSuggestions({ suggestions: [suggestion({ kind: "vibes" as Suggestion["kind"] })] }),
    (error: unknown) =>
      error instanceof EditorError && error.failure === "invalid" && /suggestions\.0\.kind/.test(error.message),
  );
  assert.throws(() => parseSuggestions({ suggestions: [suggestion({}), suggestion({})] }), /same id/);
  assert.throws(() => parseSuggestions({ suggestions: [suggestion({ quote: "" })] }), /quote/);
});

test("only the words that change are marked, whole", () => {
  const change = (quote: string, replacement: string): string => {
    const { start, end, insert } = changeOf(quote, replacement);
    return `${quote.slice(start, end)} -> ${insert}`;
  };
  assert.equal(
    change("Small tasks, one commit each, is the only", "Small tasks, one commit each, are the only"),
    "is -> are",
  );
  assert.equal(change("a very very small scope", "a tiny scope"), "very very small -> tiny");
  assert.equal(change("teh", "the"), "teh -> the");
  assert.equal(change("the hole page", "the whole page"), "hole -> whole");
  assert.equal(change("so very big", "so big"), "very  -> ");
  assert.equal(change("model", ""), "model -> ");
});
