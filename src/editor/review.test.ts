import assert from "node:assert/strict";
import { test } from "node:test";

import { history, undo } from "@codemirror/commands";
import { EditorState, type TransactionSpec } from "@codemirror/state";

import {
  accept,
  jump,
  load,
  ReplacementWidget,
  reject,
  remaining,
  review,
  reviewDecorations,
  reviewField,
  type Target,
} from "./review.ts";
import type { Suggestion } from "./suggestion.ts";

const typo: Suggestion = { id: "typo", kind: "typo", quote: "teh", replacement: "the" };
const style: Suggestion = { id: "style", kind: "style", quote: "very big", replacement: "huge" };
const gone: Suggestion = { id: "gone", kind: "grammar", quote: "not in the text", replacement: "x" };

// An editor state that updates itself, like a view would, with the suggestions loaded.
function editor(doc: string, suggestions: Suggestion[]): Target & { text: () => string } {
  let state = EditorState.create({ doc, extensions: [history(), review()] });
  state = state.update({ effects: load.of(suggestions) }).state;
  return {
    get state() {
      return state;
    },
    dispatch(spec: TransactionSpec) {
      state = state.update(spec).state;
    },
    text: () => state.doc.toString(),
  };
}
const open = (target: Target): string[] => remaining(target.state.field(reviewField)).map(s => s.id);
const lost = (target: Target): string[] => target.state.field(reviewField).lost.map(s => s.id);

test("loading places each suggestion on its quote, and lists the ones it can't place", () => {
  const view = editor("a very big teh model", [typo, style, gone]);
  const placed = view.state.field(reviewField).placed.map(p => [p.suggestion.id, p.from, p.to]);
  assert.deepEqual(placed, [
    ["style", 2, 10],
    ["typo", 11, 14],
  ]);
  assert.deepEqual(lost(view), ["gone"]);
});

test("accepting writes the replacement, or the edited text, and closes the suggestion", () => {
  const view = editor("a very big teh model", [typo, style]);
  accept(view, "typo");
  assert.equal(view.text(), "a very big the model");
  accept(view, "style", "large");
  assert.equal(view.text(), "a large the model");
  assert.deepEqual(open(view), []);
});

test("rejecting leaves the text as it is", () => {
  const view = editor("a very big teh model", [typo]);
  reject(view, "typo");
  assert.equal(view.text(), "a very big teh model");
  assert.deepEqual(open(view), []);
});

test("undo brings back the text and the suggestion, in its place", () => {
  const view = editor("a very big teh model", [typo]);
  accept(view, "typo");
  undo(view as Parameters<typeof undo>[0]);
  assert.equal(view.text(), "a very big teh model");
  const [placed] = view.state.field(reviewField).placed;
  assert.deepEqual([placed?.suggestion.id, placed?.from, placed?.to], ["typo", 11, 14]);
});

test("typing elsewhere moves suggestions along; typing inside one loses it", () => {
  const view = editor("a very big teh model", [typo, style]);
  view.dispatch({ changes: { from: 0, insert: "so, " } });
  const typoAt = view.state.field(reviewField).placed.find(p => p.suggestion.id === "typo");
  assert.equal(view.state.sliceDoc(typoAt?.from, typoAt?.to), "teh");
  // "very big" becomes "very, very big".
  view.dispatch({ changes: { from: 10, insert: ", very" } });
  assert.deepEqual(lost(view), ["style"]);
  assert.deepEqual(open(view), ["typo", "style"]);
});

test("jumping goes to the next suggestion and wraps around", () => {
  const view = editor("a very big teh model", [typo, style]);
  jump(view, 1);
  assert.equal(view.state.selection.main.head, 2);
  jump(view, 1);
  assert.equal(view.state.selection.main.head, 11);
  jump(view, 1);
  assert.equal(view.state.selection.main.head, 2);
});

test("only the words that change are struck, with the new words right after them", () => {
  const view = editor("a very big teh model", [{ ...typo, quote: "big teh model", replacement: "big the model" }]);
  const shown: string[] = [];
  reviewDecorations(view.state).between(0, view.text().length, (from, to, value) => {
    const widget = value.spec.widget;
    shown.push(widget instanceof ReplacementWidget ? `+${widget.text}@${from}` : `-${view.state.sliceDoc(from, to)}`);
  });
  assert.deepEqual(shown, ["-teh", "+the@14"]);
});
