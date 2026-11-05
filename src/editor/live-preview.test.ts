import assert from "node:assert/strict";
import { test } from "node:test";

import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { yamlFrontmatter } from "@codemirror/lang-yaml";
import { ensureSyntaxTree } from "@codemirror/language";
import { EditorSelection, EditorState } from "@codemirror/state";
import type { Decoration } from "@codemirror/view";

import { ImageWidget, livePreviewDecorations, postImageUrl } from "./live-preview.ts";

const doc = [
  "---",
  'title: "x"',
  "---",
  "",
  "## a *heading*",
  "",
  "some **bold**, `code`, [a link](https://x.y) and ==a mark==.",
  "",
  "![a plot](./plot.png)",
  "",
  "- item",
  "",
  "```py",
  "x == y == z",
  "```",
  "",
].join("\n");

// The document as it reads in the editor with the cursor at `cursor`: hidden text removed, images as [img src].
function rendered(cursor: number): string {
  let state = EditorState.create({
    doc,
    selection: EditorSelection.cursor(cursor),
    extensions: [yamlFrontmatter({ content: markdown({ base: markdownLanguage }) })],
  });
  ensureSyntaxTree(state, state.doc.length, 5000);
  state = state.update({}).state;
  const replaced: { from: number; to: number; value: Decoration }[] = [];
  livePreviewDecorations(state, url => `/files/${url}`).between(0, doc.length, (from, to, value) => {
    if (to > from && value.spec.widget !== undefined) {
      replaced.push({ from, to, value });
    } else if (to > from && value.spec.class === undefined) {
      replaced.push({ from, to, value });
    }
  });
  let out = "";
  let at = 0;
  for (const { from, to, value } of replaced.toSorted((a, b) => a.from - b.from)) {
    out += doc.slice(at, from);
    const widget = value.spec.widget;
    if (widget instanceof ImageWidget) {
      out += `[img ${widget.src} "${widget.alt}"]`;
    } else if (widget !== undefined) {
      out += "•";
    }
    at = to;
  }
  return out + doc.slice(at);
}

const lineOf = (text: string): number => doc.indexOf(text);

test("away from the cursor, the syntax is hidden and images show", () => {
  const view = rendered(0);
  assert.match(view, /^a heading$/m);
  assert.match(view, /^some bold, code, a link and a mark\.$/m);
  assert.match(view, /^\[img \/files\/\.\/plot\.png "a plot"\]$/m);
  assert.match(view, /^• item$/m);
});

test("the frontmatter and code blocks always show their source", () => {
  const view = rendered(lineOf("some"));
  assert.match(view, /^---\ntitle: "x"\n---$/m);
  assert.match(view, /^```py\nx == y == z\n```$/m);
});

test("the line with the cursor shows its Markdown, the others don't", () => {
  const view = rendered(lineOf("some") + 2);
  assert.match(view, /^some \*\*bold\*\*, `code`, \[a link\]\(https:\/\/x\.y\) and ==a mark==\.$/m);
  assert.match(view, /^a heading$/m);
  assert.match(rendered(lineOf("![a plot")), /^!\[a plot\]\(\.\/plot\.png\)$/m);
});

test("images in the post's folder are served by the editor API; others stay put", () => {
  assert.equal(postImageUrl("/api", "my-post", "./plot 1.png"), "/api/posts/my-post/images/plot%201.png");
  assert.equal(postImageUrl("/api", "my-post", "plot.png"), "/api/posts/my-post/images/plot.png");
  assert.equal(postImageUrl("/api", "my-post", "/favicon.svg"), "/favicon.svg");
  assert.equal(postImageUrl("/api", "my-post", "https://x.y/a.png"), "https://x.y/a.png");
});
