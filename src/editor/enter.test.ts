import assert from "node:assert/strict";
import { test } from "node:test";

import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { yamlFrontmatter } from "@codemirror/lang-yaml";
import { ensureSyntaxTree } from "@codemirror/language";
import { EditorState, type Transaction } from "@codemirror/state";

import { newParagraph } from "./enter.ts";

// Presses Enter at | in `marked`: the document afterwards with | at the cursor, or null when Enter is left to
// the other keymaps.
function enter(marked: string): string | null {
  const at = marked.indexOf("|");
  const state = EditorState.create({
    doc: marked.replace("|", ""),
    selection: { anchor: at },
    extensions: [yamlFrontmatter({ content: markdown({ base: markdownLanguage }) })],
  });
  ensureSyntaxTree(state, state.doc.length, 5000);
  let result: Transaction | undefined;
  const handled = newParagraph({ state, dispatch: tr => (result = tr) });
  if (!handled || result === undefined) {
    return null;
  }
  const doc = result.newDoc.toString();
  const head = result.state.selection.main.head;
  return `${doc.slice(0, head)}|${doc.slice(head)}`;
}

const front = '---\ntitle: "x"\n---\n\n';

test("Enter at the end of a paragraph starts a new one", () => {
  assert.equal(enter(`${front}one two|`), `${front}one two\n\n|`);
});

test("Enter mid-paragraph splits it in two", () => {
  assert.equal(enter(`${front}one| two`), `${front}one\n\n| two`);
});

test("lists, quotes, code and the frontmatter keep their own Enter", () => {
  assert.equal(enter(`${front}- one|`), null);
  assert.equal(enter(`${front}> one|`), null);
  assert.equal(enter(`${front}\`\`\`\ncode|\n\`\`\``), null);
  assert.equal(enter('---\ntitle: "x"|\n---\n\ntext'), null);
});

test("an empty line isn't a paragraph", () => {
  assert.equal(enter(`${front}one\n\n|`), null);
});
