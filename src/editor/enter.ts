import { syntaxTree } from "@codemirror/language";
import type { EditorState, StateCommand } from "@codemirror/state";

// Enter in a paragraph starts a new paragraph: Markdown needs a blank line for one, and a single newline only
// continues the paragraph on the site. Shift+Enter still inserts a single newline. Lists, quotes and code keep
// their own Enter, from lang-markdown and the default keymap.

const ownEnter = new Set(["ListItem", "Blockquote", "FencedCode", "CodeBlock", "Table"]);

// A paragraph of its own, not one inside a list item or a quote.
function inParagraph(state: EditorState, position: number): boolean {
  let paragraph = false;
  for (const side of [-1, 1] as const) {
    for (let node = syntaxTree(state).resolveInner(position, side); node.parent !== null; node = node.parent) {
      if (ownEnter.has(node.name)) {
        return false;
      }
      paragraph ||= node.name === "Paragraph";
    }
  }
  return paragraph;
}

export const newParagraph: StateCommand = ({ state, dispatch }) => {
  const { ranges } = state.selection;
  if (ranges.some(range => !(range.empty && inParagraph(state, range.head)))) {
    return false;
  }
  dispatch(state.update(state.replaceSelection("\n\n"), { scrollIntoView: true, userEvent: "input" }));
  return true;
};

// A single newline, which continues the paragraph: a line break in the source only.
export const softBreak: StateCommand = ({ state, dispatch }) => {
  dispatch(state.update(state.replaceSelection("\n"), { scrollIntoView: true, userEvent: "input" }));
  return true;
};

export const enterKeys = [
  { key: "Enter", run: newParagraph },
  { key: "Shift-Enter", run: softBreak },
];
