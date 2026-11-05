import { syntaxTree } from "@codemirror/language";
import { type EditorState, type Extension, type Range, RangeSet } from "@codemirror/state";
import { Decoration, type DecorationSet, EditorView, ViewPlugin, type ViewUpdate, WidgetType } from "@codemirror/view";

// Live preview, like Obsidian's: the document stays the exact Markdown of index.md, but away from the cursor the
// syntax is hidden and the text is styled as the post shows it. The lines the cursor or selection touch show their
// Markdown in full, so every character can still be edited.

// Syntax hidden away from the cursor: heading hashes, emphasis and code marks, and a link's brackets and URL.
const hiddenMarks = new Set(["HeaderMark", "EmphasisMark", "StrikethroughMark", "QuoteMark"]);
const inlineCodeMark = "CodeMark";
const lineClasses: Record<string, string> = {
  ATXHeading1: "cm-lp-h1",
  ATXHeading2: "cm-lp-h2",
  ATXHeading3: "cm-lp-h3",
  ATXHeading4: "cm-lp-h4",
  ATXHeading5: "cm-lp-h4",
  ATXHeading6: "cm-lp-h4",
  Blockquote: "cm-lp-quote",
  FencedCode: "cm-lp-code",
  Frontmatter: "cm-lp-frontmatter",
  HorizontalRule: "cm-lp-rule",
};
// `==text==`, the site's highlight (src/lib/remark-mark.ts), which Markdown itself doesn't parse.
const markPattern = /==(?=\S)(.+?)(?<=\S)==/g;

export class ImageWidget extends WidgetType {
  readonly src: string;
  readonly alt: string;
  constructor(src: string, alt: string) {
    super();
    this.src = src;
    this.alt = alt;
  }
  override eq(other: ImageWidget): boolean {
    return other.src === this.src && other.alt === this.alt;
  }
  override toDOM(): HTMLElement {
    const image = document.createElement("img");
    image.className = "cm-lp-image";
    image.src = this.src;
    image.alt = this.alt;
    return image;
  }
}

class BulletWidget extends WidgetType {
  override eq(): boolean {
    return true;
  }
  override toDOM(): HTMLElement {
    const bullet = document.createElement("span");
    bullet.className = "cm-lp-bullet";
    bullet.textContent = "•";
    return bullet;
  }
}

const hide = Decoration.replace({});
const bullet = Decoration.replace({ widget: new BulletWidget() });
const highlight = Decoration.mark({ class: "cm-lp-mark" });
const link = Decoration.mark({ class: "cm-lp-link" });
const line = (cls: string) => Decoration.line({ class: cls });

// The line numbers (1-based) that the selection touches, which keep their Markdown visible.
function activeLines(state: EditorState): Set<number> {
  const lines = new Set<number>();
  for (const range of state.selection.ranges) {
    const first = state.doc.lineAt(range.from).number;
    const last = state.doc.lineAt(range.to).number;
    for (let n = first; n <= last; n++) {
      lines.add(n);
    }
  }
  return lines;
}

// Where an image in the post's folder is served from, so it can be shown inline; other URLs are used as they are.
export type ImageSource = (url: string) => string;

// `./plot.png` or `plot.png` in content/blog/<dir>/ comes from the editor API; `/x.png` and `https://` stay put.
export function postImageUrl(apiBase: string, dir: string, url: string): string {
  if (/^([a-z][a-z0-9+.-]*:|\/)/i.test(url)) {
    return url;
  }
  const name = url.replace(/^\.\//, "");
  return `${apiBase}/posts/${encodeURIComponent(dir)}/images/${encodeURIComponent(name)}`;
}

export function livePreviewDecorations(state: EditorState, imageSource: ImageSource): DecorationSet {
  const decorations: Range<Decoration>[] = [];
  const active = activeLines(state);
  const isActive = (from: number, to: number): boolean => {
    const first = state.doc.lineAt(from).number;
    const last = state.doc.lineAt(to).number;
    for (let n = first; n <= last; n++) {
      if (active.has(n)) {
        return true;
      }
    }
    return false;
  };
  const styleLines = (from: number, to: number, cls: string): void => {
    for (let n = state.doc.lineAt(from).number; n <= state.doc.lineAt(to).number; n++) {
      decorations.push(line(cls).range(state.doc.line(n).from));
    }
  };
  const code: [number, number][] = [];

  syntaxTree(state).iterate({
    enter: node => {
      const cls = lineClasses[node.name];
      if (cls !== undefined) {
        // A frontmatter's closing line ends with its newline, which belongs to the next line.
        styleLines(node.from, node.name === "Frontmatter" ? node.to - 1 : node.to, cls);
      }
      if (node.name === "Frontmatter" || node.name === "FencedCode" || node.name === "InlineCode") {
        code.push([node.from, node.to]);
      }
      if (node.name === "Frontmatter" || node.name === "FencedCode") {
        return false;
      }
      if (isActive(node.from, node.to)) {
        return undefined;
      }
      if (hiddenMarks.has(node.name) || (node.name === inlineCodeMark && node.node.parent?.name === "InlineCode")) {
        // A heading's or quote's mark takes the space after it along.
        const space = node.name === "HeaderMark" || node.name === "QuoteMark" ? 1 : 0;
        const end = Math.min(node.to + space, state.doc.lineAt(node.from).to);
        decorations.push(hide.range(node.from, end));
      } else if (node.name === "ListMark" && /^[-*+]$/.test(state.sliceDoc(node.from, node.to))) {
        decorations.push(bullet.range(node.from, node.to));
      } else if (node.name === "Image") {
        const url = node.node.getChild("URL");
        const altEnd = node.node.getChildren("LinkMark")[1]?.from ?? node.from + 2;
        if (url !== null) {
          const widget = new ImageWidget(
            imageSource(state.sliceDoc(url.from, url.to)),
            state.sliceDoc(node.from + 2, altEnd),
          );
          decorations.push(Decoration.replace({ widget }).range(node.from, node.to));
        }
        return false;
      } else if (node.name === "Link") {
        const marks = node.node.getChildren("LinkMark");
        const close = marks[1];
        if (marks[0] !== undefined && close !== undefined) {
          decorations.push(hide.range(marks[0].from, marks[0].to));
          decorations.push(link.range(marks[0].to, close.from));
          decorations.push(hide.range(close.from, node.to));
        }
        return false;
      }
      return undefined;
    },
  });

  // Highlights outside code, one line at a time like the remark plugin.
  const inCode = (position: number): boolean => code.some(([from, to]) => position >= from && position < to);
  for (let n = 1; n <= state.doc.lines; n++) {
    const { from, text } = state.doc.line(n);
    for (const match of text.matchAll(markPattern)) {
      const start = from + match.index;
      const end = start + match[0].length;
      if (inCode(start)) {
        continue;
      }
      if (active.has(n)) {
        decorations.push(highlight.range(start, end));
      } else {
        decorations.push(hide.range(start, start + 2), highlight.range(start + 2, end - 2), hide.range(end - 2, end));
      }
    }
  }
  return RangeSet.of(decorations, true);
}

export function livePreview(imageSource: ImageSource): Extension {
  return ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;
      constructor(view: EditorView) {
        this.decorations = livePreviewDecorations(view.state, imageSource);
      }
      update(update: ViewUpdate): void {
        if (update.docChanged || update.selectionSet || syntaxTree(update.startState) !== syntaxTree(update.state)) {
          this.decorations = livePreviewDecorations(update.state, imageSource);
        }
      }
    },
    { decorations: plugin => plugin.decorations },
  );
}

// Clicking an inline image puts the cursor on it, which shows its Markdown.
export const revealOnClick = EditorView.domEventHandlers({
  mousedown: (event, view) => {
    if (!(event.target instanceof HTMLImageElement && event.target.classList.contains("cm-lp-image"))) {
      return false;
    }
    const position = view.posAtDOM(event.target);
    view.dispatch({ selection: { anchor: position } });
    view.focus();
    event.preventDefault();
    return true;
  },
});
