import { type EditorView, layer, RectangleMarker } from "@codemirror/view";

// The selection as the text itself: one band per row of selected text, as wide as the text. CodeMirror's own
// selection fills from the start of a selection to the right edge and down through the space between
// paragraphs, which reads as having selected the gap too.

export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

// Joins the boxes a browser gives for a stretch of text (one per text node or inline element) into one box per
// row: boxes whose middles sit inside each other's height are on the same row.
export function rows(boxes: readonly Box[]): Box[] {
  const result: Box[] = [];
  for (const box of boxes) {
    if (box.right - box.left <= 0) {
      continue;
    }
    const middle = (box.top + box.bottom) / 2;
    const row = result.find(r => middle > r.top && middle < r.bottom);
    if (row === undefined) {
      // Copied field by field: a DOMRect's fields are getters, which spreading leaves out.
      result.push({ left: box.left, top: box.top, right: box.right, bottom: box.bottom });
    } else {
      row.left = Math.min(row.left, box.left);
      row.right = Math.max(row.right, box.right);
      row.top = Math.min(row.top, box.top);
      row.bottom = Math.max(row.bottom, box.bottom);
    }
  }
  return result;
}

function textBoxes(view: EditorView, from: number, to: number): Box[] {
  const start = view.domAtPos(from);
  const end = view.domAtPos(to);
  const range = document.createRange();
  try {
    range.setStart(start.node, start.offset);
    range.setEnd(end.node, end.offset);
  } catch {
    return [];
  }
  return [...range.getClientRects()];
}

function markers(view: EditorView): RectangleMarker[] {
  const scroller = view.scrollDOM.getBoundingClientRect();
  const left = scroller.left - view.scrollDOM.scrollLeft * view.scaleX;
  const top = scroller.top - view.scrollDOM.scrollTop * view.scaleY;
  const result: RectangleMarker[] = [];
  for (const range of view.state.selection.ranges) {
    if (range.empty) {
      continue;
    }
    for (const visible of view.visibleRanges) {
      const from = Math.max(range.from, visible.from);
      const to = Math.min(range.to, visible.to);
      for (let pos = from; pos < to; ) {
        const line = view.state.doc.lineAt(pos);
        const end = Math.min(to, line.to);
        for (const row of rows(textBoxes(view, pos, end))) {
          result.push(
            new RectangleMarker(
              "cm-text-selection",
              row.left - left,
              row.top - top,
              row.right - row.left,
              row.bottom - row.top,
            ),
          );
        }
        pos = line.to + 1;
      }
    }
  }
  return result;
}

export const textSelection = layer({
  above: false,
  class: "cm-text-selection-layer",
  markers,
  update: update => update.docChanged || update.selectionSet || update.viewportChanged || update.geometryChanged,
});
