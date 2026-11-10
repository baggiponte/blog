import { EditorSelection, type StateCommand } from "@codemirror/state";

// Inline formatting from the keyboard: ⌘B bold, ⌘I italic, ⌘U highlight (the site's `==mark==`), ⌘⇧X strikethrough.
// Each one wraps the selection in its marker, or takes the marker off when the selection already has it, inside
// or just outside the selected text.

export type Marker = "**" | "*" | "==" | "~~";

export interface Toggle {
  changes: { from: number; to: number; insert: string }[];
  // The selection afterwards, in the changed document.
  from: number;
  to: number;
}

// Whether a run of `count` marker characters carries `marker`: `***` is both bold and italic.
function carries(marker: Marker, count: number): boolean {
  return count === marker.length || (marker[0] === "*" && count === 3);
}

function runBefore(text: string, position: number, char: string): number {
  let n = 0;
  while (text[position - n - 1] === char) {
    n++;
  }
  return n;
}

function runAfter(text: string, position: number, char: string): number {
  let n = 0;
  while (text[position + n] === char) {
    n++;
  }
  return n;
}

export function toggleMarker(doc: string, from: number, to: number, marker: Marker): Toggle {
  const n = marker.length;
  const char = marker[0] ?? "";
  // A double-clicked word often takes its space along, and `**word **` isn't bold.
  let start = from;
  let end = to;
  while (start < end && /\s/.test(doc[start] ?? "")) {
    start++;
  }
  while (end > start && /\s/.test(doc[end - 1] ?? "")) {
    end--;
  }

  // The markers are just outside the selection: `**|word|**`.
  if (carries(marker, runBefore(doc, start, char)) && carries(marker, runAfter(doc, end, char))) {
    return {
      changes: [
        { from: start - n, to: start, insert: "" },
        { from: end, to: end + n, insert: "" },
      ],
      from: start - n,
      to: end - n,
    };
  }
  // The selection takes the markers in: `|**word**|`.
  const inner = doc.slice(start, end);
  const lead = runAfter(inner, 0, char);
  const trail = runBefore(inner, inner.length, char);
  if (inner.length > 2 * n && lead < inner.length && carries(marker, lead) && carries(marker, trail)) {
    return {
      changes: [
        { from: start, to: start + n, insert: "" },
        { from: end - n, to: end, insert: "" },
      ],
      from: start,
      to: end - 2 * n,
    };
  }
  return {
    changes: [
      { from: start, to: start, insert: marker },
      { from: end, to: end, insert: marker },
    ],
    from: start + n,
    to: end + n,
  };
}

export function toggle(marker: Marker): StateCommand {
  return ({ state, dispatch }) => {
    const doc = state.doc.toString();
    dispatch(
      state.update(
        state.changeByRange(range => {
          const { changes, from, to } = toggleMarker(doc, range.from, range.to, marker);
          return { changes, range: EditorSelection.range(from, to) };
        }),
        { scrollIntoView: true, userEvent: "input.format" },
      ),
    );
    return true;
  };
}

export const formatKeys = [
  { key: "Mod-b", run: toggle("**"), preventDefault: true },
  { key: "Mod-i", run: toggle("*"), preventDefault: true },
  { key: "Mod-u", run: toggle("=="), preventDefault: true },
  { key: "Mod-Shift-x", run: toggle("~~"), preventDefault: true },
];
