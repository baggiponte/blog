import { invertedEffects } from "@codemirror/commands";
import {
  type EditorState,
  type Extension,
  Prec,
  type Range,
  StateEffect,
  StateField,
  type Transaction,
  type TransactionSpec,
} from "@codemirror/state";
import {
  Decoration,
  type DecorationSet,
  EditorView,
  keymap,
  showTooltip,
  type Tooltip,
  WidgetType,
} from "@codemirror/view";

import { changeOf, place, type Suggestion } from "./suggestion.ts";

// Proofreading suggestions in the editor: each one is shown on the words it quotes, struck through with the
// replacement after them, and is accepted, edited or rejected there. The state is a CodeMirror field, so it follows
// the text as it's typed and an accept is undone with ⌘Z like any edit.

// A suggestion with its place in the text as it is now.
export interface Placed {
  suggestion: Suggestion;
  from: number;
  to: number;
}

// The open post's suggestions not yet accepted or rejected. `lost` are the ones whose quoted words are no longer in
// the text (edited since Claude read it, or never there): the panel lists them so they can be read and dismissed.
export interface Review {
  placed: Placed[];
  lost: Suggestion[];
}

const empty: Review = { placed: [], lost: [] };

const mapPlaced = (placed: Placed, map: (pos: number, assoc: number) => number): Placed => ({
  ...placed,
  from: map(placed.from, 1),
  to: map(placed.to, -1),
});

// Replaces the suggestions with these, placed in the current text.
export const load = StateEffect.define<Suggestion[]>();
// Takes a suggestion off the list, accepted or rejected.
const resolve = StateEffect.define<string>();
// Puts one back where it was: what undoing an accept or a reject does.
const restore = StateEffect.define<Placed>({
  map: (placed, changes) => mapPlaced(placed, (pos, assoc) => changes.mapPos(pos, assoc)),
});

function loaded(state: EditorState, suggestions: Suggestion[]): Review {
  const text = state.doc.toString();
  const review: Review = { placed: [], lost: [] };
  for (const suggestion of suggestions) {
    const at = place(text, suggestion);
    if (at.found) {
      review.placed.push({ suggestion, from: at.from, to: at.to });
    } else {
      review.lost.push(suggestion);
    }
  }
  review.placed.sort((a, b) => a.from - b.from);
  return review;
}

// After an edit, a suggestion keeps its place while its quoted words are still there; typing inside them loses it.
function mapped(review: Review, tr: Transaction): Review {
  if (!tr.docChanged) {
    return review;
  }
  const placed: Placed[] = [];
  const lost = [...review.lost];
  for (const old of review.placed) {
    const next = mapPlaced(old, (pos, assoc) => tr.changes.mapPos(pos, assoc));
    if (tr.state.sliceDoc(next.from, next.to) === old.suggestion.quote) {
      placed.push(next);
    } else {
      lost.push(old.suggestion);
    }
  }
  return { placed, lost };
}

export const reviewField = StateField.define<Review>({
  create: () => empty,
  update(value, tr) {
    let review = mapped(value, tr);
    for (const effect of tr.effects) {
      if (effect.is(load)) {
        review = loaded(tr.state, effect.value);
      } else if (effect.is(resolve)) {
        review = {
          placed: review.placed.filter(p => p.suggestion.id !== effect.value),
          lost: review.lost.filter(s => s.id !== effect.value),
        };
      } else if (effect.is(restore)) {
        review = { ...review, placed: [...review.placed, effect.value].sort((a, b) => a.from - b.from) };
      }
    }
    return review;
  },
});

// Every suggestion still open, placed ones first in text order: what the editor saves back to suggestions.json.
export const remaining = (review: Review): Suggestion[] => [...review.placed.map(p => p.suggestion), ...review.lost];

// The placed suggestion the cursor is on, if any.
export function active(state: EditorState): Placed | undefined {
  const { head } = state.selection.main;
  return state.field(reviewField).placed.find(p => head >= p.from && head <= p.to);
}

// What the commands below act on: an EditorView, or in the tests a bare state that updates itself.
export interface Target {
  readonly state: EditorState;
  dispatch(spec: TransactionSpec): void;
}

// Accepting writes the replacement (or the edited text) over the quote and takes the suggestion off; rejecting only
// takes it off. Both are one transaction, so ⌘Z brings the text and the suggestion back together.
export function accept(view: Target, id: string, text?: string): boolean {
  const placed = view.state.field(reviewField).placed.find(p => p.suggestion.id === id);
  if (placed === undefined) {
    return false;
  }
  const insert = text ?? placed.suggestion.replacement;
  view.dispatch({
    changes: { from: placed.from, to: placed.to, insert },
    effects: resolve.of(id),
    selection: { anchor: placed.from + insert.length },
    userEvent: "input.suggestion",
  });
  return true;
}

export function reject(view: Target, id: string): boolean {
  view.dispatch({ effects: resolve.of(id), userEvent: "suggestion.reject" });
  return true;
}

// Moves the cursor to the next suggestion after it (or before it, backwards), wrapping around.
export function jump(view: Target, direction: 1 | -1): boolean {
  const { placed } = view.state.field(reviewField);
  const { head } = view.state.selection.main;
  const ordered = direction === 1 ? placed : placed.toReversed();
  const next = ordered.find(p => (direction === 1 ? p.from > head : p.to < head)) ?? ordered[0];
  if (next === undefined) {
    return false;
  }
  view.dispatch({ selection: { anchor: next.from }, scrollIntoView: true });
  return true;
}

export function goTo(view: EditorView, id: string): void {
  const placed = view.state.field(reviewField).placed.find(p => p.suggestion.id === id);
  if (placed !== undefined) {
    view.dispatch({ selection: { anchor: placed.from }, scrollIntoView: true });
    view.focus();
  }
}

// What undo puts back: the suggestion an accept or reject took off, where it was before; and what redo takes off
// again.
const undoable = invertedEffects.of(tr => {
  const before = tr.startState.field(reviewField, false);
  if (before === undefined) {
    return [];
  }
  return tr.effects.flatMap((effect): StateEffect<unknown>[] => {
    if (effect.is(restore)) {
      return [resolve.of(effect.value.suggestion.id)];
    }
    if (!effect.is(resolve)) {
      return [];
    }
    const placed = before.placed.find(p => p.suggestion.id === effect.value);
    return placed === undefined ? [] : [restore.of(placed)];
  });
});

// The replacement, shown after the struck-through quote.
export class ReplacementWidget extends WidgetType {
  readonly text: string;
  readonly id: string;
  readonly active: boolean;
  constructor(text: string, id: string, active: boolean) {
    super();
    this.text = text;
    this.id = id;
    this.active = active;
  }
  override eq(other: ReplacementWidget): boolean {
    return other.text === this.text && other.id === this.id && other.active === this.active;
  }
  toDOM(): HTMLElement {
    const span = document.createElement("ins");
    span.className = this.active ? "cm-sg-new cm-sg-active" : "cm-sg-new";
    span.dataset.suggestion = this.id;
    span.textContent = this.text;
    return span;
  }
  override ignoreEvent(): boolean {
    return false;
  }
}

// Each suggestion struck through where its words change (changeOf), with the new words right after; the rest of the
// quote reads as it is.
// New text that starts new paragraphs, as the blocks the editor draws for it: a heading or a paragraph each.
export type InsertedBlock = { kind: "heading"; level: number; text: string } | { kind: "paragraph"; text: string };

export function blocksOf(text: string): InsertedBlock[] {
  return text
    .split(/\n\s*\n/)
    .map(block => block.replace(/^\n+|\n+$/g, ""))
    .filter(block => block.trim() !== "")
    .map(block => {
      const heading = /^(#{1,6})\s+(.*)$/s.exec(block);
      return heading?.[1] !== undefined && heading[2] !== undefined
        ? { kind: "heading", level: heading[1].length, text: heading[2] }
        : { kind: "paragraph", text: block };
    });
}

// Inserted paragraphs, drawn below the line the insertion starts on, each styled as what it will be: a widget inside
// the line would take that line's style (a heading's size), and its blank lines would show as tall gaps.
export class InsertedBlocksWidget extends WidgetType {
  readonly text: string;
  readonly id: string;
  readonly active: boolean;
  constructor(text: string, id: string, active: boolean) {
    super();
    this.text = text;
    this.id = id;
    this.active = active;
  }
  override eq(other: InsertedBlocksWidget): boolean {
    return other.text === this.text && other.id === this.id && other.active === this.active;
  }
  toDOM(): HTMLElement {
    const dom = document.createElement("div");
    dom.className = this.active ? "cm-sg-blocks cm-sg-active" : "cm-sg-blocks";
    dom.dataset.suggestion = this.id;
    for (const block of blocksOf(this.text)) {
      const part = document.createElement("div");
      part.className = block.kind === "heading" ? `cm-sg-block cm-lp-h${block.level}` : "cm-sg-block";
      part.textContent = block.text;
      dom.append(part);
    }
    return dom;
  }
  override ignoreEvent(): boolean {
    return false;
  }
}

// Each suggestion struck through where its words change (changeOf), with the new words right after; the rest of the
// quote reads as it is. New words that run onto new lines go in a block below the line (InsertedBlocksWidget).
export function reviewDecorations(state: EditorState): DecorationSet {
  const current = active(state);
  return Decoration.set(
    state.field(reviewField).placed.flatMap(({ suggestion, from }) => {
      const on = current?.suggestion.id === suggestion.id;
      const { start, end, insert } = changeOf(suggestion.quote, suggestion.replacement);
      const ranges: Range<Decoration>[] = [];
      if (end > start) {
        ranges.push(
          Decoration.mark({
            class: `cm-sg-old cm-sg-${suggestion.kind}${on ? " cm-sg-active" : ""}`,
            attributes: { "data-suggestion": suggestion.id },
          }).range(from + start, from + end),
        );
      }
      const at = from + end;
      const newline = insert.indexOf("\n");
      const inline = newline === -1 ? insert : insert.slice(0, newline);
      if (inline !== "") {
        ranges.push(Decoration.widget({ widget: new ReplacementWidget(inline, suggestion.id, on), side: 1 }).range(at));
      }
      if (newline !== -1 && blocksOf(insert.slice(newline)).length > 0) {
        const widget = new InsertedBlocksWidget(insert.slice(newline), suggestion.id, on);
        ranges.push(Decoration.widget({ widget, side: 1, block: true }).range(state.doc.lineAt(at).to));
      }
      return ranges;
    }),
    true,
  );
}

// The card under the suggestion the cursor is on: why, and what to do with it.
function card(view: EditorView, { suggestion }: Placed): HTMLElement {
  const dom = document.createElement("div");
  dom.className = "cm-sg-card";
  const head = document.createElement("p");
  head.className = "cm-sg-kind";
  head.textContent = suggestion.kind;
  dom.append(head);
  if (suggestion.note !== undefined) {
    const note = document.createElement("p");
    note.className = "cm-sg-note";
    note.textContent = suggestion.note;
    dom.append(note);
  }
  const field = document.createElement("textarea");
  field.className = "cm-sg-edit";
  field.value = suggestion.replacement;
  field.rows = 2;
  field.hidden = true;
  field.setAttribute("aria-label", "replacement");
  const actions = document.createElement("div");
  actions.className = "cm-sg-actions";
  const button = (label: string, title: string, run: () => void): HTMLButtonElement => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    // mousedown, not click: a click would first move the editor's cursor off the suggestion and close the card.
    b.addEventListener("mousedown", event => {
      event.preventDefault();
      run();
    });
    return b;
  };
  const acceptButton = button("accept", "accept (⌘↵)", () => {
    accept(view, suggestion.id, field.hidden ? undefined : field.value);
    view.focus();
  });
  const editButton = button("edit", "change the replacement, then accept", () => {
    field.hidden = false;
    editButton.hidden = true;
    acceptButton.textContent = "accept edit";
    field.focus();
    // The words that change are selected, ready to be typed over; the rest of the quote stays as it is.
    const { start, insert } = changeOf(suggestion.quote, suggestion.replacement);
    field.setSelectionRange(start, start + insert.length);
  });
  const rejectButton = button("reject", "reject (⌘⇧↵)", () => {
    reject(view, suggestion.id);
    view.focus();
  });
  field.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      accept(view, suggestion.id, field.value);
      view.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      view.focus();
    }
  });
  actions.append(acceptButton, editButton, rejectButton);
  dom.append(field, actions);
  return dom;
}

const cardField = StateField.define<Tooltip | null>({
  create: () => null,
  update(tooltip, tr) {
    const current = active(tr.state);
    if (current === undefined) {
      return null;
    }
    const previous = tooltip as (Tooltip & { id?: string }) | null;
    if (previous?.id === current.suggestion.id && previous.pos === current.from) {
      return previous;
    }
    return {
      pos: current.from,
      end: current.to,
      above: false,
      id: current.suggestion.id,
      create: view => ({ dom: card(view, current) }),
    } as Tooltip;
  },
  provide: field => showTooltip.from(field),
});

const onActive = (run: (view: EditorView, placed: Placed) => boolean) => (view: EditorView) => {
  const current = active(view.state);
  return current !== undefined && run(view, current);
};

export function review(): Extension {
  return [
    reviewField,
    undoable,
    EditorView.decorations.compute([reviewField, "selection"], reviewDecorations),
    cardField,
    // Above the default keymap, where ⌘↵ inserts a blank line; off a suggestion these keys do what they did.
    Prec.high(
      keymap.of([
        { key: "Mod-Enter", run: onActive((view, p) => accept(view, p.suggestion.id)) },
        { key: "Mod-Shift-Enter", run: onActive((view, p) => reject(view, p.suggestion.id)) },
        { key: "Alt-]", run: view => jump(view, 1) },
        { key: "Alt-[", run: view => jump(view, -1) },
      ]),
    ),
    // Clicking the replacement puts the cursor on the suggestion, which opens its card.
    EditorView.domEventHandlers({
      mousedown: (event, view) => {
        const target = event.target instanceof HTMLElement ? event.target.closest(".cm-sg-new, .cm-sg-blocks") : null;
        if (!(target instanceof HTMLElement) || target.dataset.suggestion === undefined) {
          return false;
        }
        event.preventDefault();
        goTo(view, target.dataset.suggestion);
        return true;
      },
    }),
  ];
}
