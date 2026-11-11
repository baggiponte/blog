// A tagged template for building HTML on the server: html`<p>${text}</p>` escapes every interpolated value, except
// markup that was itself built with html``, and joins arrays. The Html class marks the markup that is already safe.

export class Html {
  readonly markup: string;
  constructor(markup: string) {
    this.markup = markup;
  }
}

type Value = Html | string | number | null | undefined | Value[];

const escapes: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const escapeHtml = (text: string | number): string => String(text).replace(/[&<>"]/g, c => escapes[c] ?? c);

function render(value: Value): string {
  if (value instanceof Html) {
    return value.markup;
  }
  if (Array.isArray(value)) {
    return value.map(render).join("");
  }
  return value === null || value === undefined ? "" : escapeHtml(value);
}

export const html = (strings: TemplateStringsArray, ...values: Value[]): Html =>
  new Html(String.raw({ raw: strings }, ...values.map(render)));

// Markup pieces one after the other, with separator between them.
export const join = (parts: readonly Html[], separator: string): Html =>
  new Html(parts.map(part => part.markup).join(separator));
