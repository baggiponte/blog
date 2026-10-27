// Small DOM helpers shared by the page scripts.

// The element matching selector, which the page always renders: fail loudly if a template change drops it.
export function find<T extends Element = HTMLElement>(selector: string, root: ParentNode = document): T {
  const element = root.querySelector<T>(selector);
  if (element === null) {
    throw new Error(`no ${selector} on this page`);
  }
  return element;
}

// document.createElement with its properties and children in one call. Text children are set as text, never as
// HTML, so data from a post can't inject markup.
export function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  properties: Partial<HTMLElementTagNameMap[K]> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), properties);
  node.append(...children);
  return node;
}

// The element an event happened on, or null when it happened on the document or a text node.
export const targetOf = (event: Event): Element | null => (event.target instanceof Element ? event.target : null);
