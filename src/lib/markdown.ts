import { unified } from "@astrojs/markdown-remark";
import type { AstroUserConfig } from "astro";

import { codeFigure } from "./code-figure.ts";
import { codeThemes } from "./code-themes.ts";
import { remarkLineParagraphs } from "./remark-line-paragraphs.ts";
import { remarkMark } from "./remark-mark.ts";

// How posts are rendered: astro.config.ts uses it, and the tests render Markdown with it.
export const markdown = {
  // unified() rather than the default Sätteri so `==highlights==` and one paragraph per line can be small remark
  // plugins.
  processor: unified({ remarkPlugins: [remarkLineParagraphs, remarkMark] }),
  shikiConfig: {
    themes: codeThemes,
    defaultColor: false,
    transformers: [codeFigure()],
  },
} satisfies AstroUserConfig["markdown"];

// Plain Markdown for "copy page", "view as markdown" and search: marimo cell attributes
// (```python {.marimo name="setup"}) and cell separators (<!---->) are noise outside the notebook.
export function plainMarkdown(body: string): string {
  return body
    .replace(/^(```\w*)\s*\{[^}\n]*\}/gm, "$1")
    .replace(/^<!---->$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
