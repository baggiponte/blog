import type { ShikiTransformer } from "@shikijs/types";

// Shiki transformer: wraps each highlighted block in <figure class="code"> with a caption.
// The caption is the marimo cell name (```python {.marimo name="setup"}) or the language;
// copy and wrap buttons are added client side so they never render dead without JS.
export function codeFigure(): ShikiTransformer {
  return {
    name: "code-figure",
    root(root) {
      const meta = this.options.meta?.__raw ?? "";
      // Astro hands a fence without a language to Shiki as "plaintext".
      const { lang = "plaintext" } = this.options;
      const caption = /name="([^"]+)"/.exec(meta)?.[1] ?? (lang === "plaintext" ? "text" : lang);
      root.children = [
        {
          type: "element",
          tagName: "figure",
          properties: { className: ["code"] },
          children: [
            {
              type: "element",
              tagName: "figcaption",
              properties: {},
              children: [
                { type: "element", tagName: "span", properties: {}, children: [{ type: "text", value: caption }] },
              ],
            },
            ...root.children.filter(child => child.type !== "doctype"),
          ],
        },
      ];
    },
  };
}
