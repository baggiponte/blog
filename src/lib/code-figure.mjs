// Shiki transformer: wraps each highlighted block in <figure class="code"> with a caption.
// The caption is the marimo cell name (```python {.marimo name="setup"}) or the language;
// copy and wrap buttons are added client side so they never render dead without JS.
export function codeFigure() {
  return {
    name: "code-figure",
    root(root) {
      const meta = this.options.meta?.__raw ?? "";
      const name = meta.match(/name="([^"]+)"/)?.[1];
      const caption = name ?? this.options.lang ?? "text";
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
              children: [{ type: "element", tagName: "span", properties: {}, children: [{ type: "text", value: caption }] }],
            },
            ...root.children,
          ],
        },
      ];
    },
  };
}
