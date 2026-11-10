import type { Paragraph, PhrasingContent, Root } from "mdast";
import { visit } from "unist-util-visit";

// Every line is its own paragraph: posts are written one paragraph per line, without blank lines between them,
// so a single newline inside a paragraph starts a new one instead of continuing it. Newlines inside inline
// nodes (emphasis or a link that spans lines) and hard breaks stay as they are.
export function remarkLineParagraphs(): (tree: Root) => void {
  return tree => {
    visit(tree, "paragraph", (node, index, parent) => {
      if (parent === undefined || index === undefined) {
        return;
      }
      const lines: PhrasingContent[][] = [[]];
      for (const child of node.children) {
        if (child.type !== "text" || !child.value.includes("\n")) {
          lines.at(-1)?.push(child);
          continue;
        }
        child.value.split("\n").forEach((part, n) => {
          if (n > 0) {
            lines.push([]);
          }
          if (part !== "") {
            lines.at(-1)?.push({ type: "text", value: part });
          }
        });
      }
      const paragraphs: Paragraph[] = lines
        .filter(children => children.length > 0)
        .map(children => ({ type: "paragraph", children }));
      if (paragraphs.length <= 1) {
        return;
      }
      parent.children.splice(index, 1, ...paragraphs);
      return index + paragraphs.length;
    });
  };
}
