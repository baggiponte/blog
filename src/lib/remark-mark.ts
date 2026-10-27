import type { Parent, PhrasingContent, Root, Text } from "mdast";
import { visit } from "unist-util-visit";

// `==text==` becomes <mark>text</mark>; marks cannot span other inline nodes (links, code).
// A node whose `data.hName` is set turns into that HTML element, so the text is escaped like any other.
export interface Mark extends Parent {
  type: "mark";
  data: { hName: "mark" };
  children: [Text];
}
declare module "mdast" {
  interface PhrasingContentMap {
    mark: Mark;
  }
}

const markPattern = /==(?=\S)(.+?)(?<=\S)==/g;

export function remarkMark(): (tree: Root) => void {
  return tree => {
    visit(tree, "text", (node, index, parent) => {
      if (parent === undefined || index === undefined) {
        return;
      }
      const parts: PhrasingContent[] = [];
      let last = 0;
      for (const match of node.value.matchAll(markPattern)) {
        const [whole, text = ""] = match;
        if (match.index > last) {
          parts.push({ type: "text", value: node.value.slice(last, match.index) });
        }
        parts.push({ type: "mark", data: { hName: "mark" }, children: [{ type: "text", value: text }] });
        last = match.index + whole.length;
      }
      if (parts.length === 0) {
        return;
      }
      if (last < node.value.length) {
        parts.push({ type: "text", value: node.value.slice(last) });
      }
      parent.children.splice(index, 1, ...parts);
      // Skip the new nodes: the marks' text must not be searched again.
      return index + parts.length;
    });
  };
}
