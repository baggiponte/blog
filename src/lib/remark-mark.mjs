import { visit } from "unist-util-visit";

// `==text==` becomes <mark>text</mark>; marks cannot span other inline nodes (links, code).
const MARK = /==(?=\S)(.+?)(?<=\S)==/g;
const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function remarkMark() {
  return (tree) => {
    visit(tree, "text", (node, index, parent) => {
      if (!parent || index === undefined || !MARK.test(node.value)) return;
      MARK.lastIndex = 0;
      const parts = [];
      let last = 0;
      for (const m of node.value.matchAll(MARK)) {
        if (m.index > last) parts.push({ type: "text", value: node.value.slice(last, m.index) });
        parts.push({ type: "html", value: `<mark>${escape(m[1])}</mark>` });
        last = m.index + m[0].length;
      }
      if (last < node.value.length) parts.push({ type: "text", value: node.value.slice(last) });
      parent.children.splice(index, 1, ...parts);
      return index + parts.length;
    });
  };
}
