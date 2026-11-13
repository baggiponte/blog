import { parseFrontmatter } from "@astrojs/markdown-remark";
import type { z } from "astro/zod";
import type { Paragraph, PhrasingContent, Root } from "mdast";
import { visit } from "unist-util-visit";

// A fenced block whose body is YAML fields, rendered as a card instead of code: ```player (src/lib/remark-player.ts)
// and ```link (src/lib/remark-link.ts). Each kind brings its language, its schema and how it renders; the parsing,
// the error messages with file and line, the editor's check and the remark plugin are the same for all of them.
// Where nothing renders the block (the editor, "copy page", GitHub) it reads as YAML.
export interface BlockKind<Fields> {
  lang: string;
  schema: z.ZodType<Fields>;
  render(fields: Fields): Paragraph;
}

// A kind of block with its fields type closed over, so blocks of different kinds can share a list: it turns the
// block's YAML into its card, or throws saying where the block is and what is wrong with it.
export interface Block {
  lang: string;
  card(yaml: string, where: string): Paragraph;
}

export const block = <Fields>(kind: BlockKind<Fields>): Block => ({
  lang: kind.lang,
  card: (yaml, where) => kind.render(parseBlock(kind, yaml, where)),
});

// The parts a card is built from: Markdown nodes that turn into the HTML element named by `hName`.
export interface Part {
  type: "cardPart";
  data: { hName: string; hProperties?: Record<string, string | string[]> };
  children: PhrasingContent[];
}
declare module "mdast" {
  interface PhrasingContentMap {
    cardPart: Part;
  }
}

export const part = (hName: string, className: string, children: PhrasingContent[] = [], aria = false): Part => ({
  type: "cardPart",
  data: { hName, hProperties: { className: [className], ...(aria ? { ariaHidden: "true" } : {}) } },
  children,
});
export const text = (hName: string, className: string, value: string): Part =>
  part(hName, className, [{ type: "text", value }]);

// The block's fields, or an error that says where the block starts (`index.md:12`) and what is wrong with it.
export function parseBlock<Fields>(kind: BlockKind<Fields>, yaml: string, where: string): Fields {
  let fields: unknown;
  try {
    fields = parseFrontmatter(`---\n${yaml}\n---\n`).frontmatter;
  } catch (error) {
    throw new Error(`${where}: a ${kind.lang} block holds YAML fields`, { cause: error });
  }
  const result = kind.schema.safeParse(fields);
  if (!result.success) {
    const issues = result.error.issues.map(issue => `${issue.path.join(".") || "block"}: ${issue.message}`);
    throw new Error(`${where}: ${issues.join("; ")}`);
  }
  return result.data;
}

// Every block in `lang` in a post's Markdown, with the line its fence opens on, for checks that don't render the
// post (the editor's save). Fences are found the way CommonMark opens them: up to three spaces, then ``` or ~~~.
export function fencedBlocks(markdown: string, lang: string): { yaml: string; line: number }[] {
  const blocks: { yaml: string; line: number }[] = [];
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const open = /^ {0,3}(`{3,}|~{3,})\s*([^\s{`]*)/.exec(lines[i] ?? "");
    if (open === null) {
      continue;
    }
    const fence = open[1] ?? "```";
    const end = lines.findIndex(
      (line, j) => j > i && line.trim().startsWith(fence) && line.trim().replaceAll(fence[0] ?? "`", "") === "",
    );
    const close = end === -1 ? lines.length : end;
    if (open[2] === lang) {
      blocks.push({ yaml: lines.slice(i + 1, close).join("\n"), line: i + 1 });
    }
    // Skipping the whole block, whatever its language, so a fence quoted inside another one isn't read as a block.
    i = close;
  }
  return blocks;
}

// Throws on the first block of any kind the build would refuse, with its line, without rendering the post.
export function checkBlocks(markdown: string, blocks: readonly Block[]): void {
  for (const kind of blocks) {
    for (const found of fencedBlocks(markdown, kind.lang)) {
      kind.card(found.yaml, `${kind.lang} block on line ${found.line}`);
    }
  }
}

// The file a remark plugin is handed; only its path is used, for error messages.
interface Source {
  path?: string;
}

// Replaces every block of these kinds with its card; a block the schema refuses stops the build.
export function remarkBlocks(blocks: readonly Block[]): () => (tree: Root, file: Source) => void {
  const byLang = new Map(blocks.map(kind => [kind.lang, kind]));
  return () => (tree, file) => {
    visit(tree, "code", (node, index, parent) => {
      const kind = byLang.get(node.lang ?? "");
      if (kind === undefined || parent === undefined || index === undefined) {
        return;
      }
      const where = `${file.path ?? "post"}:${node.position?.start.line ?? "?"}`;
      parent.children.splice(index, 1, kind.card(node.value, where));
    });
  };
}
