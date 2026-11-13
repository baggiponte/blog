import { parseFrontmatter } from "@astrojs/markdown-remark";
import { z } from "astro/zod";
import type { Paragraph, PhrasingContent, Root } from "mdast";
import { visit } from "unist-util-visit";

import { type Card, cardSchema, imageSize, type LinkCard, type MediaCard } from "./card.ts";

// A ```card block becomes a card, anywhere in a post and as often as you like; cards in a row read as a list:
//
//   ```card
//   url: https://jvns.ca/blog/2024/02/16/popular-git-config-options/   # the only field required
//   type: link                       # or song, podcast, video; the link decides when left out
//   title: popular git config options
//   note: the options people actually set
//   image: ./cover.jpg               # songs, podcasts and videos: a file next to index.md
//   site: julia evans                # links only: the host when left out
//   ```
//
// The fields are the frontmatter `song`'s (src/lib/card.ts), checked the same way: a typo stops the build instead
// of rendering a broken card, and the editor refuses to save it (src/editor/post.ts). Where nothing renders the block
// (the editor, "copy page", GitHub) it reads as YAML.
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

// The file a remark plugin is handed; only its path is used, for error messages.
interface Source {
  path?: string;
}

// A picture on the other site would be fetched by every reader's browser, so a block's image is a file.
const blockSchema = cardSchema(
  z
    .string()
    .min(1)
    .refine(path => !/^[a-z][a-z\d+.-]*:/i.test(path), "a file next to index.md, not a link"),
);

const part = (hName: string, className: string, children: PhrasingContent[] = [], aria = false): Part => ({
  type: "cardPart",
  data: { hName, hProperties: { className: [className], ...(aria ? { ariaHidden: "true" } : {}) } },
  children,
});
const text = (hName: string, className: string, value: string): Part =>
  part(hName, className, [{ type: "text", value }]);

// The block's fields, or an error that says where the block starts (`index.md:12`) and what is wrong with it.
export function parseCard(yaml: string, where: string): Card {
  let fields: unknown;
  try {
    fields = parseFrontmatter(`---\n${yaml}\n---\n`).frontmatter;
  } catch (error) {
    throw new Error(`${where}: a card block holds YAML fields`, { cause: error });
  }
  const result = blockSchema.safeParse(fields);
  if (!result.success) {
    const issues = result.error.issues.map(issue => `${issue.path.join(".") || "block"}: ${issue.message}`);
    throw new Error(`${where}: ${issues.join("; ")}`);
  }
  return result.data;
}

// Every ```card block in a post's Markdown, with the line its fence opens on, for checks that don't render the
// post (the editor's save). Fences are found the way CommonMark opens them: up to three spaces, then ``` or ~~~, and
// every fenced block is skipped whole, so a card block quoted inside another fence is an example, not a card.
export function cardBlocks(markdown: string): { yaml: string; line: number }[] {
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
    if (open[2] === "card") {
      blocks.push({ yaml: lines.slice(i + 1, close).join("\n"), line: i + 1 });
    }
    i = close;
  }
  return blocks;
}

// The image stays a Markdown image, collected and resized by Astro after this plugin runs, at twice its size on
// the page.
function image(card: MediaCard): PhrasingContent | undefined {
  if (card.image === undefined) {
    return undefined;
  }
  const { width, height } = imageSize(card.type);
  return { type: "image", url: card.image, alt: "", data: { hProperties: { width: width * 2, height: height * 2 } } };
}

const outbound = (url: string, className: string, children: PhrasingContent[]): PhrasingContent => ({
  type: "link",
  url,
  data: { hProperties: { className: [className], target: "_blank", rel: "noopener" } },
  children,
});

// A song, podcast or video: the sleeve (with its record, for a song) and what to do with it.
function mediaCard(card: MediaCard): Paragraph {
  const cover = image(card) ?? part("span", "blank", [], true);
  const sleeve = card.type === "song" ? [part("span", "vinyl", [], true), cover] : [cover];
  return {
    type: "paragraph",
    data: { hName: "div", hProperties: { className: ["player"], dataType: card.type, dataPlatform: card.platform } },
    children: [
      outbound(card.url, "player-card", [
        part("span", "sleeve", sleeve),
        part("span", "player-t", [
          text("span", "player-k", card.type),
          text("b", "player-title", card.title),
          ...(card.note === undefined ? [] : [text("span", "player-note", card.note)]),
          text("span", "player-go", `${card.verb} on ${card.platform} ↗`),
        ]),
      ]),
    ],
  };
}

// A page to read: a text bookmark with its title, a line about it and who published it.
function linkCard(card: LinkCard): Paragraph {
  return {
    type: "paragraph",
    data: { hName: "div", hProperties: { className: ["link"] } },
    children: [
      outbound(card.url, "link-card", [
        part("span", "link-t", [
          text("b", "link-title", card.title),
          ...(card.note === undefined ? [] : [text("span", "link-desc", card.note)]),
          part("span", "link-site", [
            text("span", "link-mark", card.site.slice(0, 1)),
            text("span", "link-name", card.site),
            ...(card.site === card.host ? [] : [text("span", "link-host", card.host)]),
            { type: "text", value: " ↗" },
          ]),
        ]),
      ]),
    ],
  };
}

export const cardNode = (card: Card): Paragraph => (card.type === "link" ? linkCard(card) : mediaCard(card));

export function remarkCard(): (tree: Root, file: Source) => void {
  return (tree, file) => {
    visit(tree, "code", (node, index, parent) => {
      if (node.lang !== "card" || parent === undefined || index === undefined) {
        return;
      }
      const where = `${file.path ?? "post"}:${node.position?.start.line ?? "?"}`;
      parent.children.splice(index, 1, cardNode(parseCard(node.value, where)));
    });
  };
}
