import { parseFrontmatter } from "@astrojs/markdown-remark";
import { z } from "astro/zod";
import type { Paragraph, PhrasingContent, Root } from "mdast";
import { visit } from "unist-util-visit";

import { type Player, playerSchema } from "./player.ts";

// A ```player block becomes a player card, anywhere in a post and as often as you like:
//
//   ```player
//   type: podcast                                      # or music, the default
//   url: https://open.spotify.com/episode/…            # the only field required
//   title: how we use agents
//   note: the part on shared sessions, from minute 32
//   cover: ./episode.jpg
//   ```
//
// The fields are the frontmatter `song`'s (src/lib/player.ts), checked the same way: a typo stops the build instead
// of rendering a broken card, and the editor refuses to save it (src/editor/post.ts). Where nothing renders the block
// (the editor, "copy page", GitHub) it reads as YAML.
export interface Part {
  type: "playerPart";
  data: { hName: string; hProperties?: Record<string, string | string[]> };
  children: PhrasingContent[];
}
declare module "mdast" {
  interface PhrasingContentMap {
    playerPart: Part;
  }
}

// The file a remark plugin is handed; only its path is used, for error messages.
interface Source {
  path?: string;
}

const blockSchema = playerSchema(z.string().min(1));

const part = (hName: string, className: string, children: PhrasingContent[] = [], aria = false): Part => ({
  type: "playerPart",
  data: { hName, hProperties: { className: [className], ...(aria ? { ariaHidden: "true" } : {}) } },
  children,
});
const text = (hName: string, className: string, value: string): Part =>
  part(hName, className, [{ type: "text", value }]);

// The block's fields, or an error that says where the block starts (`index.md:12`) and what is wrong with it.
export function parsePlayer(yaml: string, where: string): Player {
  let fields: unknown;
  try {
    fields = parseFrontmatter(`---\n${yaml}\n---\n`).frontmatter;
  } catch (error) {
    throw new Error(`${where}: a player block holds YAML fields`, { cause: error });
  }
  const result = blockSchema.safeParse(fields);
  if (!result.success) {
    const issues = result.error.issues.map(issue => `${issue.path.join(".") || "block"}: ${issue.message}`);
    throw new Error(`${where}: ${issues.join("; ")}`);
  }
  return result.data;
}

// Every ```player block in a post's Markdown, with the line its fence opens on, for checks that don't render the
// post (the editor's save). Fences are found the way CommonMark opens them: up to three spaces, then ``` or ~~~.
export function playerBlocks(markdown: string): { yaml: string; line: number }[] {
  const blocks: { yaml: string; line: number }[] = [];
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const open = /^ {0,3}(`{3,}|~{3,})\s*player\s*$/.exec(lines[i] ?? "");
    if (open === null) {
      continue;
    }
    const fence = open[1] ?? "```";
    const end = lines.findIndex(
      (line, j) => j > i && line.trim().startsWith(fence) && line.trim().replaceAll(fence[0] ?? "`", "") === "",
    );
    const close = end === -1 ? lines.length : end;
    blocks.push({ yaml: lines.slice(i + 1, close).join("\n"), line: i + 1 });
    i = close;
  }
  return blocks;
}

// The card as Markdown nodes that turn into HTML elements; the cover stays a Markdown image, collected and resized
// by Astro after this plugin runs.
export function playerCard(player: Player): Paragraph {
  // 2× the largest sleeve, so Astro resizes the cover instead of shipping it at full size.
  const cover: PhrasingContent =
    player.cover === undefined
      ? part("span", "blank", [], true)
      : { type: "image", url: player.cover, alt: "", data: { hProperties: { width: 256, height: 256 } } };
  const sleeve = player.type === "music" ? [part("span", "vinyl", [], true), cover] : [cover];
  return {
    type: "paragraph",
    data: {
      hName: "div",
      hProperties: { className: ["player"], dataType: player.type, dataPlatform: player.platform },
    },
    children: [
      {
        type: "link",
        url: player.url,
        data: { hProperties: { className: ["player-card"], target: "_blank", rel: "noopener" } },
        children: [
          part("span", "sleeve", sleeve),
          part("span", "player-t", [
            text("span", "player-k", player.type),
            text("b", "player-title", player.title),
            ...(player.note === undefined ? [] : [text("span", "player-note", player.note)]),
            text("span", "player-go", `${player.verb} on ${player.platform} ↗`),
          ]),
        ],
      },
    ],
  };
}

export function remarkPlayer(): (tree: Root, file: Source) => void {
  return (tree, file) => {
    visit(tree, "code", (node, index, parent) => {
      if (node.lang !== "player" || parent === undefined || index === undefined) {
        return;
      }
      parent.children.splice(
        index,
        1,
        playerCard(parsePlayer(node.value, `${file.path ?? "post"}:${node.position?.start.line ?? "?"}`)),
      );
    });
  };
}
