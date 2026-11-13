import { z } from "astro/zod";
import type { Paragraph, PhrasingContent } from "mdast";

import { type BlockKind, part, text } from "./fenced-blocks.ts";
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
// of rendering a broken card, and the editor refuses to save it (src/editor/post.ts).

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

export const playerBlock: BlockKind<Player> = {
  lang: "player",
  schema: playerSchema(z.string().min(1)),
  render: playerCard,
};
