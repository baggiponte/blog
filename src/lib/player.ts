import { z } from "astro/zod";

import { listenLink } from "./listen.ts";

// A player card: a song or a podcast episode, which opens on a listening (or watching) platform. The same fields fill
// a ```player block anywhere in a post (src/lib/remark-player.ts) and the `song` a post ends with (frontmatter).
// Only the link is required: without a title the card says what it is, and without a cover it draws a blank sleeve
// (music) or a blank episode (podcast). A cover is an image next to index.md, so a reader's browser asks the
// platform for nothing until they click.
export const playerTypes = ["music", "podcast"] as const;
export type PlayerType = (typeof playerTypes)[number];

const defaultTitle = (type: PlayerType, verb: string): string =>
  `${type === "music" ? "a song" : "an episode"} worth a ${verb}`;

// `cover` is Astro's image() in the frontmatter, and a plain path in a block, which the remark plugin turns into a
// Markdown image so Astro resizes it all the same.
export const playerSchema = <Cover extends z.ZodType>(cover: Cover) =>
  z
    .strictObject({
      type: z.enum(playerTypes).default("music"),
      title: z.string().min(1).optional(),
      // Why it's here, or where to start listening.
      note: z.string().min(1).optional(),
      url: z.string(),
      cover: cover.optional(),
    })
    .transform((player, context) => {
      const link = listenLink(player.url);
      if (link === undefined) {
        context.addIssue({ code: "custom", path: ["url"], message: "a Spotify or YouTube link" });
        return z.NEVER;
      }
      return { ...player, ...link, title: player.title ?? defaultTitle(player.type, link.verb) };
    });

export type Player = z.output<ReturnType<typeof playerSchema<z.ZodString>>>;
