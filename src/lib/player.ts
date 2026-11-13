import { z } from "astro/zod";

import { listenLink } from "./listen.ts";

// A player card: the cover of a song or a podcast episode, which opens it on a listening platform. The same fields
// fill a ```player block anywhere in a post (src/lib/remark-player.ts) and the `song` a post ends with (frontmatter).
// The cover is an image next to index.md, so a reader's browser asks the platform for nothing until they click.
export const playerTypes = ["music", "podcast"] as const;
export type PlayerType = (typeof playerTypes)[number];

// `cover` is Astro's image() in the frontmatter, and a plain path in a block, which the remark plugin turns into a
// Markdown image so Astro resizes it all the same.
export const playerSchema = <Cover extends z.ZodType>(cover: Cover) =>
  z
    .strictObject({
      type: z.enum(playerTypes).default("music"),
      title: z.string().min(1),
      // The artist, or the podcast's name.
      by: z.string().min(1),
      // An optional line under it: why it's here, or where to start listening.
      note: z.string().min(1).optional(),
      url: z.string(),
      cover,
    })
    .transform((player, context) => {
      const link = listenLink(player.url);
      if (link === undefined) {
        context.addIssue({ code: "custom", path: ["url"], message: "a Spotify track, album, episode or show link" });
        return z.NEVER;
      }
      return { ...player, ...link };
    });

export type Player = z.output<ReturnType<typeof playerSchema<z.ZodString>>>;
