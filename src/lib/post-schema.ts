import { z } from "astro/zod";

import { spotifyLink } from "./spotify.ts";

// A post's index.md frontmatter. Drafts only show in `astro dev`, so the rules readers would notice (a lowercase
// title, a description) apply once a post is published: a half-written draft never stops the dev server.
// Plain posts and short tidbits; the editor's "new post" form offers the same two.
export const formatSchema = z.enum(["post", "tidbits"]);

// The song (or podcast episode) a post ends with: the card shows the cover and opens it on Spotify. The cover is an
// image next to index.md, so a reader's browser never asks Spotify for anything until they click.
const songSchema = <Cover extends z.ZodType>(cover: Cover) =>
  z
    .object({ title: z.string().min(1), artist: z.string().min(1), url: z.string(), cover })
    .transform((song, context) => {
      const link = spotifyLink(song.url);
      if (link === undefined) {
        context.addIssue({ code: "custom", path: ["url"], message: "a Spotify track, album, episode or show link" });
        return z.NEVER;
      }
      return { ...song, ...link };
    });

// `cover` is Astro's image() in the content collection, which resolves "cover.jpg" to an image it can resize; the
// editor checks frontmatter without Astro, where the path is just a string.
export const makePostSchema = <Cover extends z.ZodType>(cover: Cover) =>
  z
    .object({
      title: z.string().min(1),
      description: z.string(),
      date: z.coerce.date(),
      format: formatSchema.default("post"),
      // The export next to index.md, as `just notebook` writes it: "<name>.md".
      notebook: z
        .string()
        .regex(/^[\w-]+\.md$/, "a <name>.md file next to index.md")
        .optional(),
      draft: z.boolean().default(false),
      song: songSchema(cover).optional(),
    })
    .superRefine((post, context) => {
      if (post.draft) {
        return;
      }
      // Titles are lowercase, like the rest of the site (AGENTS.md).
      if (post.title !== post.title.toLowerCase()) {
        context.addIssue({ code: "custom", path: ["title"], message: "titles are lowercase (drafts may wait)" });
      }
      if (post.description.trim() === "") {
        context.addIssue({ code: "custom", path: ["description"], message: "a published post needs a description" });
      }
    });

export const postSchema = makePostSchema(z.string().min(1));
