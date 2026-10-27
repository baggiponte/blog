import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// Every post is a bundle: content/blog/<slug>/index.md holds the metadata (and the body,
// unless the post is a marimo notebook, whose generated <name>.md is named in `notebook`).
const blog = defineCollection({
  loader: glob({
    pattern: "*/index.md",
    base: "./content/blog",
    generateId: ({ entry }) => entry.split("/")[0],
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    format: z.enum(["post", "tidbits"]).default("post"),
    notebook: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

// Markdown exported by `just marimo-export`; never edited by hand.
const notebooks = defineCollection({
  loader: glob({ pattern: ["*/*.md", "!*/index.md"], base: "./content/blog" }),
  schema: z.looseObject({}),
});

const pages = defineCollection({
  loader: glob({ pattern: "*.md", base: "./content" }),
  schema: z.object({ title: z.string() }),
});

export const collections = { blog, notebooks, pages };
