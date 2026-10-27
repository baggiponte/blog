import { defineCollection } from "astro:content";
import { dirname } from "node:path";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// Every post is a bundle: content/blog/<slug>/index.md holds the metadata (and the body,
// unless the post is a marimo notebook, whose generated <name>.md is named in `notebook`).
const blog = defineCollection({
  loader: glob({
    pattern: "*/index.md",
    base: "./content/blog",
    // "01-groundup-linear-regression/index.md" has the id "01-groundup-linear-regression".
    generateId: ({ entry }) => dirname(entry),
  }),
  schema: z.object({
    // Titles are lowercase, like the rest of the site (AGENTS.md).
    title: z
      .string()
      .min(1)
      .refine(title => title === title.toLowerCase(), "titles are lowercase"),
    description: z.string().min(1),
    date: z.coerce.date(),
    format: z.enum(["post", "tidbits"]).default("post"),
    // The export next to index.md, as `just notebook` writes it: "<name>.md".
    notebook: z
      .string()
      .regex(/^[\w-]+\.md$/, "a <name>.md file next to index.md")
      .optional(),
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
