import { z } from "astro/zod";

// A post's index.md frontmatter. Drafts only show in `astro dev`, so the rules readers would notice (a lowercase
// title, a description) apply once a post is published: a half-written draft never stops the dev server.
export const postSchema = z
  .object({
    title: z.string().min(1),
    description: z.string(),
    date: z.coerce.date(),
    format: z.enum(["post", "tidbits"]).default("post"),
    // The export next to index.md, as `just notebook` writes it: "<name>.md".
    notebook: z
      .string()
      .regex(/^[\w-]+\.md$/, "a <name>.md file next to index.md")
      .optional(),
    draft: z.boolean().default(false),
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
