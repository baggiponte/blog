// The shape of the content/talks/*.json data files, checked on every build.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";

const Edition = z.strictObject({
  date: z.iso.date(),
  venue: z.string().min(1),
  // Kept for the record when an edition was named differently; not shown.
  title: z.string().optional(),
  // "/talks/<slug>/" for a deck in decks/, or any external URL.
  slides: z.union([z.string().regex(/^\/talks\/[\w-]+\/$/, "a /talks/<slug>/ deck path"), z.url()]).optional(),
  recording: z.url().optional(),
  source: z.url().optional(),
});

const Talk = z.strictObject({
  title: z.string().min(1),
  description: z.string().min(1),
  // A hands-on workshop rather than a talk: tagged as such in the talks view.
  workshop: z.literal(true).optional(),
  // Non-empty, so given[0] (the newest edition, once sorted) always exists.
  given: z.tuple([Edition], Edition),
});

export const Talks = z
  .array(Talk)
  .min(1)
  .superRefine((talks, ctx) => {
    const seen = new Set<string>();
    for (const [index, { title }] of talks.entries()) {
      if (seen.has(slugify(title)))
        ctx.addIssue({ code: "custom", path: [index, "title"], message: `two talks share the id ${slugify(title)}` });
      seen.add(slugify(title));
    }
  });

// Keyed by venue name prefix ("PyData" covers London, Global, Amsterdam). Each field comes from the longest prefix that sets it,
// so "PyData London 2026" takes its logo from "PyData" and its city from "PyData London".
export const Venues = z.record(
  z.string(),
  z.strictObject({
    // A key of cities.json, or "online".
    city: z.string().optional(),
    logo: z
      .string()
      .regex(/^[\w-]+\.(png|svg)$/, "a file in public/talks/logos/")
      .optional(),
    source: z.url().optional(),
    note: z.string().optional(),
  }),
);

export const Cities = z.record(
  z.string(),
  z.strictObject({
    at: z.tuple([z.number().min(-180).max(180), z.number().min(-90).max(90)]),
    label: z.enum(["left", "right"]).optional(),
  }),
);

export type Talk = z.infer<typeof Talk>;
export type Edition = z.infer<typeof Edition>;
export type Venues = z.infer<typeof Venues>;
export type Cities = z.infer<typeof Cities>;

// A talk's id in the page (data-talk) and in tests: stable when talks are added or reordered.
export const slugify = (title: string) =>
  title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

export async function load<T>(root: string, file: string, schema: z.ZodType<T>): Promise<T> {
  const parsed = schema.safeParse(JSON.parse(await readFile(resolve(root, "content/talks", file), "utf8")));
  if (!parsed.success) throw new Error(`content/talks/${file}:\n${z.prettifyError(parsed.error)}`);
  return parsed.data;
}
