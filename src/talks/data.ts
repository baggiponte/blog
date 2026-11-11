// The shape of the content/talks/*.json data files, checked on every build. catalog.ts reads them and checks them
// against each other.
import { z } from "zod";

import { slugify } from "../lib/slugs.ts";

// Where an edition's slides are: a Slidev deck in decks/, served at /talks/<slug>/, or anywhere else on the web.
// talks.json writes either as a string; parsing tells them apart once, so nothing downstream slices paths.
export type Slides = { kind: "deck"; slug: string } | { kind: "external"; url: string };

const deckPath = /^\/talks\/([\w-]+)\/$/;
const Slides = z.union([
  z
    .string()
    .regex(deckPath, "a /talks/<slug>/ deck path")
    .transform((path): Slides => ({ kind: "deck", slug: path.slice("/talks/".length, -1) })),
  z.url().transform((url): Slides => ({ kind: "external", url })),
]);

const Edition = z.strictObject({
  date: z.iso.date(),
  venue: z.string().min(1),
  // Kept for the record when an edition was named differently; not shown.
  title: z.string().optional(),
  slides: Slides.optional(),
  recording: z.url().optional(),
  source: z.url().optional(),
});

const Talk = z.strictObject({
  title: z.string().min(1),
  description: z.string().min(1),
  // A hands-on workshop rather than a talk: tagged as such in the talks view.
  workshop: z.literal(true).optional(),
  // Non-empty, so every talk has a newest edition.
  given: z.tuple([Edition], Edition),
});

export const Talks = z
  .array(Talk)
  .min(1)
  .superRefine((talks, ctx) => {
    const seen = new Set<string>();
    for (const [index, { title }] of talks.entries()) {
      if (seen.has(slugify(title))) {
        ctx.addIssue({ code: "custom", path: [index, "title"], message: `two talks share the id ${slugify(title)}` });
      }
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

export type Venues = z.infer<typeof Venues>;
export type Cities = z.infer<typeof Cities>;
