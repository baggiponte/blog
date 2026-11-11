// The talks catalog as the page shows it: content/talks/*.json, each checked against its schema (data.ts) and against
// the others and decks/, then resolved, so the views (views.ts) get plain data and never look anything up.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";

import { slugify } from "../lib/slugs.ts";
import { Cities, type Slides, Talks, Venues } from "./data.ts";

// A venue with the city and logo its name resolves to in venues.json.
export interface Venue {
  name: string;
  // A key of cities.json, or "online".
  city: string;
  // A file in public/talks/logos/.
  logo: string | undefined;
}

// One time a talk was given, with what every view needs from its talk.
export interface Edition {
  // The talk's id, data-talk in the page: it ties every edition of a talk together.
  talk: string;
  title: string;
  date: string;
  venue: Venue;
  slides: Slides | undefined;
  recording: string | undefined;
  source: string | undefined;
}

export interface Talk {
  id: string;
  title: string;
  description: string;
  workshop: boolean;
  // Newest first.
  given: [Edition, ...Edition[]];
}

export interface Catalog {
  // Newest first, by their newest edition.
  talks: Talk[];
  // Every edition of every talk, newest first.
  editions: Edition[];
  cities: Cities;
}

// The three data files as JSON values, before any check.
export interface CatalogFiles {
  talks: unknown;
  venues: unknown;
  cities: unknown;
}

// Every error names the file it is about.
function check<T>(file: string, schema: z.ZodType<T>, data: unknown): T {
  const parsed = schema.safeParse(data);
  if (!parsed.success) {
    throw new Error(`content/talks/${file}:\n${z.prettifyError(parsed.error)}`);
  }
  return parsed.data;
}

const byDate = (a: { date: string }, b: { date: string }): number => b.date.localeCompare(a.date);
// Sorting keeps a list non-empty, which TypeScript can't follow through toSorted.
const newestFirst = <T extends { date: string }>(items: readonly [T, ...T[]]): [T, ...T[]] =>
  items.toSorted(byDate) as [T, ...T[]];

// Each field of a venue comes from the longest name prefix in venues.json that sets it.
function venueResolver(venues: Venues, cities: Cities): (name: string) => Venue {
  const byLength = Object.entries(venues).sort(([a], [b]) => b.length - a.length);
  const lookup = (name: string, field: "city" | "logo"): string | undefined =>
    byLength.find(([prefix, entry]) => name.startsWith(prefix) && entry[field])?.[1][field];
  return name => {
    const city = lookup(name, "city");
    if (!city) {
      throw new Error(`content/talks/venues.json: no city for ${name}`);
    }
    if (city !== "online" && !cities[city]) {
      throw new Error(`content/talks/venues.json: ${name} is in ${city}, which is not in content/talks/cities.json`);
    }
    return { name, city, logo: lookup(name, "logo") };
  };
}

// The catalog from the data files' contents and the decks in decks/: every deck must be linked from talks.json, and
// every deck link must have a deck.
export function parseCatalog(files: CatalogFiles, decks: readonly string[]): Catalog {
  const talks = check("talks.json", Talks, files.talks);
  const venues = check("venues.json", Venues, files.venues);
  const cities = check("cities.json", Cities, files.cities);

  const linked = new Set<string>();
  for (const { given } of talks) {
    for (const { slides } of given) {
      if (slides?.kind !== "deck") {
        continue;
      }
      if (!decks.includes(slides.slug)) {
        throw new Error(`content/talks/talks.json: no deck for /talks/${slides.slug}/`);
      }
      linked.add(slides.slug);
    }
  }
  const missing = decks.filter(slug => !linked.has(slug));
  if (missing.length > 0) {
    throw new Error(`content/talks/talks.json: decks not listed: ${missing.join(", ")}`);
  }

  const venueOf = venueResolver(venues, cities);
  const resolved = talks.map(({ title, description, workshop, given: [first, ...rest] }): Talk => {
    const id = slugify(title);
    const edition = ({ date, venue, slides, recording, source }: typeof first): Edition => ({
      talk: id,
      title,
      date,
      venue: venueOf(venue),
      slides,
      recording,
      source,
    });
    // Mapped first and rest apart, so the list stays a non-empty one.
    const given = newestFirst([edition(first), ...rest.map(edition)]);
    return { id, title, description, workshop: workshop === true, given };
  });
  const sorted = resolved.toSorted((a, b) => byDate(a.given[0], b.given[0]));
  return { talks: sorted, editions: sorted.flatMap(talk => talk.given).toSorted(byDate), cities };
}

// Reads content/talks/*.json under root.
export async function loadCatalog(root: string, decks: readonly string[]): Promise<Catalog> {
  const read = async (file: string): Promise<unknown> => {
    const text = await readFile(resolve(root, "content/talks", file), "utf8");
    try {
      return JSON.parse(text);
    } catch (error) {
      throw new Error(`content/talks/${file}: ${error instanceof Error ? error.message : "not valid JSON"}`, {
        cause: error,
      });
    }
  };
  const [talks, venues, cities] = await Promise.all([read("talks.json"), read("venues.json"), read("cities.json")]);
  return parseCatalog({ talks, venues, cities }, decks);
}
