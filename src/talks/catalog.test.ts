// The checks on content/talks/*.json and how the catalog resolves them, from plain values: no files, no rendering.
import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { loadCatalog, parseCatalog } from "./catalog.ts";
import { cities, decks, files, talk, venues } from "./fixture.ts";

test("venues resolve field by field from the longest prefix, and editions sort newest first", () => {
  const { talks, editions } = parseCatalog(files, decks);
  assert.equal(talks[0]?.id, "polars-friends");
  assert.deepEqual(
    editions.map(({ date, venue }) => [date, venue]),
    [
      ["2025-06-06", { name: "PyData London 2025", city: "London", logo: "pydata.png" }],
      ["2024-05-23", { name: "PyCon Italia 2024", city: "Florence", logo: "pycon-italia.png" }],
    ],
  );
});

test("slides are a deck in decks/ or a link elsewhere", () => {
  const { editions } = parseCatalog(files, decks);
  assert.deepEqual(
    editions.map(edition => edition.slides),
    [
      { kind: "external", url: "https://example.com/slides" },
      { kind: "deck", slug: "pyconit-polars" },
    ],
  );
});

test("talks sort by their newest edition", () => {
  const older = { ...talk, title: "older", given: [{ date: "2023-01-01", venue: "PyData London 2023" }] };
  const { talks } = parseCatalog({ ...files, talks: [older, talk] }, decks);
  assert.deepEqual(
    talks.map(each => each.id),
    ["polars-friends", "older"],
  );
});

test("a deck missing from talks.json fails", () => {
  assert.throws(() => parseCatalog(files, [...decks, "orphan"]), /decks not listed: orphan/);
});

test("a /talks/<slug>/ link without a deck fails", () => {
  assert.throws(() => parseCatalog(files, []), /no deck for \/talks\/pyconit-polars\//);
});

test("a venue without a city fails", () => {
  const { "PyData London": _, ...rest } = venues;
  assert.throws(() => parseCatalog({ ...files, venues: rest }, decks), /no city for PyData London 2025/);
});

test("a city missing from cities.json fails", () => {
  assert.throws(
    () => parseCatalog({ ...files, cities: { Florence: cities.Florence } }, decks),
    /PyData London 2025 is in London, which is not in content\/talks\/cities.json/,
  );
});

test("a malformed entry fails with its file and path", () => {
  const talks = [{ ...talk, given: [{ ...talk.given[0], date: "23 May 2024" }] }];
  assert.throws(
    () => parseCatalog({ ...files, talks }, decks),
    /content\/talks\/talks.json:[\s\S]*at \[0\]\.given\[0\]\.date/,
  );
});

test("slides that are neither a deck path nor a URL fail", () => {
  const talks = [{ ...talk, given: [{ ...talk.given[0], slides: "talks/pyconit-polars" }] }];
  assert.throws(() => parseCatalog({ ...files, talks }, decks), /content\/talks\/talks.json:[\s\S]*slides/);
});

test("two talks with the same id fail", () => {
  const talks = [talk, { ...talk, title: "Polars, & friends!", given: [talk.given[1]] }];
  assert.throws(() => parseCatalog({ ...files, talks }, decks), /two talks share the id polars-friends/);
});

// Reading the files is all loadCatalog adds; this is the one test that needs them on disk.
test("a file that isn't JSON fails with its name", async () => {
  const root = await mkdtemp(join(tmpdir(), "talks-"));
  await mkdir(join(root, "content/talks"), { recursive: true });
  await writeFile(join(root, "content/talks/talks.json"), JSON.stringify([talk]));
  await writeFile(join(root, "content/talks/venues.json"), JSON.stringify(venues));
  await writeFile(join(root, "content/talks/cities.json"), '{ "Florence": { "at": [11.25, 43.77] }, }');
  await assert.rejects(loadCatalog(root, decks), /content\/talks\/cities.json: .*JSON/);
});
