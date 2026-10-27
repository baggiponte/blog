// Fast checks of the catalog renderer and the data validation, without Astro, Slidev or a browser: pnpm test:unit.
import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { renderCatalog } from "./catalog.ts";

const talk = {
  title: "polars & friends",
  description: "Dataframes, fast.",
  given: [
    { date: "2024-05-23", venue: "PyCon Italia 2024", slides: "/talks/pyconit-polars/" },
    { date: "2025-06-06", venue: "PyData London 2025", slides: "https://example.com/slides" },
  ],
};
const venues = {
  "PyCon Italia": { logo: "pycon-italia.png" },
  "PyCon Italia 2024": { city: "Florence" },
  PyData: { logo: "pydata.png" },
  "PyData London": { city: "London" },
};
const cities = { Florence: { at: [11.25, 43.77] }, London: { at: [-0.13, 51.51] } };

// A content/talks/ directory with the given data files.
async function fixture(data: { talks?: unknown; venues?: unknown; cities?: unknown } = {}) {
  const root = await mkdtemp(join(tmpdir(), "talks-"));
  await mkdir(join(root, "content/talks"), { recursive: true });
  const files = {
    "talks.json": data.talks ?? [talk],
    "venues.json": data.venues ?? venues,
    "cities.json": data.cities ?? cities,
  };
  for (const [file, value] of Object.entries(files))
    await writeFile(join(root, "content/talks", file), JSON.stringify(value));
  return root;
}

test("renders every view, escapes data and ids talks by title", async () => {
  const views = await renderCatalog(await fixture(), ["pyconit-polars"]);
  const page = Object.values(views).join("\n");
  assert.match(page, /<h2><a href="https:\/\/example.com\/slides">polars &amp; friends<\/a><\/h2>/);
  assert.match(page, /data-talk="polars-friends"/);
  // Logo from "PyData", city from "PyData London".
  assert.match(page, /src="\/talks\/logos\/pydata.png" alt="" title="PyData London 2025"/);
  assert.match(page, /data-city="London"/);
  assert.match(page, /<strong>2 talks<\/strong> given in 2 cities</);
});

test("tags workshops in the talks view", async () => {
  const { talks } = await renderCatalog(await fixture({ talks: [{ ...talk, workshop: true }] }), ["pyconit-polars"]);
  assert.match(talks, /polars &amp; friends<\/a> <span class="kind">workshop<\/span><\/h2>/);
});

test("a deck missing from talks.json fails", async () => {
  await assert.rejects(renderCatalog(await fixture(), ["pyconit-polars", "orphan"]), /decks not listed: orphan/);
});

test("a /talks/<slug>/ link without a deck fails", async () => {
  await assert.rejects(renderCatalog(await fixture(), []), /no deck for \/talks\/pyconit-polars\//);
});

test("a venue without a city fails", async () => {
  const { "PyData London": _, ...rest } = venues;
  await assert.rejects(
    renderCatalog(await fixture({ venues: rest }), ["pyconit-polars"]),
    /no city for PyData London 2025/,
  );
});

test("a city missing from cities.json fails", async () => {
  await assert.rejects(
    renderCatalog(await fixture({ cities: { Florence: cities.Florence } }), ["pyconit-polars"]),
    /PyData London 2025 is in London, which is not in content\/talks\/cities.json/,
  );
});

test("a malformed entry fails with its path", async () => {
  const talks = [{ ...talk, given: [{ ...talk.given[0], date: "23 May 2024" }] }];
  await assert.rejects(
    renderCatalog(await fixture({ talks }), ["pyconit-polars"]),
    /content\/talks\/talks.json:[\s\S]*at \[0\]\.given\[0\]\.date/,
  );
});

test("two talks with the same id fail", async () => {
  const talks = [talk, { ...talk, title: "Polars, & friends!", given: [talk.given[1]] }];
  await assert.rejects(
    renderCatalog(await fixture({ talks }), ["pyconit-polars"]),
    /two talks share the id polars-friends/,
  );
});
