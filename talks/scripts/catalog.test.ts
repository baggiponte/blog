// Fast checks of the catalog renderer and the data validation, without Slidev or a browser: pnpm test:unit.
import assert from "node:assert/strict";
import { copyFile, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { renderCatalog } from "./catalog.ts";

const site = fileURLToPath(new URL("../site/", import.meta.url));
const talk = {
  title: "polars & friends",
  description: "Dataframes, fast.",
  given: [
    { date: "2024-05-23", venue: "PyCon Italia 2024", slides: "/pyconit-polars/" },
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

// A site/ directory with the real page template and the given data files.
async function fixture(data: { talks?: unknown; venues?: unknown; cities?: unknown; page?: string } = {}) {
  const root = await mkdtemp(join(tmpdir(), "talks-"));
  await mkdir(join(root, "site"));
  if (data.page === undefined) await copyFile(resolve(site, "index.html"), join(root, "site/index.html"));
  else await writeFile(join(root, "site/index.html"), data.page);
  const files = {
    "talks.json": data.talks ?? [talk],
    "venues.json": data.venues ?? venues,
    "cities.json": data.cities ?? cities,
  };
  for (const [file, value] of Object.entries(files)) await writeFile(join(root, "site", file), JSON.stringify(value));
  return root;
}

test("renders every view, escapes data and ids talks by title", async () => {
  const page = await renderCatalog(await fixture(), ["pyconit-polars"]);
  for (const placeholder of ["<!-- talks -->", "<!-- timeline -->", "<!-- map -->"])
    assert.ok(!page.includes(placeholder));
  assert.match(page, /<h2><a href="https:\/\/example.com\/slides">polars &amp; friends<\/a><\/h2>/);
  assert.match(page, /data-talk="polars-friends"/);
  // Logo from "PyData", city from "PyData London".
  assert.match(page, /src="\/logos\/pydata.png" alt="" title="PyData London 2025"/);
  assert.match(page, /data-city="London"/);
  assert.match(page, /<strong>2 talks<\/strong> given in 2 cities</);
});

test("tags workshops in the talks view", async () => {
  const page = await renderCatalog(await fixture({ talks: [{ ...talk, workshop: true }] }), ["pyconit-polars"]);
  assert.match(page, /polars &amp; friends<\/a> <span class="kind">workshop<\/span><\/h2>/);
});

test("a deck missing from talks.json fails", async () => {
  await assert.rejects(renderCatalog(await fixture(), ["pyconit-polars", "orphan"]), /decks not listed: orphan/);
});

test("a /<slug>/ link without a deck fails", async () => {
  await assert.rejects(renderCatalog(await fixture(), []), /no deck for \/pyconit-polars\//);
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
    /PyData London 2025 is in London, which is not in site\/cities.json/,
  );
});

test("a malformed entry fails with its path", async () => {
  const talks = [{ ...talk, given: [{ ...talk.given[0], date: "23 May 2024" }] }];
  await assert.rejects(
    renderCatalog(await fixture({ talks }), ["pyconit-polars"]),
    /site\/talks.json:[\s\S]*at \[0\]\.given\[0\]\.date/,
  );
});

test("two talks with the same id fail", async () => {
  const talks = [talk, { ...talk, title: "Polars, & friends!", given: [talk.given[1]] }];
  await assert.rejects(
    renderCatalog(await fixture({ talks }), ["pyconit-polars"]),
    /two talks share the id polars-friends/,
  );
});

test("a template without a placeholder fails", async () => {
  const page = (await readFile(resolve(site, "index.html"), "utf8")).replace(/^ *<!-- map -->\n/m, "");
  await assert.rejects(
    renderCatalog(await fixture({ page }), ["pyconit-polars"]),
    /expected one <!-- map --> placeholder/,
  );
});
