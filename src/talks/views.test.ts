// What the talks page shows for a small catalog: links, escaping, ids and totals.
import assert from "node:assert/strict";
import { test } from "node:test";

import { parseCatalog } from "./catalog.ts";
import { decks, files, talk } from "./fixture.ts";
import { renderViews } from "./views.ts";

const views = renderViews(parseCatalog(files, decks));

test("a talk's heading links its newest slides, and its title is escaped", () => {
  assert.match(views.talks, /<h2><a href="https:\/\/example.com\/slides">polars &amp; friends<\/a><\/h2>/);
});

test("a deck's slides link to its page on the site", () => {
  assert.match(views.talks, /<a href="\/talks\/pyconit-polars\/">slides<\/a>/);
});

test("every row of a talk carries its id, so the page can light them up together", () => {
  assert.equal(views.timeline.match(/data-talk="polars-friends"/g)?.length, 4);
});

test("a venue shows its logo, named after the venue", () => {
  assert.match(views.talks, /src="\/talks\/logos\/pydata.png" alt="" title="PyData London 2025"/);
});

test("the map counts editions and cities", () => {
  assert.match(views.map, /data-city="London"/);
  assert.match(views.map, /<strong>2 talks<\/strong> given in 2 cities</);
});

test("workshops are tagged in the talks view", () => {
  const { talks } = renderViews(parseCatalog({ ...files, talks: [{ ...talk, workshop: true }] }, decks));
  assert.match(talks, /polars &amp; friends<\/a> <span class="kind">workshop<\/span><\/h2>/);
});
