// Renders site/talks.json into the <!-- talks -->, <!-- timeline --> and <!-- map --> placeholders of site/index.html.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { height, landDots, locate, width } from "./map.mjs";

// Markup built with html`` is kept as is; every other interpolated value is escaped, arrays are joined.
class Html { constructor(markup) { this.markup = markup; } }
const escape = text => String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const render = value => value instanceof Html ? value.markup : Array.isArray(value) ? value.map(render).join("") : value == null ? "" : escape(value);
const html = (strings, ...values) => new Html(String.raw({ raw: strings }, ...values.map(render)));

const date = iso => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const time = iso => html`<time datetime="${iso}">${date(iso)}</time>`;
const link = (href, label) => html`<a href="${href}">${label}</a>`;
const count = n => `${n} ${n === 1 ? "talk" : "talks"}`;
const newestFirst = (a, b) => b.date.localeCompare(a.date);
// Always three slots, empty when missing, so slides / video / code line up down the list.
const links = given => [[given.slides, "slides"], [given.recording, "video"], [given.source, "code"]]
  .map(([href, label]) => href ? link(href, label) : html`<span></span>`);
const titleOf = given => given.slides ? link(given.slides, given.title) : given.title;

// venues.json and places.json key venues by name prefix ("PyData" covers London, Global, Amsterdam); the longest match wins.
const prefixLookup = table => {
  const prefixes = Object.keys(table).sort((a, b) => b.length - a.length);
  return venue => table[prefixes.find(prefix => venue.startsWith(prefix))];
};

export async function renderCatalog(root, decks) {
  const read = async file => JSON.parse(await readFile(resolve(root, "site", file), "utf8"));
  const [talks, venues, places] = await Promise.all([read("talks.json"), read("venues.json"), read("places.json")]);
  const logoOf = prefixLookup(venues);
  const cityOf = prefixLookup(places.venues);

  // Venues without a logo keep the slot so names line up. Pass the venue as alt when no name follows.
  const venueLogo = (venue, alt = "") => {
    const logo = logoOf(venue)?.logo;
    return logo ? html`<img class="logo" src="/logos/${logo}" alt="${alt}" title="${venue}" width="20" height="20">` : html`<span class="logo"></span>`;
  };
  const venueLabel = venue => html`${venueLogo(venue)}${venue}`;

  const linked = new Set();
  for (const talk of talks) {
    talk.given.sort(newestFirst);
    for (const { slides } of talk.given) {
      if (!slides?.startsWith("/")) continue;
      const slug = slides.slice(1, -1);
      if (!decks.includes(slug)) throw new Error(`site/talks.json: no deck for ${slides}`);
      linked.add(slug);
    }
  }
  const missing = decks.filter(slug => !linked.has(slug));
  if (missing.length) throw new Error(`site/talks.json: decks not listed: ${missing.join(", ")}`);
  talks.sort((a, b) => newestFirst(a.given[0], b.given[0]));

  // Every appearance, newest first. data-talk="<talk>" ties an appearance to its talk, so the page script can light up all of them at once.
  const appearances = talks.flatMap(({ title, given }, talk) => given.map(given => {
    const city = cityOf(given.venue);
    if (!city) throw new Error(`site/places.json: no city for ${given.venue}`);
    if (city !== "online" && !places.cities[city]) throw new Error(`site/places.json: ${given.venue} is in ${city}, which is not in cities`);
    return { ...given, talk, title, city };
  })).sort(newestFirst);

  const page = await readFile(resolve(root, "site/index.html"), "utf8");
  return fill(page, {
    talks: renderTalks(talks, venueLabel),
    timeline: renderTimeline(talks, appearances, venueLabel),
    map: renderMap(appearances, places, venueLogo, venueLabel, await landDots()),
  });
}

// Each placeholder must be there exactly once, so a template edit can't silently drop a view.
function fill(page, views) {
  for (const [name, markup] of Object.entries(views)) {
    const parts = page.split(`    <!-- ${name} -->\n`);
    if (parts.length !== 2) throw new Error(`site/index.html: expected one <!-- ${name} --> placeholder`);
    page = parts.join(`${render(markup)}\n`);
  }
  return page;
}

function renderTalks(talks, venueLabel) {
  const articles = talks.map(({ title, description, given }) => {
    // Not every edition has slides online: the heading links the newest one that does.
    const slides = given.find(given => given.slides)?.slides;
    // Every talk, given once or many times, lists its editions the same way.
    return html`    <article>
      <h2>${slides ? link(slides, title) : title}</h2>
      ${time(given[0].date)}
      <p>${description}</p>
      <ol class="editions">${given.map(given => html`
        <li>${time(given.date)}<span>${venueLabel(given.venue)}</span><span class="links">${links(given)}</span></li>`)}
      </ol>
    </article>`;
  });
  return new Html(articles.map(render).join("\n"));
}

// Every appearance on one time axis, plus the same appearances listed by year.
function renderTimeline(talks, appearances, venueLabel) {
  const first = Number(appearances.at(-1).date.slice(0, 4));
  const last = Number(appearances[0].date.slice(0, 4)) + 1;
  const start = Date.UTC(first, 0, 1);
  const x = iso => (Date.parse(iso) - start) / (Date.UTC(last, 0, 1) - start) * 100;

  // Dots closer than this (in % of the axis) stack: each takes the lowest free row at its date.
  const gap = 3.5;
  const rows = [];
  const dots = appearances.toReversed().map(given => {
    const left = x(given.date);
    let row = rows.findIndex(end => left - end >= gap);
    if (row < 0) row = rows.length;
    rows[row] = left;
    return html`
        <button type="button" data-talk="${given.talk}" style="left: ${left.toFixed(2)}%; --row: ${row}" aria-label="${given.title}, ${given.venue}, ${date(given.date)}"></button>`;
  });
  const years = Array.from({ length: last - first }, (_, i) => first + i);
  const byYear = Map.groupBy(appearances, given => Number(given.date.slice(0, 4)));
  const ofYear = year => byYear.get(year) ?? [];
  const ticks = years.map(year => html`
        <span style="left: ${x(`${year}-01-01`).toFixed(2)}%">${year}<b>${ofYear(year).length}</b></span>`);
  const lists = years.toReversed().map(year => html`
      <h2>${year}<span>${count(ofYear(year).length)}</span></h2>
      <ol>${ofYear(year).map(given => html`
          <li data-talk="${given.talk}">${time(given.date)}<span><span class="talk">${titleOf(given)}</span>${venueLabel(given.venue)}</span></li>`)}
      </ol>`);
  return html`    <p class="total"><strong>${count(appearances.length)}</strong> given since ${first}, ${talks.length} different ones</p>
      <div class="axis" style="--rows: ${rows.length}">${dots}${ticks}
      </div>
      <p class="caption" aria-live="polite">Hover or tap a talk to see every time it was given.</p>${lists}`;
}

// Every city on a dot map of Europe, plus the same appearances listed by city.
// A city lists the talks given there; data-talk on each row lets a talk light up every city it went to.
function renderMap(appearances, places, venueLogo, venueLabel, land) {
  const byCity = Map.groupBy(appearances, given => given.city);
  // Busiest city first, online last.
  const cities = [...byCity.keys()].sort((a, b) => (a === "online") - (b === "online") || byCity.get(b).length - byCity.get(a).length);
  const percent = (value, of) => `${(value / of * 100).toFixed(2)}%`;

  const marks = cities.map(city => {
    const given = byCity.get(city);
    const talks = [...new Set(given.map(given => given.talk))].join(" ");
    // Online talks have no place: they share a pill in the empty Atlantic corner.
    const [x, y] = city === "online" ? [110, 50] : locate(city, places.cities[city].at);
    const side = city === "online" ? "online" : places.cities[city].label ?? "right";
    const position = `--x: ${percent(x, width)}; --y: ${percent(y, height)}`;
    // The card listing a city's talks, each after its venue's logo, opens away from the nearest edges of the map.
    return html`
        <button type="button" class="place ${side}" data-city="${city}" data-talks="${talks}" style="${position}; --r: ${(4 + Math.sqrt(given.length) * 3).toFixed(1)}" aria-label="${city}, ${count(given.length)}"><span>${city} <b>${given.length}</b></span></button>
        <div class="card${x > width * .7 ? " west" : ""}${y > height * .55 ? " up" : ""}" data-city="${city}" style="${position}">
          <strong>${city}</strong> ${count(given.length)}
          <ol>${given.map(given => html`
            <li>${time(given.date)}<span>${venueLogo(given.venue, given.venue)}${titleOf(given)}</span></li>`)}
          </ol>
        </div>`;
  });
  const lists = cities.map(city => html`
      <h2>${city}<span>${count(byCity.get(city).length)}</span></h2>
      <ol>${byCity.get(city).map(given => html`
          <li data-talk="${given.talk}" data-city="${city}">${time(given.date)}<span><span class="talk">${titleOf(given)}</span>${venueLabel(given.venue)}</span></li>`)}
      </ol>`);
  const located = cities.filter(city => city !== "online").length;
  return html`    <p class="total"><strong>${count(appearances.length)}</strong> given in ${located} cities${byCity.has("online") ? " and online" : ""}</p>
      <div class="places">
        <svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><path d="${land}"/></svg>${marks}
      </div>
      <p class="caption" aria-live="polite">Hover or tap a city to see the talks given there, or a talk to see everywhere it went.</p>${lists}`;
}
