// Renders site/talks.json into the <!-- talks --> placeholder of site/index.html.
import { readFile } from "node:fs/promises";
import { height, landDots, locate, width } from "./map.mjs";

const escape = text => String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const date = iso => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const time = iso => `<time datetime="${iso}">${date(iso)}</time>`;
const link = (href, label) => `<a href="${escape(href)}">${escape(label)}</a>`;
// Always three slots, empty when missing, so slides / video / code line up down the list.
const links = given => [[given.slides, "slides"], [given.recording, "video"], [given.source, "code"]]
  .map(([href, label]) => href ? link(href, label) : "<span></span>").join("");

// Organiser logos keyed by venue prefix ("PyData" covers London, Global, Amsterdam); the longest match wins.
const venues = JSON.parse(await readFile(new URL("../site/venues.json", import.meta.url), "utf8"));
const prefixes = Object.keys(venues).sort((a, b) => b.length - a.length);
export const venueLabel = venue => {
  const match = prefixes.find(prefix => venue.startsWith(prefix));
  // Empty alt: the venue name follows. Venues without a logo keep the slot so names line up.
  const logo = match ? `<img class="logo" src="/logos/${venues[match].logo}" alt="" width="20" height="20">` : `<span class="logo"></span>`;
  return `${logo}${escape(venue)}`;
};

export async function renderCatalog(root, decks) {
  const talks = JSON.parse(await readFile(new URL("site/talks.json", root), "utf8"));
  const linked = new Set();
  for (const talk of talks) {
    talk.given.sort((a, b) => b.date.localeCompare(a.date));
    for (const { slides } of talk.given) {
      if (!slides?.startsWith("/")) continue;
      const slug = slides.slice(1, -1);
      if (!decks.includes(slug)) throw new Error(`site/talks.json: no deck for ${slides}`);
      linked.add(slug);
    }
  }
  const missing = decks.filter(slug => !linked.has(slug));
  if (missing.length) throw new Error(`site/talks.json: decks not listed: ${missing.join(", ")}`);
  talks.sort((a, b) => b.given[0].date.localeCompare(a.given[0].date));

  const articles = talks.map(({ title, description, given: [latest, ...rest] }) => {
    // Every talk, given once or many times, lists its editions the same way.
    const editions = `
      <ol class="editions">${[latest, ...rest].map(given => `
        <li>${time(given.date)}<span>${venueLabel(given.venue)}</span><span class="links">${links(given)}</span></li>`).join("")}
      </ol>`;
    // Not every edition has slides online: the heading links the newest one that does.
    const slides = [latest, ...rest].find(given => given.slides)?.slides;
    return `
    <article>
      <h2>${slides ? link(slides, title) : escape(title)}</h2>
      ${time(latest.date)}
      <p>${escape(description)}</p>${editions}
    </article>`;
  });
  const page = await readFile(new URL("site/index.html", root), "utf8");
  const places = JSON.parse(await readFile(new URL("site/places.json", root), "utf8"));
  return page.replace("    <!-- talks -->", articles.join("").slice(1))
    .replace("    <!-- timeline -->", renderTimeline(talks))
    .replace("    <!-- map -->", await renderMap(talks, places));
}

// Every appearance on one time axis, plus the same appearances listed by year.
// data-talk ties each dot and row to its talk, so the page script can light up all of them at once.
function renderTimeline(talks) {
  const appearances = talks.flatMap(({ title, given }, talk) => given.map(given => ({ ...given, talk, title })));
  appearances.sort((a, b) => b.date.localeCompare(a.date));
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
    const label = `${given.title}, ${given.venue}, ${date(given.date)}`;
    return `
        <button type="button" data-talk="${given.talk}" style="left: ${left.toFixed(2)}%; --row: ${row}" aria-label="${escape(label)}"></button>`;
  });
  const years = Array.from({ length: last - first }, (_, i) => first + i);
  const given = year => appearances.filter(given => given.date.startsWith(year));
  const count = n => `${n} ${n === 1 ? "talk" : "talks"}`;
  const ticks = years.map(year => `
        <span style="left: ${x(`${year}-01-01`).toFixed(2)}%">${year}<b>${given(year).length}</b></span>`);
  const lists = years.toReversed().map(year => {
    const items = given(year).map(given => `
          <li data-talk="${given.talk}">${time(given.date)}<span><span class="talk">${given.slides ? link(given.slides, given.title) : escape(given.title)}</span>${venueLabel(given.venue, true)}</span></li>`);
    return `
      <h2>${year}<span>${count(given(year).length)}</span></h2>
      <ol>${items.join("")}
      </ol>`;
  });
  return `    <p class="total"><strong>${count(appearances.length)}</strong> given since ${first}, ${talks.length} different ones</p>
      <div class="axis" style="--rows: ${rows.length}">${dots.join("")}${ticks.join("")}
      </div>
      <p class="caption" aria-live="polite">Hover or tap a talk to see every time it was given.</p>${lists.join("")}`;
}

// Every city on a dot map of Europe, plus the same appearances listed by city.
// A city lists the talks given there; data-talk on each row lets a talk light up every city it went to.
async function renderMap(talks, places) {
  const venues = Object.keys(places.venues).sort((a, b) => b.length - a.length);
  const cityOf = venue => {
    const match = venues.find(prefix => venue.startsWith(prefix));
    if (!match) throw new Error(`site/places.json: no city for ${venue}`);
    return places.venues[match];
  };
  const appearances = talks.flatMap(({ title, given }, talk) => given.map(given => ({ ...given, talk, title, city: cityOf(given.venue) })));
  appearances.sort((a, b) => b.date.localeCompare(a.date));
  const byCity = Map.groupBy(appearances, given => given.city);
  // Busiest city first, online last.
  const cities = [...byCity.keys()].sort((a, b) => (a === "online") - (b === "online") || byCity.get(b).length - byCity.get(a).length);
  const count = n => `${n} ${n === 1 ? "talk" : "talks"}`;
  const percent = (value, of) => `${(value / of * 100).toFixed(2)}%`;
  const title = given => given.slides ? link(given.slides, given.title) : escape(given.title);

  const marks = cities.map(city => {
    const given = byCity.get(city);
    const talks = [...new Set(given.map(given => given.talk))].join(" ");
    // Online talks have no place: they share a pill in the empty Atlantic corner.
    const [x, y] = city === "online" ? [110, 50] : locate(city, places.cities[city].at);
    const side = city === "online" ? "online" : places.cities[city].label ?? "right";
    const position = `--x: ${percent(x, width)}; --y: ${percent(y, height)}`;
    // The card listing a city's talks opens away from the nearest edges of the map.
    const items = given.map(given => `
            <li>${time(given.date)}<span>${title(given)}</span></li>`);
    return `
        <button type="button" class="place ${side}" data-city="${city}" data-talks="${talks}" style="${position}; --r: ${(4 + Math.sqrt(given.length) * 3).toFixed(1)}" aria-label="${escape(`${city}, ${count(given.length)}`)}"><span>${city} <b>${given.length}</b></span></button>
        <div class="card${x > width * .7 ? " west" : ""}${y > height * .55 ? " up" : ""}" data-city="${city}" style="${position}">
          <strong>${city}</strong> ${count(given.length)}
          <ol>${items.join("")}
          </ol>
        </div>`;
  });
  const lists = cities.map(city => {
    const items = byCity.get(city).map(given => `
          <li data-talk="${given.talk}" data-city="${city}">${time(given.date)}<span><span class="talk">${title(given)}</span>${venueLabel(given.venue)}</span></li>`);
    return `
      <h2>${city}<span>${count(items.length)}</span></h2>
      <ol>${items.join("")}
      </ol>`;
  });
  const located = cities.filter(city => city !== "online").length;
  return `    <p class="total"><strong>${count(appearances.length)}</strong> given in ${located} cities${byCity.has("online") ? " and online" : ""}</p>
      <div class="places">
        <svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><path d="${await landDots()}"/></svg>${marks.join("")}
      </div>
      <p class="caption" aria-live="polite">Hover or tap a city to see the talks given there, or a talk to see everywhere it went.</p>${lists.join("")}`;
}
