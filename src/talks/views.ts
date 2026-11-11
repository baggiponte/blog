// The markup of the talks page's three views, by talk, timeline and map, from a resolved catalog (catalog.ts).
import { formatDate } from "../lib/dates.ts";
import { type Html, html, join } from "../lib/html.ts";
import type { Catalog, Edition, Talk, Venue } from "./catalog.ts";
import type { Slides } from "./data.ts";
import { height, landDots, locate, width } from "./map.ts";

export interface Views {
  talks: string;
  timeline: string;
  map: string;
}

// Edition dates are ISO calendar days ("2024-05-23"), which Date reads as midnight UTC.
const date = (iso: string): string => formatDate(new Date(iso));
const time = (iso: string): Html => html`<time datetime="${iso}">${date(iso)}</time>`;
const link = (href: string, label: string): Html => html`<a href="${href}">${label}</a>`;
const count = (n: number): string => `${n} ${n === 1 ? "talk" : "talks"}`;
const hrefOf = (slides: Slides): string => (slides.kind === "deck" ? `/talks/${slides.slug}/` : slides.url);
// Always three slots, empty when missing, so slides / video / code line up down the list.
const links = (edition: Edition): Html[] =>
  (
    [
      [edition.slides && hrefOf(edition.slides), "slides"],
      [edition.recording, "video"],
      [edition.source, "code"],
    ] as const
  ).map(([href, label]) => (href ? link(href, label) : html`<span></span>`));
const titleOf = (edition: Edition): Html | string =>
  edition.slides ? link(hrefOf(edition.slides), edition.title) : edition.title;

// Venues without a logo keep the slot so names line up. Pass the venue's name as alt when no name follows.
const venueLogo = (venue: Venue, alt = ""): Html =>
  venue.logo
    ? html`<img class="logo" src="/talks/logos/${venue.logo}" alt="${alt}" title="${venue.name}" width="20" height="20">`
    : html`<span class="logo"></span>`;
const venueLabel = (venue: Venue): Html => html`${venueLogo(venue)}${venue.name}`;

export function renderViews(catalog: Catalog): Views {
  return {
    talks: renderTalks(catalog.talks).markup,
    timeline: renderTimeline(catalog).markup,
    map: renderMap(catalog, landDots()).markup,
  };
}

function renderTalks(talks: Talk[]): Html {
  const articles = talks.map(({ title, description, workshop, given }) => {
    // Not every edition has slides online: the heading links the newest one that does.
    const slides = given.find(edition => edition.slides)?.slides;
    // Every talk, given once or many times, lists its editions the same way.
    return html`    <article>
      <h2>${slides ? link(hrefOf(slides), title) : title}${workshop ? html` <span class="kind">workshop</span>` : ""}</h2>
      ${time(given[0].date)}
      <p>${description}</p>
      <ol class="editions">${given.map(
        edition => html`
        <li>${time(edition.date)}<span>${venueLabel(edition.venue)}</span><span class="links">${links(edition)}</span></li>`,
      )}
      </ol>
    </article>`;
  });
  return join(articles, "\n");
}

// Every edition on one time axis, plus the same editions listed by year.
function renderTimeline({ talks, editions }: Catalog): Html {
  const [newest, oldest] = [editions[0], editions.at(-1)];
  if (!(newest && oldest)) {
    throw new Error("content/talks/talks.json: no talks to show");
  }
  const first = Number(oldest.date.slice(0, 4));
  const last = Number(newest.date.slice(0, 4)) + 1;
  const start = Date.UTC(first, 0, 1);
  const x = (iso: string): number => ((Date.parse(iso) - start) / (Date.UTC(last, 0, 1) - start)) * 100;

  // Dots closer than this (in % of the axis) stack: each takes the lowest free row at its date.
  const gap = 3.5;
  const rows: number[] = [];
  const dots = editions.toReversed().map(edition => {
    const left = x(edition.date);
    let row = rows.findIndex(end => left - end >= gap);
    if (row < 0) {
      row = rows.length;
    }
    rows[row] = left;
    return html`
        <button type="button" data-talk="${edition.talk}" style="left: ${left.toFixed(2)}%; --row: ${row}" aria-label="${edition.title}, ${edition.venue.name}, ${date(edition.date)}"></button>`;
  });
  const years = Array.from({ length: last - first }, (_, i) => first + i);
  const byYear = Map.groupBy(editions, edition => Number(edition.date.slice(0, 4)));
  const ofYear = (year: number): Edition[] => byYear.get(year) ?? [];
  const ticks = years.map(
    year => html`
        <span style="left: ${x(`${year}-01-01`).toFixed(2)}%">${year}<b>${ofYear(year).length}</b></span>`,
  );
  const lists = years.toReversed().map(
    year => html`
      <h2>${year}<span>${count(ofYear(year).length)}</span></h2>
      <ol>${ofYear(year).map(
        edition => html`
          <li data-talk="${edition.talk}">${time(edition.date)}<span><span class="talk">${titleOf(edition)}</span>${venueLabel(edition.venue)}</span></li>`,
      )}
      </ol>`,
  );
  return html`    <p class="total"><strong>${count(editions.length)}</strong> given since ${first}, ${talks.length} different ones</p>
      <div class="axis" style="--rows: ${rows.length}">${dots}${ticks}
      </div>
      <p class="caption" aria-live="polite">hover or tap a talk to see every time it was given.</p>${lists}`;
}

// Where the "online" pill sits on the map, in map units: the empty Atlantic corner.
const onlinePill: [number, number] = [110, 50];

// Every city on a dot map of Europe, plus the same editions listed by city.
// A city lists the talks given there; data-talk on each row lets a talk light up every city it went to.
function renderMap({ editions, cities }: Catalog, land: string): Html {
  const byCity = Map.groupBy(editions, edition => edition.venue.city);
  const ofCity = (city: string): Edition[] => byCity.get(city) ?? [];
  // Busiest city first, online last.
  const order = [...byCity.keys()].sort(
    (a, b) => Number(a === "online") - Number(b === "online") || ofCity(b).length - ofCity(a).length,
  );
  const percent = (value: number, of: number): string => `${((value / of) * 100).toFixed(2)}%`;

  const marks = order.map(city => {
    const given = ofCity(city);
    const talks = [...new Set(given.map(edition => edition.talk))].join(" ");
    // Online talks have no place in cities.json: they share a pill in the empty Atlantic corner.
    const place = cities[city];
    const [x, y] = place ? locate(city, place.at) : onlinePill;
    const side = place ? (place.label ?? "right") : "online";
    const position = `--x: ${percent(x, width)}; --y: ${percent(y, height)}`;
    // The card listing a city's talks, each after its venue's logo, opens away from the nearest edges of the map.
    return html`
        <button type="button" class="place ${side}" data-city="${city}" data-talks="${talks}" style="${position}; --r: ${(4 + Math.sqrt(given.length) * 3).toFixed(1)}" aria-label="${city}, ${count(given.length)}"><span>${city} <b>${given.length}</b></span></button>
        <div class="card${x > width * 0.7 ? " west" : ""}${y > height * 0.55 ? " up" : ""}" data-city="${city}" style="${position}">
          <strong>${city}</strong> ${count(given.length)}
          <ol>${given.map(
            edition => html`
            <li>${time(edition.date)}<span>${venueLogo(edition.venue, edition.venue.name)}${titleOf(edition)}</span></li>`,
          )}
          </ol>
        </div>`;
  });
  const lists = order.map(
    city => html`
      <h2>${city}<span>${count(ofCity(city).length)}</span></h2>
      <ol>${ofCity(city).map(
        edition => html`
          <li data-talk="${edition.talk}" data-city="${city}">${time(edition.date)}<span><span class="talk">${titleOf(edition)}</span>${venueLabel(edition.venue)}</span></li>`,
      )}
      </ol>`,
  );
  const located = order.filter(city => city !== "online").length;
  return html`    <p class="total"><strong>${count(editions.length)}</strong> given in ${located} cities${byCity.has("online") ? " and online" : ""}</p>
      <div class="places">
        <svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><path d="${land}"/></svg>${marks}
      </div>
      <p class="caption" aria-live="polite">hover or tap a city to see the talks given there, or a talk to see everywhere it went.</p>${lists}`;
}
