// Renders content/talks/*.json into the markup of the talks page's three views: by talk, timeline and map.
import { formatDate } from "../lib/dates.ts";
import { Cities, type Edition, load, slugify, type Talk, Talks, Venues } from "./data.ts";
import { height, landDots, locate, width } from "./map.ts";

// Markup built with html`` is kept as is; every other interpolated value is escaped, arrays are joined.
class Html {
  markup: string;
  constructor(markup: string) {
    this.markup = markup;
  }
}
type Value = Html | string | number | null | undefined | Value[];
const escapes: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const escapeHtml = (text: string | number) => String(text).replace(/[&<>"]/g, c => escapes[c] ?? c);
const render = (value: Value): string =>
  value instanceof Html
    ? value.markup
    : Array.isArray(value)
      ? value.map(render).join("")
      : value == null
        ? ""
        : escapeHtml(value);
const html = (strings: TemplateStringsArray, ...values: Value[]) =>
  new Html(String.raw({ raw: strings }, ...values.map(render)));

// An edition with what the views need from its talk: data-talk="<talk>" ties it to the talk, so the page script can light up all of them at once.
type Appearance = Edition & { talk: string; title: string; city: string };
type Label = (venue: string) => Html;

// Edition dates are ISO calendar days ("2024-05-23"), which Date reads as midnight UTC.
const date = (iso: string): string => formatDate(new Date(iso));
const time = (iso: string) => html`<time datetime="${iso}">${date(iso)}</time>`;
const link = (href: string, label: string) => html`<a href="${href}">${label}</a>`;
const count = (n: number) => `${n} ${n === 1 ? "talk" : "talks"}`;
const newestFirst = (a: { date: string }, b: { date: string }) => b.date.localeCompare(a.date);
// Always three slots, empty when missing, so slides / video / code line up down the list.
const links = (given: Edition) =>
  (
    [
      [given.slides, "slides"],
      [given.recording, "video"],
      [given.source, "code"],
    ] as const
  ).map(([href, label]) => (href ? link(href, label) : html`<span></span>`));
const titleOf = (given: Appearance) => (given.slides ? link(given.slides, given.title) : given.title);

export async function renderCatalog(root: string, decks: string[]) {
  const [talks, venues, cities] = await Promise.all([
    load(root, "talks.json", Talks),
    load(root, "venues.json", Venues),
    load(root, "cities.json", Cities),
  ]);
  // Each field comes from the longest venue prefix that sets it.
  const byLength = Object.entries(venues).sort(([a], [b]) => b.length - a.length);
  const lookup = (venue: string, field: "city" | "logo") =>
    byLength.find(([prefix, entry]) => venue.startsWith(prefix) && entry[field])?.[1][field];

  // Venues without a logo keep the slot so names line up. Pass the venue as alt when no name follows.
  const venueLogo = (venue: string, alt = "") => {
    const logo = lookup(venue, "logo");
    return logo
      ? html`<img class="logo" src="/talks/logos/${logo}" alt="${alt}" title="${venue}" width="20" height="20">`
      : html`<span class="logo"></span>`;
  };
  const venueLabel: Label = venue => html`${venueLogo(venue)}${venue}`;

  const linked = new Set<string>();
  for (const talk of talks) {
    talk.given.sort(newestFirst);
    for (const { slides } of talk.given) {
      if (!slides?.startsWith("/")) continue;
      const slug = slides.slice("/talks/".length, -1);
      if (!decks.includes(slug)) throw new Error(`content/talks/talks.json: no deck for ${slides}`);
      linked.add(slug);
    }
  }
  const missing = decks.filter(slug => !linked.has(slug));
  if (missing.length) throw new Error(`content/talks/talks.json: decks not listed: ${missing.join(", ")}`);
  talks.sort((a, b) => newestFirst(a.given[0], b.given[0]));

  const appearances: Appearance[] = talks
    .flatMap(({ title, given }) =>
      given.map(given => {
        const city = lookup(given.venue, "city");
        if (!city) throw new Error(`content/talks/venues.json: no city for ${given.venue}`);
        if (city !== "online" && !cities[city])
          throw new Error(
            `content/talks/venues.json: ${given.venue} is in ${city}, which is not in content/talks/cities.json`,
          );
        return { ...given, talk: slugify(title), title, city };
      }),
    )
    .sort(newestFirst);

  return {
    talks: renderTalks(talks, venueLabel).markup,
    timeline: renderTimeline(talks, appearances, venueLabel).markup,
    map: renderMap(appearances, cities, venueLogo, venueLabel, landDots()).markup,
  };
}

function renderTalks(talks: Talk[], venueLabel: Label) {
  const articles = talks.map(({ title, description, workshop, given }) => {
    // Not every edition has slides online: the heading links the newest one that does.
    const slides = given.find(given => given.slides)?.slides;
    // Every talk, given once or many times, lists its editions the same way.
    return html`    <article>
      <h2>${slides ? link(slides, title) : title}${workshop ? html` <span class="kind">workshop</span>` : ""}</h2>
      ${time(given[0].date)}
      <p>${description}</p>
      <ol class="editions">${given.map(
        given => html`
        <li>${time(given.date)}<span>${venueLabel(given.venue)}</span><span class="links">${links(given)}</span></li>`,
      )}
      </ol>
    </article>`;
  });
  return new Html(articles.map(render).join("\n"));
}

// Every appearance on one time axis, plus the same appearances listed by year.
function renderTimeline(talks: Talk[], appearances: Appearance[], venueLabel: Label) {
  const [newest, oldest] = [appearances[0], appearances.at(-1)];
  if (!newest || !oldest) throw new Error("content/talks/talks.json: no talks to show");
  const first = Number(oldest.date.slice(0, 4));
  const last = Number(newest.date.slice(0, 4)) + 1;
  const start = Date.UTC(first, 0, 1);
  const x = (iso: string) => ((Date.parse(iso) - start) / (Date.UTC(last, 0, 1) - start)) * 100;

  // Dots closer than this (in % of the axis) stack: each takes the lowest free row at its date.
  const gap = 3.5;
  const rows: number[] = [];
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
  const ofYear = (year: number) => byYear.get(year) ?? [];
  const ticks = years.map(
    year => html`
        <span style="left: ${x(`${year}-01-01`).toFixed(2)}%">${year}<b>${ofYear(year).length}</b></span>`,
  );
  const lists = years.toReversed().map(
    year => html`
      <h2>${year}<span>${count(ofYear(year).length)}</span></h2>
      <ol>${ofYear(year).map(
        given => html`
          <li data-talk="${given.talk}">${time(given.date)}<span><span class="talk">${titleOf(given)}</span>${venueLabel(given.venue)}</span></li>`,
      )}
      </ol>`,
  );
  return html`    <p class="total"><strong>${count(appearances.length)}</strong> given since ${first}, ${talks.length} different ones</p>
      <div class="axis" style="--rows: ${rows.length}">${dots}${ticks}
      </div>
      <p class="caption" aria-live="polite">hover or tap a talk to see every time it was given.</p>${lists}`;
}

// Every city on a dot map of Europe, plus the same appearances listed by city.
// A city lists the talks given there; data-talk on each row lets a talk light up every city it went to.
function renderMap(
  appearances: Appearance[],
  cities: Cities,
  venueLogo: (venue: string, alt?: string) => Html,
  venueLabel: Label,
  land: string,
) {
  const byCity = Map.groupBy(appearances, given => given.city);
  const ofCity = (city: string) => byCity.get(city) ?? [];
  // Busiest city first, online last.
  const order = [...byCity.keys()].sort(
    (a, b) => Number(a === "online") - Number(b === "online") || ofCity(b).length - ofCity(a).length,
  );
  const percent = (value: number, of: number) => `${((value / of) * 100).toFixed(2)}%`;

  const marks = order.map(city => {
    const given = ofCity(city);
    const talks = [...new Set(given.map(given => given.talk))].join(" ");
    // Online talks have no place in cities.json: they share a pill in the empty Atlantic corner.
    const place = cities[city];
    const [x, y] = place ? locate(city, place.at) : [110, 50];
    const side = place ? (place.label ?? "right") : "online";
    const position = `--x: ${percent(x, width)}; --y: ${percent(y, height)}`;
    // The card listing a city's talks, each after its venue's logo, opens away from the nearest edges of the map.
    return html`
        <button type="button" class="place ${side}" data-city="${city}" data-talks="${talks}" style="${position}; --r: ${(4 + Math.sqrt(given.length) * 3).toFixed(1)}" aria-label="${city}, ${count(given.length)}"><span>${city} <b>${given.length}</b></span></button>
        <div class="card${x > width * 0.7 ? " west" : ""}${y > height * 0.55 ? " up" : ""}" data-city="${city}" style="${position}">
          <strong>${city}</strong> ${count(given.length)}
          <ol>${given.map(
            given => html`
            <li>${time(given.date)}<span>${venueLogo(given.venue, given.venue)}${titleOf(given)}</span></li>`,
          )}
          </ol>
        </div>`;
  });
  const lists = order.map(
    city => html`
      <h2>${city}<span>${count(ofCity(city).length)}</span></h2>
      <ol>${ofCity(city).map(
        given => html`
          <li data-talk="${given.talk}" data-city="${city}">${time(given.date)}<span><span class="talk">${titleOf(given)}</span>${venueLabel(given.venue)}</span></li>`,
      )}
      </ol>`,
  );
  const located = order.filter(city => city !== "online").length;
  return html`    <p class="total"><strong>${count(appearances.length)}</strong> given in ${located} cities${byCity.has("online") ? " and online" : ""}</p>
      <div class="places">
        <svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><path d="${land}"/></svg>${marks}
      </div>
      <p class="caption" aria-live="polite">hover or tap a city to see the talks given there, or a talk to see everywhere it went.</p>${lists}`;
}
