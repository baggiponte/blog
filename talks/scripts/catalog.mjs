// Renders site/talks.json into the <!-- talks --> placeholder of site/index.html.
import { readFile } from "node:fs/promises";

const escape = text => String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const date = iso => new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const time = iso => `<time datetime="${iso}">${date(iso)}</time>`;
const link = (href, label) => `<a href="${escape(href)}">${escape(label)}</a>`;
const links = (given, [slides, recording, source]) => [
  given.slides && link(given.slides, slides),
  given.recording && link(given.recording, recording),
  given.source && link(given.source, source),
].filter(Boolean).join("");

const linksOf = given => {
  const html = links(given, ["slides →", "recording", given.sourceLabel || "original source"]);
  return html && `
      <div class="links">${html}</div>`;
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
    // A talk given once links its slides directly; repeats list every edition instead.
    const meta = rest.length ? `given ${rest.length === 1 ? "twice" : `${rest.length + 1} times`}, latest at ${escape(latest.venue)}` : escape(latest.venue);
    const more = rest.length ? `
      <ol class="editions">${[latest, ...rest].map(given => `
        <li>${time(given.date)}<span>${escape(given.venue)}${given.title ? `<em>${escape(given.title)}</em>` : ""}</span><span class="links">${links(given, ["slides", "video", "code"])}</span></li>`).join("")}
      </ol>` : linksOf(latest);
    // Not every edition has slides online: the heading links the newest one that does.
    const slides = [latest, ...rest].find(given => given.slides)?.slides;
    return `
    <article>
      <h2>${slides ? link(slides, title) : escape(title)}</h2>
      ${time(latest.date)}
      <p class="meta">${meta}</p>
      <p>${escape(description)}</p>${more}
    </article>`;
  });
  const page = await readFile(new URL("site/index.html", root), "utf8");
  return page.replace("    <!-- talks -->", articles.join("").slice(1));
}
