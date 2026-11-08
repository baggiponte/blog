// Turns the templates in tools/site/ into pages that share baggiponte.com's header, footer and stylesheet.
import { nav, type Tool } from "./catalog.ts";

const escapes: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const escapeHtml = (text: string): string => text.replace(/[&<>"]/g, char => escapes[char] ?? char);

// The text of css from the line that starts with `from` up to, not including, the one that starts with `to`.
function section(css: string, from: string, to: string): string {
  const start = css.indexOf(from);
  const end = css.indexOf(to, start);
  if (start === -1 || end === -1) {
    throw new Error(`global.css has no "${from}" … "${to}" section`);
  }
  return css.slice(start, end);
}

// The blog stylesheet's palette, fonts, base styles, header and footer, so the tools can't drift from the blog's
// look. The blog marks its sections with `/* ---- name ---- */` comments; the rest is its pages and posts.
export function blogChrome(css: string): string {
  const tokens = section(css, ":root {", "/* ---- home ---- */");
  const footer = section(css, "/* ---- footer ---- */", "/* ---- search ---- */");
  return `/* copied from the blog's src/styles/global.css by tools/build.ts: edit it there */\n${tokens}${footer}`;
}

const sun =
  '<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4"></path></svg>';
const moon =
  '<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M20.5 13.5A8.5 8.5 0 1 1 10.5 3.5a6.5 6.5 0 0 0 10 10z"></path></svg>';

// Light by default; a saved choice wins. Runs before paint to avoid a flash, like the blog's Base.astro.
const themeScript = `<script>try{const t=localStorage.getItem("theme");if(t==="dark"||t==="light"){document.documentElement.dataset.theme=t}}catch{}</script>`;

function head(title: string, description: string): string {
  return [
    '<meta charset="utf-8" />',
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />',
    `<title>${escapeHtml(title)}</title>`,
    `<meta name="description" content="${escapeHtml(description)}" />`,
    '<link rel="icon" href="/favicon.svg" type="image/svg+xml" />',
    '<link rel="stylesheet" href="/fonts/instrument-sans/index.css" />',
    '<link rel="stylesheet" href="/fonts/bricolage-grotesque/index.css" />',
    '<link rel="stylesheet" href="/blog.css" />',
    '<link rel="stylesheet" href="/tools.css" />',
    themeScript,
  ].join("\n    ");
}

function header(): string {
  const tabs = nav
    .map(link => {
      const active = link.href === "/";
      return `<a class="tab${active ? " active" : ""}" href="${link.href}"${active ? ' aria-current="page"' : ""}>${link.label}</a>`;
    })
    .join("");
  return `<a class="skip" href="#main">skip to content</a>
    <header class="top">
      <div class="wrap bar">
        <a class="brand" href="https://baggiponte.com/">luca baggi<i>.</i></a>
        <nav class="tabs" aria-label="main">${tabs}</nav>
        <button class="theme-toggle" type="button" aria-label="toggle dark mode">${moon}${sun}</button>
      </div>
    </header>`;
}

function footer(year: number): string {
  return `<footer>
      <div class="wrap foot">
        <span>© ${year} luca baggi · small tools that run in your browser</span>
        <span class="links"><a href="https://baggiponte.com/">blog</a><a href="https://github.com/baggiponte">github</a></span>
      </div>
    </footer>
    <script src="/theme.js"></script>`;
}

function list(entries: Tool[]): string {
  const items = entries.map(
    tool =>
      `<li><a href="/${tool.slug}/"><h2>${escapeHtml(tool.name)} <span class="arr">→</span></h2><p>${escapeHtml(tool.description)}</p></a></li>`,
  );
  return `<ul class="tool-list">\n${items.join("\n")}\n</ul>`;
}

// What a template's comments stand for: `<!-- head: title | description -->`, `<!-- header -->`, `<!-- footer -->`
// and, on the index, `<!-- tools -->`.
export function render(template: string, entries: Tool[], year: number): string {
  const markers: [RegExp, (...groups: string[]) => string][] = [
    [/<!-- head: (.+?) \| (.+?) -->/, (_, title = "", description = "") => head(title, description)],
    [/<!-- header -->/, () => header()],
    [/<!-- footer -->/, () => footer(year)],
  ];
  let page = template;
  for (const [marker, replace] of markers) {
    if (!marker.test(page)) {
      throw new Error(`template has no ${marker.source} marker`);
    }
    page = page.replace(marker, (...groups: string[]) => replace(...groups));
  }
  return page.replace("<!-- tools -->", () => list(entries));
}
