// Builds tools.baggiponte.com into tools/dist/: the static files in tools/site/, each page wrapped in the blog's
// header and footer, plus the blog's stylesheet chrome and fonts.
import { cp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { tools } from "./catalog.ts";
import { blogChrome, render } from "./page.ts";

const root = fileURLToPath(new URL("../", import.meta.url));
const site = join(root, "tools/site");
const dist = join(root, "tools/dist");

await rm(dist, { recursive: true, force: true });
await cp(site, dist, { recursive: true });

// The blog's own fonts, from the same packages and versions the blog uses.
for (const font of ["instrument-sans", "bricolage-grotesque"]) {
  const from = join(root, "node_modules/@fontsource-variable", font);
  await cp(join(from, "index.css"), join(dist, "fonts", font, "index.css"));
  await cp(join(from, "files"), join(dist, "fonts", font, "files"), { recursive: true });
}
await writeFile(join(dist, "blog.css"), blogChrome(await readFile(join(root, "src/styles/global.css"), "utf8")));

// Every tool in the catalog has a page, and every page gets the shared chrome.
for (const tool of tools) {
  await stat(join(dist, tool.slug, "index.html")).catch((error: unknown) => {
    throw new Error(`${tool.slug} is in tools/catalog.ts but has no tools/site/${tool.slug}/index.html`, {
      cause: error,
    });
  });
}
const year = new Date().getFullYear();
for (const entry of await readdir(dist, { recursive: true })) {
  if (entry.endsWith(".html")) {
    const path = join(dist, entry);
    await writeFile(path, render(await readFile(path, "utf8"), tools, year));
  }
}
// Permanent redirects for a tool's address without its trailing slash, rather than the assets service's 307s.
await writeFile(join(dist, "_redirects"), tools.map(tool => `/${tool.slug} /${tool.slug}/ 301\n`).join(""));
console.log(`tools: ${tools.length} tools in tools/dist/`);
