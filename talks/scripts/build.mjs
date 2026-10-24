import { spawnSync } from "node:child_process";
import { copyFile, mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parser } from "@slidev/cli";
import { renderCatalog } from "./catalog.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const dist = resolve(root, "dist");
await rm(dist, { recursive: true, force: true });
await mkdir(dist);
const decks = (await readdir(resolve(root, "decks"))).sort();
await writeFile(resolve(dist, "index.html"), await renderCatalog(new URL("../", import.meta.url), decks));
await copyFile(resolve(root, "site/404.html"), resolve(dist, "404.html"));

const rewrites = [];
for (const slug of decks) {
  const directory = resolve(root, "decks", slug);
  const entry = resolve(directory, "slides.md");
  const data = await parser.load({ roots: [directory], userRoot: directory }, entry);
  const result = spawnSync("pnpm", [
    "exec", "slidev", "build", entry,
    "--base", `/${slug}/`, "--out", resolve(dist, slug),
    "--router-mode", "history",
  ], { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);

  // Exact slide routes: no global SPA fallback swallowing other decks or missing assets.
  for (let slide = 1; slide <= data.slides.length; slide++) {
    for (const suffix of [String(slide), `${slide}/`, `presenter/${slide}`, `presenter/${slide}/`]) {
      rewrites.push(`/${slug}/${suffix} /${slug}/ 200`);
    }
  }
  for (const suffix of ["presenter", "presenter/", "overview", "overview/", "notes", "notes/"]) {
    rewrites.push(`/${slug}/${suffix} /${slug}/ 200`);
  }
  // Slidev's GitHub Pages fallback must not turn unknown paths into an app shell.
  await rm(resolve(dist, slug, "404.html"), { force: true });
  console.log(`${slug}: ${data.slides.length} slides`);
}
await writeFile(resolve(dist, "_redirects"), `${rewrites.join("\n")}\n`);
