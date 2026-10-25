import { execFile } from "node:child_process";
import { copyFile, cp, mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { parser } from "@slidev/cli";
import { renderCatalog } from "./catalog.ts";

const root = fileURLToPath(new URL("../", import.meta.url));
const dist = resolve(root, "dist");
await rm(dist, { recursive: true, force: true });
await mkdir(dist);
const decks = (await readdir(resolve(root, "decks"))).sort();
await writeFile(resolve(dist, "index.html"), await renderCatalog(root, decks));
for (const file of ["404.html", "style.css", "main.js"])
  await copyFile(resolve(root, "site", file), resolve(dist, file));
await cp(resolve(root, "site/logos"), resolve(dist, "logos"), { recursive: true });

const run = promisify(execFile);
// The first failed build stops the ones still running and keeps the queue from starting more.
const failed = new AbortController();
async function build(slug: string) {
  const directory = resolve(root, "decks", slug);
  const entry = resolve(directory, "slides.md");
  const data = await parser.load({ roots: [directory], userRoot: directory }, entry);
  try {
    await run(
      "pnpm",
      [
        "exec",
        "slidev",
        "build",
        entry,
        "--base",
        `/${slug}/`,
        "--out",
        resolve(dist, slug),
        "--router-mode",
        "history",
      ],
      { cwd: root, maxBuffer: 64 * 1024 * 1024, signal: failed.signal },
    );
  } catch (error) {
    if (failed.signal.aborted) throw error;
    failed.abort();
    // Buffered so parallel builds don't interleave; only a failure needs the log.
    const { stdout = "", stderr = "" } = error as { stdout?: string; stderr?: string };
    process.stderr.write(`${stdout}${stderr}`);
    throw new Error(`${slug}: slidev build failed`, { cause: error });
  }

  // Exact slide routes: no global SPA fallback swallowing other decks or missing assets.
  const rewrites = [];
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
  return rewrites;
}

// Each Slidev build is mostly single-threaded, so run one per CPU.
// The workers share one iterator, so each deck is taken by exactly one of them.
const queue = decks.values();
const results = new Map<string, string[]>();
await Promise.all(
  Array.from({ length: Math.min(availableParallelism(), decks.length) }, async () => {
    for (const slug of queue) {
      if (failed.signal.aborted) break;
      results.set(slug, await build(slug));
    }
  }),
);
const rewrites = decks.flatMap(slug => results.get(slug) ?? []);
await writeFile(resolve(dist, "_redirects"), `${rewrites.join("\n")}\n`);
