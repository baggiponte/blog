import { execFile, spawnSync } from "node:child_process";
import { readdir, rm, writeFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { parser } from "@slidev/cli";

const root = fileURLToPath(new URL("../", import.meta.url));
const dist = resolve(root, "dist");
function run(command: string, args: string[], env = process.env) {
  const result = spawnSync(command, args, { cwd: root, stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run("pnpm", ["exec", "astro", "build"]);

// Each deck in decks/ is a Slidev app at /talks/<slug>/. Astro empties dist/ first, so they build after it.
const decks = (await readdir(resolve(root, "decks"))).sort();
const execute = promisify(execFile);
// The first failed build stops the ones still running and keeps the queue from starting more.
const failed = new AbortController();
async function build(slug: string) {
  const directory = resolve(root, "decks", slug);
  const entry = resolve(directory, "slides.md");
  const data = await parser.load({ roots: [directory], userRoot: directory }, entry);
  const base = `/talks/${slug}/`;
  try {
    await execute(
      "pnpm",
      ["exec", "slidev", "build", entry, "--base", base, "--out", join(dist, base), "--router-mode", "history"],
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
      rewrites.push(`${base}${suffix} ${base} 200`);
    }
  }
  for (const suffix of ["presenter", "presenter/", "overview", "overview/", "notes", "notes/"]) {
    rewrites.push(`${base}${suffix} ${base} 200`);
  }
  // Slidev's GitHub Pages fallback must not turn unknown paths into an app shell.
  await rm(join(dist, base, "404.html"), { force: true });
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

// Keep permanent directory redirects rather than the assets service's 307s.
const redirects: string[] = [];
async function visit(directory: string, url = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  if (url && entries.some(entry => entry.name === "index.html" && entry.isFile())) {
    redirects.push(`${url} ${url}/ 301`);
  }
  for (const entry of entries) {
    if (entry.isDirectory() && !entry.name.startsWith("_"))
      await visit(join(directory, entry.name), `${url}/${entry.name}`);
  }
}
await visit(dist);
const rewrites = decks.flatMap(slug => results.get(slug) ?? []);
await writeFile(join(dist, "_redirects"), `${[...redirects.sort(), ...rewrites].join("\n")}\n`);

// `cf build` autodetects Astro and expects its Cloudflare adapter, so call the
// Wrangler delegate it would otherwise use to package dist/ as static assets.
// Deploy the result with `cf deploy --prebuilt`.
run("pnpm", ["exec", "cf-wrangler", "build"], { ...process.env, CLOUDFLARE_CF_AUTH: "true" });
