import { execFile } from "node:child_process";
import { appendFile, rm } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import type { AstroIntegration, AstroIntegrationLogger } from "astro";

import { deckEntry, deckRewrites, listDecks } from "../../talks/decks.ts";
import { slideCount } from "../../talks/slides.ts";

const execute = promisify(execFile);

// Astro sets NODE_ENV=production for its build, and Slidev's UnoCSS then generates fewer utilities than the decks were
// made with: each deck builds with the environment the build was started with, as it did before Astro ran it.
const { NODE_ENV: _astro, ...deckEnv } = process.env;

// Each directory in decks/ is a Slidev app, built into /talks/<slug>/ once Astro has written dist/: Astro empties
// dist/ when it starts, so the decks can't be built before it. `astro dev` doesn't serve them.
export function decks(): AstroIntegration {
  let root = "";
  return {
    name: "decks",
    hooks: {
      "astro:config:done": ({ config }) => {
        root = fileURLToPath(config.root);
      },
      "astro:build:done": async ({ dir, logger }) => {
        const dist = fileURLToPath(dir);
        const rewrites = await buildDecks(root, dist, logger);
        await appendFile(join(dist, "_redirects"), rewrites.map(line => `${line}\n`).join(""));
      },
    },
  };
}

// Builds every deck, one per CPU since each Slidev build is mostly single-threaded, and returns their _redirects
// rewrites in deck order. The first failure stops the builds still running and keeps the queue from starting more.
async function buildDecks(root: string, dist: string, logger: AstroIntegrationLogger): Promise<string[]> {
  const slugs = await listDecks(root);
  const failed = new AbortController();
  const results = new Map<string, string[]>();

  async function build(slug: string): Promise<void> {
    const slides = await slideCount(root, slug);
    const base = `/talks/${slug}/`;
    try {
      await execute(
        "pnpm",
        [
          "exec",
          "slidev",
          "build",
          deckEntry(root, slug),
          "--base",
          base,
          "--out",
          join(dist, base),
          "--router-mode",
          "history",
        ],
        { cwd: root, env: deckEnv, maxBuffer: 64 * 1024 * 1024, signal: failed.signal },
      );
    } catch (error) {
      if (failed.signal.aborted) {
        throw error;
      }
      failed.abort();
      // Buffered so parallel builds don't interleave; only a failure needs the log. A child that never started
      // (pnpm missing) has no output, and its error, kept as the cause, says why.
      const { stdout = "", stderr = "" } = error as { stdout?: string; stderr?: string };
      logger.error(`${stdout}${stderr}`);
      throw new Error(`${slug}: slidev build failed`, { cause: error });
    }
    // Slidev's GitHub Pages fallback must not turn unknown paths into an app shell.
    await rm(join(dist, base, "404.html"), { force: true });
    logger.info(`${slug}: ${slides} slides`);
    results.set(slug, deckRewrites(slug, slides));
  }

  // The workers share one iterator, so each deck is taken by exactly one of them.
  const queue = slugs.values();
  await Promise.all(
    Array.from({ length: Math.min(availableParallelism(), slugs.length) }, async () => {
      for (const slug of queue) {
        if (failed.signal.aborted) {
          break;
        }
        await build(slug);
      }
    }),
  );
  return slugs.flatMap(slug => results.get(slug) ?? []);
}
