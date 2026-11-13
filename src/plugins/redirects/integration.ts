import { appendFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { AstroIntegration } from "astro";

import { directoryRedirects } from "./directories.ts";

// Writes the permanent directory redirects to dist/_redirects. List it after every integration that adds pages to
// dist/ (decks), since it only sees the directories already there.
export function permanentRedirects(): AstroIntegration {
  return {
    name: "permanent-redirects",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        const dist = fileURLToPath(dir);
        const lines = await directoryRedirects(dist);
        await appendFile(join(dist, "_redirects"), lines.map(line => `${line}\n`).join(""));
      },
    },
  };
}
