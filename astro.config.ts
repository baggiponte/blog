import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";

import { editor } from "./src/editor/integration.ts";
import { markdown } from "./src/lib/markdown.ts";
import { decks } from "./src/plugins/decks/integration.ts";
import { permanentRedirects } from "./src/plugins/redirects/integration.ts";

export default defineConfig({
  site: "https://baggiponte.com",
  trailingSlash: "always",
  // Astro runs their build hooks in this order: permanentRedirects() comes after decks() so the decks get theirs.
  integrations: [sitemap(), editor(), decks(), permanentRedirects()],
  markdown,
});
