import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";
import { markdown } from "./src/lib/markdown.ts";
import { decks } from "./src/plugins/decks/integration.ts";
import { editor } from "./src/plugins/editor/integration.ts";
import { permanentRedirects } from "./src/plugins/redirects/integration.ts";

export default defineConfig({
  site: "https://baggiponte.com",
  trailingSlash: "always",
  // Astro runs their build hooks in this order: permanentRedirects() comes after decks() so the decks get theirs.
  integrations: [sitemap(), editor(), decks(), permanentRedirects()],
  markdown,
});
