import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";
import { markdown } from "./src/lib/markdown.ts";
import { decks } from "./src/plugins/decks/integration.ts";
import { editor } from "./src/plugins/editor/integration.ts";
import { fontsource } from "./src/plugins/fonts/fontsource.ts";
import { permanentRedirects } from "./src/plugins/redirects/integration.ts";

export default defineConfig({
  site: "https://baggiponte.com",
  trailingSlash: "always",
  // Astro runs their build hooks in this order: permanentRedirects() comes after decks() so the decks get theirs.
  integrations: [sitemap(), editor(), decks(), permanentRedirects()],
  markdown,
  // Astro's fonts, from the Fontsource packages the slides theme and the editor also use: Astro serves the files, preloads
  // the latin ones and sizes a system-font fallback to their metrics, so text doesn't jump when they arrive.
  fonts: [
    {
      provider: fontsource("@fontsource-variable/instrument-sans"),
      name: "Instrument Sans Variable",
      cssVariable: "--font-sans",
      weights: ["400 700"],
      subsets: ["latin", "latin-ext"],
      fallbacks: ["system-ui", "-apple-system", "sans-serif"],
      // Astro sizes its Arial fallback from these files' metrics to 135%, far wider than the real font: text then
      // jumps more when the font arrives, not less.
      optimizedFallbacks: false,
    },
    {
      provider: fontsource("@fontsource-variable/bricolage-grotesque"),
      name: "Bricolage Grotesque Variable",
      cssVariable: "--font-display",
      weights: ["200 800"],
      subsets: ["latin", "latin-ext"],
      fallbacks: ["system-ui", "-apple-system", "sans-serif"],
      // Astro sizes its Arial fallback from these files' metrics to 135%, far wider than the real font: text then
      // jumps more when the font arrives, not less.
      optimizedFallbacks: false,
    },
  ],
});
