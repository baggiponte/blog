import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";

import { markdown } from "./src/lib/markdown.ts";

export default defineConfig({
  site: "https://baggiponte.com",
  trailingSlash: "always",
  integrations: [sitemap()],
  markdown,
});
