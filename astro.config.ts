import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";

import { editor } from "./src/editor/integration.ts";
import { markdown } from "./src/lib/markdown.ts";

export default defineConfig({
  site: "https://baggiponte.com",
  trailingSlash: "always",
  integrations: [sitemap(), editor()],
  markdown,
});
