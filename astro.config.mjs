import { unified } from "@astrojs/markdown-remark";
import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";

import { codeFigure } from "./src/lib/code-figure.mjs";
import { remarkMark } from "./src/lib/remark-mark.mjs";

export default defineConfig({
  site: "https://baggiponte.com",
  trailingSlash: "always",
  integrations: [sitemap()],
  markdown: {
    // unified() rather than the default Sätteri so `==highlights==` can be a small remark plugin.
    processor: unified({ remarkPlugins: [remarkMark] }),
    shikiConfig: {
      themes: { light: "vitesse-light", dark: "vitesse-dark" },
      defaultColor: false,
      transformers: [codeFigure()],
    },
  },
});
