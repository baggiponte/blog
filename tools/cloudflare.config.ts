import { defineConfig } from "cf/config";

// tools.baggiponte.com: static pages built into tools/dist/ by tools/build.ts.
export default defineConfig({
  accountId: "3dd4d1b3677089482ceefbf6c3c06e1f",
  worker: {
    name: "tools",
    compatibilityDate: "2026-10-24",
    domains: ["tools.baggiponte.com"],
    workersDev: false,
    previewUrls: false,
    assets: {
      htmlHandling: "auto-trailing-slash",
      notFoundHandling: "404-page",
    },
  },
});
