import { defineConfig } from "cf/config";

export default defineConfig({
  accountId: "3dd4d1b3677089482ceefbf6c3c06e1f",
  worker: {
    // Deliberately isolated from the live Pages site: no domains or zone routes.
    name: "blog-preview",
    compatibilityDate: "2026-10-24",
    workersDev: true,
    previewUrls: false,
    assets: {
      htmlHandling: "auto-trailing-slash",
      notFoundHandling: "404-page",
    },
  },
});
