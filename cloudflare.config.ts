import { defineConfig } from "cf/config";

export default defineConfig({
  accountId: "3dd4d1b3677089482ceefbf6c3c06e1f",
  worker: {
    name: "blog",
    compatibilityDate: "2026-10-24",
    domains: ["baggiponte.com", "www.baggiponte.com"],
    workersDev: false,
    previewUrls: false,
    assets: {
      htmlHandling: "auto-trailing-slash",
      notFoundHandling: "404-page",
    },
  },
});
