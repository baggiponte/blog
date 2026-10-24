import { defineConfig, triggers } from "cf/config";

export default defineConfig(({ mode }) => {
  const production = mode === "production";
  return {
    accountId: "3dd4d1b3677089482ceefbf6c3c06e1f",
    worker: {
      name: production ? "blog" : "blog-preview",
      compatibilityDate: "2026-10-24",
      workersDev: !production,
      previewUrls: false,
      // Retain GitHub's DNS targets as fallback; never match sibling subdomains.
      triggers: production
        ? [triggers.fetch({ pattern: "baggiponte.com/*", zone: "baggiponte.com" })]
        : [],
      assets: {
        htmlHandling: "auto-trailing-slash",
        notFoundHandling: "404-page",
      },
    },
  };
});
