import { defineConfig } from "cf/config";

export default defineConfig({
  accountId: "3dd4d1b3677089482ceefbf6c3c06e1f",
  worker: {
    name: "talks",
    compatibilityDate: "2026-10-24",
    domains: ["talks.baggiponte.com"],
    workersDev: false,
    previewUrls: false,
    observability: {
      enabled: true,
      logs: { enabled: true },
      traces: { enabled: true, headSamplingRate: 0.01 },
    },
  },
});
