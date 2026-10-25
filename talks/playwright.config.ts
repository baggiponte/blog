import { defineConfig } from "@playwright/test";

const destination = process.env.TALKS_BASE_URL;
export default defineConfig({
  // Unit tests (scripts/*.test.ts) run with node --test instead.
  testDir: "tests",
  // Tests only read the site, so spread the per-deck tests across workers.
  fullyParallel: true,
  use: {
    baseURL: destination || "http://localhost:8790",
    // Full Chromium: the default headless shell denies Slidev's Wake Lock request.
    channel: "chromium",
  },
  webServer: destination ? undefined : {
    command: "pnpm exec cf dev --port 8790",
    url: "http://localhost:8790",
    reuseExistingServer: !process.env.CI,
  },
});
