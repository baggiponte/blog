import { defineConfig } from "@playwright/test";

// Set SITE_BASE_URL to test a deployed site; otherwise the tests serve the production build in dist/.
const destination = process.env.SITE_BASE_URL;
const local = "http://localhost:8790";

export default defineConfig({
  // Unit tests (src/**/*.test.ts) run with node --test instead.
  testDir: "tests",
  // Tests only read the site, so spread the per-deck tests across workers.
  fullyParallel: true,
  // A test.only left in by mistake would silently skip the rest of the suite in CI.
  forbidOnly: Boolean(process.env.CI),
  use: {
    baseURL: destination || local,
    // Full Chromium: the default headless shell denies Slidev's Wake Lock request.
    channel: "chromium",
  },
  ...(destination
    ? {}
    : {
        webServer: {
          // `cf dev` would start Astro's dev server; this serves the production build in dist/ instead.
          command: "pnpm exec cf-wrangler dev --port 8790",
          env: { CLOUDFLARE_CF_AUTH: "true" },
          url: local,
          reuseExistingServer: !process.env.CI,
        },
      }),
});
