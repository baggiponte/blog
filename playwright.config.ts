import { defineConfig } from "@playwright/test";

// Set SITE_BASE_URL to test a deployed site; otherwise the tests serve the production build in dist/.
const destination = process.env.SITE_BASE_URL;
const local = "http://localhost:8790";
// The drafting editor only exists in `astro dev` (src/plugins/editor/), so its tests get a dev server of their own.
const dev = "http://localhost:4329";

export default defineConfig({
  // Unit tests (src/**/*.test.ts) run with node --test instead.
  testDir: "tests",
  // Tests only read the site, so spread the per-deck tests across workers.
  fullyParallel: true,
  // A test.only left in by mistake would silently skip the rest of the suite in CI.
  forbidOnly: Boolean(process.env.CI),
  use: {
    // Full Chromium: the default headless shell denies Slidev's Wake Lock request.
    channel: "chromium",
  },
  projects: [
    { name: "site", testIgnore: "editor.spec.ts", use: { baseURL: destination || local } },
    // A deployed site has no editor.
    ...(destination ? [] : [{ name: "editor", testMatch: "editor.spec.ts", use: { baseURL: dev } }]),
  ],
  ...(destination
    ? {}
    : {
        webServer: [
          {
            // `cf dev` would start Astro's dev server; this serves the production build in dist/ instead.
            command: "pnpm exec cf-wrangler dev --port 8790",
            env: { CLOUDFLARE_CF_AUTH: "true" },
            url: local,
            reuseExistingServer: !process.env.CI,
          },
          {
            // --ignore-lock: Astro allows one dev server per project, and the one you write in may already be running.
            command: "pnpm exec astro dev --port 4329 --ignore-lock",
            url: `${dev}/editor/`,
            reuseExistingServer: !process.env.CI,
          },
        ],
      }),
});
