import { defineConfig } from "@playwright/test";

// Set SITE_BASE_URL to test a deployed site; otherwise the tests serve the production build in dist/.
const destination = process.env.SITE_BASE_URL;
const local = "http://localhost:8790";
// The same for tools.baggiponte.com (TOOLS_BASE_URL) and its build in tools/dist/; tests/tools.spec.ts reads it.
const tools = process.env.TOOLS_BASE_URL;
export const toolsLocal = "http://localhost:8791";

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
  webServer: [
    ...(destination
      ? []
      : [
          {
            // `cf dev` would start Astro's dev server; this serves the production build in dist/ instead.
            command: "pnpm exec cf-wrangler dev --port 8790",
            env: { CLOUDFLARE_CF_AUTH: "true" },
            url: local,
            reuseExistingServer: !process.env.CI,
          },
        ]),
    ...(tools
      ? []
      : [
          {
            // Serves tools/dist/, which `pnpm tools:build` writes.
            command: "pnpm exec cf-wrangler dev --port 8791",
            cwd: "tools",
            env: { CLOUDFLARE_CF_AUTH: "true" },
            url: toolsLocal,
            reuseExistingServer: !process.env.CI,
          },
        ]),
  ],
});
