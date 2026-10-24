import { defineConfig } from "@playwright/test";

const destination = process.env.TALKS_BASE_URL;
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  use: {
    baseURL: destination || "http://localhost:8790",
    browserName: "chromium",
    channel: process.env.PLAYWRIGHT_CHANNEL || "chromium",
    viewport: { width: 1440, height: 1000 },
    trace: "retain-on-failure",
  },
  webServer: destination ? undefined : {
    command: "pnpm exec cf dev --port 8790",
    url: "http://localhost:8790",
    reuseExistingServer: !process.env.CI,
  },
});
