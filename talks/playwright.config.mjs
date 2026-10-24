import { defineConfig } from "@playwright/test";

const destination = process.env.TALKS_BASE_URL;
export default defineConfig({
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
