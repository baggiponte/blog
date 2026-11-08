import { defineWranglerConfig } from "wrangler/experimental-config";

// Build adapter only; runtime configuration lives in cloudflare.config.ts.
export default defineWranglerConfig({
  assetsDirectory: "dist",
});
