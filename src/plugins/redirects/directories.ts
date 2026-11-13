import { readdir } from "node:fs/promises";
import { join } from "node:path";

// A permanent redirect from /x to /x/ for every directory of the build that holds an index.html: Cloudflare's assets
// service would answer those with a temporary 307. Directories starting with "_" (Astro's own assets) are skipped.
export async function directoryRedirects(dist: string): Promise<string[]> {
  const redirects: string[] = [];
  async function visit(directory: string, url: string): Promise<void> {
    const entries = await readdir(directory, { withFileTypes: true });
    if (url !== "" && entries.some(entry => entry.name === "index.html" && entry.isFile())) {
      redirects.push(`${url} ${url}/ 301`);
    }
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith("_")) {
        await visit(join(directory, entry.name), `${url}/${entry.name}`);
      }
    }
  }
  await visit(dist, "");
  return redirects.sort();
}
