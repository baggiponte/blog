import { spawnSync } from "node:child_process";
import { readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const result = spawnSync("hugo", [
  "--cleanDestinationDir", "--gc", "--minify",
], { cwd: root, stdio: "inherit", env: { ...process.env, HUGO_ENVIRONMENT: "production" } });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);

// Keep Pages' permanent directory redirects rather than the assets service's 307s.
const redirects = [];
async function visit(directory, url = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  if (url && entries.some((entry) => entry.name === "index.html" && entry.isFile())) {
    redirects.push(`${url} ${url}/ 301`);
  }
  for (const entry of entries) {
    if (entry.isDirectory()) await visit(join(directory, entry.name), `${url}/${entry.name}`);
  }
}
await visit(join(root, "public"));
await writeFile(join(root, "public/_redirects"), `${redirects.sort().join("\n")}\n`);
