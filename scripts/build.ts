import { spawnSync } from "node:child_process";
import { readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
function run(command: string, args: string[], env = process.env) {
  const result = spawnSync(command, args, { cwd: root, stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run("pnpm", ["exec", "astro", "build"]);

// Keep permanent directory redirects rather than the assets service's 307s.
const redirects: string[] = [];
async function visit(directory: string, url = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  if (url && entries.some((entry) => entry.name === "index.html" && entry.isFile())) {
    redirects.push(`${url} ${url}/ 301`);
  }
  for (const entry of entries) {
    if (entry.isDirectory() && !entry.name.startsWith("_")) await visit(join(directory, entry.name), `${url}/${entry.name}`);
  }
}
await visit(join(root, "dist"));
await writeFile(join(root, "dist/_redirects"), `${redirects.sort().join("\n")}\n`);

// `cf build` autodetects Astro and expects its Cloudflare adapter, so call the
// Wrangler delegate it would otherwise use to package dist/ as static assets.
// Deploy the result with `cf deploy --prebuilt`.
run("pnpm", ["exec", "cf-wrangler", "build"], { ...process.env, CLOUDFLARE_CF_AUTH: "true" });
