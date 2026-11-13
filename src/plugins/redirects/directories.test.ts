import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { directoryRedirects } from "./directories.ts";

test("every directory with a page redirects to its slashed URL, and nothing else does", async () => {
  const dist = await mkdtemp(join(tmpdir(), "dist-"));
  for (const page of ["index.html", "about/index.html", "talks/pymi/index.html", "_astro/index.html"]) {
    await mkdir(join(dist, page, ".."), { recursive: true });
    await writeFile(join(dist, page), "");
  }
  await mkdir(join(dist, "images"));
  await writeFile(join(dist, "images", "cover.jpg"), "");
  // /talks has no page of its own here, the site root needs no redirect and _astro is Astro's.
  assert.deepEqual(await directoryRedirects(dist), ["/about /about/ 301", "/talks/pymi /talks/pymi/ 301"]);
});
