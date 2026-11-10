import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";

import { editorApi } from "./api.ts";

let blogDir: string;
let base: string;
const server = createServer((request, response) => {
  editorApi(blogDir)(request, response);
});

before(async () => {
  blogDir = await mkdtemp(join(tmpdir(), "editor-api-"));
  await mkdir(join(blogDir, "hello"));
  await writeFile(join(blogDir, "hello", "index.md"), '---\ntitle: "hello"\n---\n');
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(() => {
  server.close();
});

test("posts round-trip through the API", async () => {
  const list = await (await fetch(`${base}/posts`)).json();
  assert.deepEqual(list, [{ dir: "hello", title: "hello", date: "", draft: false, url: "/blog/hello/" }]);
  const put = await fetch(`${base}/posts/hello`, { method: "PUT", body: "changed", headers: { Origin: base } });
  assert.equal(put.status, 204);
  assert.equal(await (await fetch(`${base}/posts/hello`)).text(), "changed");
});

test("an image upload lands next to index.md", async () => {
  const response = await fetch(`${base}/posts/hello/images?name=My%20Plot.png`, {
    method: "POST",
    body: new Uint8Array([137, 80, 78, 71]),
  });
  assert.equal(response.status, 201);
  assert.deepEqual(await response.json(), { name: "my-plot.png" });
  assert.deepEqual([...(await readFile(join(blogDir, "hello", "my-plot.png")))], [137, 80, 78, 71]);
});

test("other sites open in the browser can't write", async () => {
  const response = await fetch(`${base}/posts/hello`, {
    method: "PUT",
    body: "pwned",
    headers: { Origin: "https://evil.example" },
  });
  assert.equal(response.status, 403);
  assert.notEqual(await readFile(join(blogDir, "hello", "index.md"), "utf8"), "pwned");
});

test("bad requests get an error, not a crash", async () => {
  const bad = await fetch(`${base}/posts`, { method: "POST", body: "{not json" });
  assert.equal(bad.status, 400);
  const incomplete = await fetch(`${base}/posts`, { method: "POST", body: JSON.stringify({ title: "x" }) });
  assert.equal(incomplete.status, 400);
  assert.equal((await fetch(`${base}/posts/hello`, { method: "DELETE" })).status, 405);
  assert.equal((await fetch(`${base}/nothing`)).status, 404);
});

test("images next to index.md are served for the editor, nothing else is", async () => {
  await writeFile(join(blogDir, "hello", "shot.png"), new Uint8Array([1, 2]));
  const image = await fetch(`${base}/posts/hello/images/shot.png`);
  assert.equal(image.headers.get("content-type"), "image/png");
  assert.deepEqual([...new Uint8Array(await image.arrayBuffer())], [1, 2]);
  assert.equal((await fetch(`${base}/posts/hello/images/missing.png`)).status, 404);
  for (const sneaky of ["index.md", "..%2Findex.md", "%2E%2E%2F..%2Fsecret.png", ".hidden.png", "%E0%A4%A"]) {
    assert.equal((await fetch(`${base}/posts/hello/images/${sneaky}`)).status, 400, sneaky);
  }
});
