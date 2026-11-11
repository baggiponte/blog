import assert from "node:assert/strict";
import { test } from "node:test";

import { editorApi, maxUpload } from "./api.ts";
import { type BundleDir, EditorError, type ImageName, type PostSummary } from "./post.ts";
import { type PostStore, type Version, versionOf } from "./store.ts";

// The API alone, over posts kept in memory: these tests check what the API answers, and store.test.ts checks the
// files on disk.
function memoryStore(posts: Record<string, string>): PostStore & { posts: Map<string, string> } {
  const files = new Map(Object.entries(posts));
  const find = (dir: BundleDir): string => {
    const markdown = files.get(dir);
    if (markdown === undefined) {
      throw new EditorError(`no ${dir}`, { failure: "missing" });
    }
    return markdown;
  };
  const save = (dir: BundleDir, markdown: string, expected?: Version): Version => {
    if (expected !== undefined && versionOf(find(dir)) !== expected) {
      throw new EditorError("changed", { failure: "conflict" });
    }
    files.set(dir, markdown);
    return versionOf(markdown);
  };
  return {
    posts: files,
    list: async () =>
      [...files.keys()].map(dir => ({ dir, title: dir, date: "", draft: false, url: `/blog/${dir}/` }) as PostSummary),
    read: async dir => ({ markdown: find(dir), version: versionOf(find(dir)) }),
    write: async (dir, markdown, expected) => save(dir, markdown, expected),
    create: async () => "new-post" as BundleDir,
    addImage: async (_dir, original) => original as ImageName,
    readImage: async () => ({ bytes: new Uint8Array([1, 2]), type: "image/png" }),
  };
}

const valid = '---\ntitle: "hello"\ndescription: "about it"\ndate: 2026-11-10\n---\n\nbody\n';
const origin = "http://localhost:4321";
const request = (path: string, init: RequestInit = {}): Request => new Request(`${origin}${path}`, init);
const write = (body: BodyInit, headers: Record<string, string> = {}): RequestInit => ({
  method: "PUT",
  body,
  headers: { Origin: origin, ...headers },
});

test("a post is read with its version, and saved only over that version", async () => {
  const store = memoryStore({ hello: valid });
  const api = editorApi(store);
  const read = await api(request("/posts/hello"));
  assert.equal(await read.text(), valid);
  const etag = read.headers.get("ETag") ?? "";
  assert.equal(etag, `"${versionOf(valid)}"`);

  const changed = valid.replace("body", "more");
  const saved = await api(request("/posts/hello", write(changed, { "If-Match": etag })));
  assert.equal(saved.status, 204);
  assert.equal(saved.headers.get("ETag"), `"${versionOf(changed)}"`);
  assert.equal(store.posts.get("hello"), changed);

  // The old version again: someone saved in between.
  const stale = await api(request("/posts/hello", write(valid, { "If-Match": etag })));
  assert.equal(stale.status, 412);
  assert.equal(store.posts.get("hello"), changed);
});

test("a post the site couldn't build is never saved", async () => {
  const store = memoryStore({ hello: valid });
  const response = await editorApi(store)(request("/posts/hello", write('---\ntitle: "x\n---\n')));
  assert.equal(response.status, 400);
  assert.match(((await response.json()) as { error: string }).error, /isn't valid YAML/);
  assert.equal(store.posts.get("hello"), valid);
});

test("other sites open in the browser can't write", async () => {
  const store = memoryStore({ hello: valid });
  const response = await editorApi(store)(
    request("/posts/hello", { method: "PUT", body: valid, headers: { Origin: "https://evil.example" } }),
  );
  assert.equal(response.status, 403);
});

test("a new post answers with its folder, dated by the clock it's given", async () => {
  let given: Date | undefined;
  const store: PostStore = {
    ...memoryStore({}),
    create: (_post, date) => {
      given = date;
      return Promise.resolve("x" as BundleDir);
    },
  };
  const now = new Date("2026-05-05");
  const response = await editorApi(
    store,
    () => now,
  )(request("/posts", { method: "POST", body: JSON.stringify({ title: "x", description: "", format: "post" }) }));
  assert.equal(response.status, 201);
  assert.deepEqual(await response.json(), { dir: "x" });
  assert.equal(given, now);
});

test("uploads over the limit are cut off", async () => {
  const api = editorApi(memoryStore({ hello: valid }));
  const ok = await api(request("/posts/hello/images?name=a.png", { method: "POST", body: new Uint8Array(10) }));
  assert.equal(ok.status, 201);
  const big = await api(
    request("/posts/hello/images?name=a.png", { method: "POST", body: new Uint8Array(maxUpload + 1) }),
  );
  assert.equal(big.status, 413);
});

test("each failure gets its own status, and nothing crashes", async () => {
  const api = editorApi(memoryStore({ hello: valid }));
  const status = async (path: string, init?: RequestInit): Promise<number> => (await api(request(path, init))).status;
  assert.equal(await status("/posts", { method: "POST", body: "{not json" }), 400);
  assert.equal(await status("/posts", { method: "POST", body: JSON.stringify({ title: "x" }) }), 400);
  assert.equal(await status("/posts/missing"), 404);
  assert.equal(await status("/posts/hello", { method: "DELETE" }), 405);
  assert.equal(await status("/nothing"), 404);
  assert.equal(await status("/posts/hello/images/shot.png"), 200);
  for (const sneaky of ["index.md", "..%2Findex.md", "%2E%2E%2F..%2Fsecret.png", ".hidden.png", "%E0%A4%A"]) {
    assert.equal(await status(`/posts/hello/images/${sneaky}`), 400, sneaky);
  }
  assert.equal(await status("/posts/..%2Fetc"), 400);
});
