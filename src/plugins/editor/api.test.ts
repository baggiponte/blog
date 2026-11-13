import assert from "node:assert/strict";
import { test } from "node:test";

import { editorApi, maxUpload } from "./api.ts";
import { EditorError } from "./failure.ts";
import { type BundleDir, type ImageName, isConflict, type PostSummary, type Version } from "./post.ts";
import { type PostStore, versionOf } from "./store.ts";
import { type Suggestion, suggestionsJson } from "./suggestion.ts";

// The API alone, over posts kept in memory: these tests check what the API answers, and store.test.ts checks the
// files on disk. The store's one rule, when a save is a conflict, is the same isConflict both stores use.
function memoryStore(
  posts: Record<string, string>,
): PostStore & { posts: Map<string, string>; suggestions: Map<string, Suggestion[]> } {
  const files = new Map(Object.entries(posts));
  const suggestions = new Map<string, Suggestion[]>();
  const suggestionsVersion = (dir: string): Version => {
    const list = suggestions.get(dir) ?? [];
    return versionOf(list.length === 0 ? "" : suggestionsJson(list));
  };
  const find = (dir: BundleDir): string => {
    const markdown = files.get(dir);
    if (markdown === undefined) {
      throw new EditorError(`no ${dir}`, { failure: "missing" });
    }
    return markdown;
  };
  const save = (dir: BundleDir, markdown: string, expected?: Version): Version => {
    if (isConflict(versionOf(find(dir)), expected, versionOf(markdown))) {
      throw new EditorError("changed", { failure: "conflict" });
    }
    files.set(dir, markdown);
    return versionOf(markdown);
  };
  const readSuggestions = (dir: BundleDir) => {
    find(dir);
    return { suggestions: suggestions.get(dir) ?? [], version: suggestionsVersion(dir) };
  };
  const saveSuggestions = (dir: BundleDir, list: Suggestion[], expected?: Version): Version => {
    const incoming = versionOf(list.length === 0 ? "" : suggestionsJson(list));
    if (isConflict(suggestionsVersion(dir), expected, incoming)) {
      throw new EditorError("changed", { failure: "conflict" });
    }
    suggestions.set(dir, list);
    return incoming;
  };
  return {
    posts: files,
    list: async () =>
      [...files.keys()].map(dir => ({ dir, title: dir, date: "", draft: false, url: `/blog/${dir}/` }) as PostSummary),
    read: async dir => ({ markdown: find(dir), version: versionOf(find(dir)) }),
    write: async (dir, markdown, expected) => save(dir, markdown, expected),
    create: async (dir, markdown) => void files.set(dir, markdown),
    addImage: async (_dir, original) => original as ImageName,
    readImage: async () => ({ bytes: new Uint8Array([1, 2]), type: "image/png" }),
    suggestions,
    readSuggestions: async dir => readSuggestions(dir),
    writeSuggestions: async (dir, list, expected) => saveSuggestions(dir, list, expected),
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
  assert.deepEqual(await stale.json(), { error: "changed", failure: "conflict" });
  // Saving the text already there is never a conflict, whatever version the page holds.
  assert.equal((await api(request("/posts/hello", write(changed, { "If-Match": etag })))).status, 204);
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

test("a new post is a draft named after its title, dated by the clock it's given", async () => {
  const store = memoryStore({});
  const body = JSON.stringify({ title: "Hello World", description: "", format: "post" });
  const response = await editorApi(store, () => new Date("2026-05-05"))(request("/posts", { method: "POST", body }));
  assert.equal(response.status, 201);
  assert.deepEqual(await response.json(), { dir: "hello-world" });
  assert.match(store.posts.get("hello-world") ?? "", /^date: 2026-05-05$/m);
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

test("suggestions are read with their version, and saved back only over that version", async () => {
  const store = memoryStore({ hello: valid });
  const typo: Suggestion = { id: "s1", kind: "typo", quote: "body", replacement: "Body" };
  store.suggestions.set("hello", [typo]);
  const api = editorApi(store);

  const read = await api(request("/posts/hello/suggestions"));
  assert.equal(read.status, 200);
  assert.deepEqual(await read.json(), { suggestions: [typo] });
  const version = read.headers.get("ETag") ?? "";

  // Claude writes new ones in the meantime: the page's save is refused, so it reloads them instead of dropping them.
  store.suggestions.set("hello", [typo, { ...typo, id: "s2" }]);
  const stale = await api(request("/posts/hello/suggestions", write('{"suggestions":[]}', { "If-Match": version })));
  assert.equal(stale.status, 412);

  const fresh = (await api(request("/posts/hello/suggestions"))).headers.get("ETag") ?? "";
  const done = await api(request("/posts/hello/suggestions", write('{"suggestions":[]}', { "If-Match": fresh })));
  assert.equal(done.status, 204);
  assert.deepEqual(store.suggestions.get("hello"), []);
});

test("a suggestions file that breaks the format is refused with the reason", async () => {
  const api = editorApi(memoryStore({ hello: valid }));
  const answer = await api(request("/posts/hello/suggestions", write('{"suggestions":[{"id":"x"}]}')));
  assert.equal(answer.status, 400);
  assert.match(((await answer.json()) as { error: string }).error, /suggestions\.0\.kind/);
});
