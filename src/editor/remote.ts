import type { ErrorBody } from "./api.ts";
import { apiBase } from "./base.ts";
import type { Failure } from "./failure.ts";
import type { BundleDir, ImageName, NewPost, PostSummary, Version } from "./post.ts";
import type { Stored, Suggestions } from "./store.ts";
import type { Suggestion } from "./suggestion.ts";

// The editor API (api.ts) as the page calls it: one typed function per route, so the page never builds a URL or
// reads a header itself. Only types come from the server's modules, so no Node code reaches the browser.

// A failed request. `failure` is the editor's own reason, as the API sent it, so the page tells a conflict from any
// other error by the same Failure type the server threw, not by an HTTP status; requests the API turned down before
// reaching a post (an unknown route, another origin) have none.
export class ApiError extends Error {
  readonly failure: Failure | undefined;
  constructor({ error, failure }: ErrorBody) {
    super(error);
    this.failure = failure;
  }
}

async function call(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${apiBase}${path}`, init);
  if (!response.ok) {
    throw new ApiError(await response.json().catch((): ErrorBody => ({ error: response.statusText })));
  }
  return response;
}

const versionIn = (response: Response): Version =>
  (response.headers.get("ETag") ?? "").replace(/^"|"$/g, "") as Version;

const postPath = (dir: BundleDir): string => `/posts/${dir}`;

export const listPosts = async (): Promise<PostSummary[]> => (await call("/posts")).json() as Promise<PostSummary[]>;

export async function readPost(dir: BundleDir): Promise<Stored> {
  const response = await call(postPath(dir));
  return { markdown: await response.text(), version: versionIn(response) };
}

// The init for saving `markdown` over `version`; also used as is by the save on closing the tab (client.ts).
export const saveInit = (markdown: string, version: Version): RequestInit => ({
  method: "PUT",
  body: markdown,
  headers: { "If-Match": `"${version}"` },
});

export const writePost = async (dir: BundleDir, markdown: string, version: Version): Promise<Version> =>
  versionIn(await call(postPath(dir), saveInit(markdown, version)));

export const postUrl = (dir: BundleDir): string => `${apiBase}${postPath(dir)}`;

export async function createPost(post: NewPost): Promise<BundleDir> {
  const { dir } = (await (await call("/posts", { method: "POST", body: JSON.stringify(post) })).json()) as {
    dir: BundleDir;
  };
  return dir;
}

export async function addImage(dir: BundleDir, file: File): Promise<ImageName> {
  const original = file.name || `pasted.${file.type.split("/")[1] ?? "png"}`;
  const response = await call(`${postPath(dir)}/images?name=${encodeURIComponent(original)}`, {
    method: "POST",
    body: file,
  });
  return ((await response.json()) as { name: ImageName }).name;
}

export async function readSuggestions(dir: BundleDir): Promise<Suggestions> {
  const response = await call(`${postPath(dir)}/suggestions`);
  const { suggestions } = (await response.json()) as { suggestions: Suggestion[] };
  return { suggestions, version: versionIn(response) };
}

export const writeSuggestions = async (dir: BundleDir, suggestions: Suggestion[], version: Version): Promise<Version> =>
  versionIn(
    await call(`${postPath(dir)}/suggestions`, {
      method: "PUT",
      body: JSON.stringify({ suggestions }),
      headers: { "If-Match": `"${version}"` },
    }),
  );
