import { apiBase } from "./base.ts";
import type { BundleDir, ImageName, NewPost, PostSummary } from "./post.ts";
import type { Stored, Version } from "./store.ts";

// The editor API (api.ts) as the page calls it: one typed function per route, so the page never builds a URL or
// reads a header itself. Only types come from post.ts and store.ts, so no Node code reaches the browser.

// A failed request, with the status kept so the page can tell a conflict from any other error.
export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
  // The post changed elsewhere since the page read it (api.ts answers 412).
  get conflict(): boolean {
    return this.status === 412;
  }
}

async function call(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${apiBase}${path}`, init);
  if (!response.ok) {
    const { error } = (await response.json().catch(() => ({ error: response.statusText }))) as { error: string };
    throw new ApiError(response.status, error);
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
