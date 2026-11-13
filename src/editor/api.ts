import { EditorError, type Failure } from "./failure.ts";
import {
  checkPost,
  dirForTitle,
  newPostMarkdown,
  parseBundleDir,
  parseImageName,
  parseNewPost,
  type Version,
} from "./post.ts";
import type { PostStore } from "./store.ts";
import { parseSuggestions } from "./suggestion.ts";

// The editor's API, mounted at apiBase (base.ts) on the dev server only:
//   GET  /posts                     every hand-written post
//   POST /posts                     a new bundle, from {title, description, format}
//   GET  /posts/<dir>               its index.md, with its version as the ETag
//   PUT  /posts/<dir>               replaces its index.md if If-Match is still its version; answers the new ETag
//   POST /posts/<dir>/images?name=  saves the image in the body next to index.md
//   GET  /posts/<dir>/images/<name> an image next to index.md, for the editor to show inline
//   GET  /posts/<dir>/suggestions   the proofreading suggestions left in suggestions.json, with its version as the ETag
//   PUT  /posts/<dir>/suggestions   the ones still open, if If-Match is still its version; none left removes the file
//
// It takes a web Request and gives back a Response, like fetch: no server is needed to call it, so the tests call it
// directly, and the dev server adapts Node's request to it (integration.ts).

// Bigger than any screenshot or plot a post should carry.
export const maxUpload = 20 * 1024 * 1024;

// Each failure the editor knows about, as the HTTP status the page sees. A switch with no default makes the type
// checker and the linter flag a new Failure that has no status yet.
function statusOf(failure: Failure): number {
  switch (failure) {
    case "invalid":
      return 400;
    case "missing":
      return 404;
    case "exists":
      return 409;
    case "conflict":
      // If-Match no longer matches: the standard answer is 412 Precondition Failed.
      return 412;
    case "too-large":
      return 413;
    case "unsupported":
      return 415;
  }
}

// What a failed request answers: the message to show, and the editor's reason when there is one.
export interface ErrorBody {
  error: string;
  failure?: Failure;
}

class HttpError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const json = (status: number, body: unknown, headers: HeadersInit = {}): Response =>
  Response.json(body, { status, headers });

// The body, read in chunks so a huge upload is cut off at the limit instead of filling memory.
async function bodyOf(request: Request): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  let size = 0;
  for await (const chunk of request.body ?? []) {
    size += chunk.length;
    if (size > maxUpload) {
      throw new EditorError(`files over ${maxUpload / 1024 / 1024} MB are not accepted`, { failure: "too-large" });
    }
    chunks.push(chunk);
  }
  const body = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.length;
  }
  return body;
}

const textOf = async (request: Request): Promise<string> => new TextDecoder().decode(await bodyOf(request));

// The dev server only listens on this machine, but any page open in the browser can still send it requests: only
// the editor itself, on the same origin, may write.
function sameOrigin(request: Request): boolean {
  const origin = request.headers.get("Origin");
  return origin === null || new URL(origin).host === new URL(request.url).host;
}

// ETags are quoted strings; a browser sends back exactly what it was given.
const etag = (version: Version): string => `"${version}"`;
const ifMatch = (request: Request): Version | undefined =>
  (request.headers.get("If-Match")?.replace(/^"|"$/g, "") as Version | undefined) || undefined;

async function route(store: PostStore, request: Request, now: () => Date): Promise<Response> {
  const url = new URL(request.url);
  const [resource, dirText, sub, file, ...rest] = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const method = request.method;
  if (resource !== "posts" || rest.length > 0) {
    throw new HttpError(404, `no such route: ${url.pathname}`);
  }
  if (method !== "GET" && !sameOrigin(request)) {
    throw new HttpError(403, "only the editor page can write posts");
  }

  if (dirText === undefined) {
    if (method === "GET") {
      return json(200, await store.list());
    }
    if (method === "POST") {
      const post = parseNewPost(JSON.parse(await textOf(request)));
      const dir = dirForTitle(post.title);
      await store.create(dir, newPostMarkdown(post, now()));
      return json(201, { dir });
    }
  } else if (sub === undefined) {
    const dir = parseBundleDir(dirText);
    if (method === "GET") {
      const { markdown, version } = await store.read(dir);
      return new Response(markdown, {
        headers: { "Content-Type": "text/markdown; charset=utf-8", ETag: etag(version) },
      });
    }
    if (method === "PUT") {
      const markdown = await textOf(request);
      checkPost(markdown);
      const version = await store.write(dir, markdown, ifMatch(request));
      return new Response(null, { status: 204, headers: { ETag: etag(version) } });
    }
  } else if (sub === "images" && file !== undefined) {
    if (method === "GET") {
      const { bytes, type } = await store.readImage(parseBundleDir(dirText), parseImageName(file));
      return new Response(bytes, { headers: { "Content-Type": type, "Cache-Control": "no-cache" } });
    }
  } else if (sub === "suggestions" && file === undefined) {
    const dir = parseBundleDir(dirText);
    if (method === "GET") {
      const { suggestions, version } = await store.readSuggestions(dir);
      return json(200, { suggestions }, { ETag: etag(version) });
    }
    if (method === "PUT") {
      const suggestions = parseSuggestions(JSON.parse(await textOf(request)));
      const version = await store.writeSuggestions(dir, suggestions, ifMatch(request));
      return new Response(null, { status: 204, headers: { ETag: etag(version) } });
    }
  } else if (sub === "images" && method === "POST") {
    const dir = parseBundleDir(dirText);
    const name = await store.addImage(dir, url.searchParams.get("name") ?? "", await bodyOf(request));
    return json(201, { name });
  }
  throw new HttpError(405, `${method} is not allowed on ${url.pathname}`);
}

// The API over a store. `now` is the date a new post gets; tests pass a fixed one.
export function editorApi(
  store: PostStore,
  now: () => Date = () => new Date(),
): (request: Request) => Promise<Response> {
  return async request => {
    try {
      return await route(store, request, now);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (error instanceof EditorError) {
        return json(statusOf(error.failure), { error: message, failure: error.failure } satisfies ErrorBody);
      }
      let status = 500;
      if (error instanceof HttpError) {
        status = error.status;
      } else if (error instanceof SyntaxError || error instanceof URIError) {
        // A body that isn't JSON, or a path that isn't URL-encoded.
        status = 400;
      }
      return json(status, { error: message } satisfies ErrorBody);
    }
  };
}
