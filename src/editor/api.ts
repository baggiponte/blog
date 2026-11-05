import type { IncomingMessage, ServerResponse } from "node:http";

import { createPost, EditorError, type Format, listPosts, readPost, saveImage, writePost } from "./posts.ts";

// The editor's API, mounted at apiBase (base.ts) on the dev server only:
//   GET  /posts                    every hand-written post
//   POST /posts                    a new bundle, from {title, description, format}
//   GET  /posts/<dir>              its index.md
//   PUT  /posts/<dir>              replaces its index.md
//   POST /posts/<dir>/images?name= saves the image in the body next to index.md

const maxBody = 50 * 1024 * 1024;

async function readBody(request: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += (chunk as Buffer).length;
    if (size > maxBody) {
      throw new EditorError(413, "files over 50 MB are not accepted");
    }
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

function send(response: ServerResponse, status: number, body?: unknown): void {
  response.statusCode = status;
  if (body === undefined) {
    response.end();
    return;
  }
  if (typeof body === "string") {
    response.setHeader("Content-Type", "text/markdown; charset=utf-8");
    response.end(body);
    return;
  }
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(body));
}

// The dev server only listens on this machine, but any page open in the browser can still send it requests:
// only the editor itself, on the same origin, may write.
function sameOrigin(request: IncomingMessage): boolean {
  const origin = request.headers.origin;
  return origin === undefined || new URL(origin).host === request.headers.host;
}

const isFormat = (value: unknown): value is Format => value === "post" || value === "tidbits";

async function route(blogDir: string, request: IncomingMessage, response: ServerResponse): Promise<void> {
  const url = new URL(request.url ?? "/", "http://editor");
  const [resource, dir, sub, ...rest] = url.pathname.split("/").filter(Boolean);
  const method = request.method ?? "GET";
  if (resource !== "posts" || rest.length > 0) {
    throw new EditorError(404, `no such route: ${url.pathname}`);
  }
  if (method !== "GET" && !sameOrigin(request)) {
    throw new EditorError(403, "only the editor page can write posts");
  }

  if (dir === undefined) {
    if (method === "GET") {
      return send(response, 200, await listPosts(blogDir));
    }
    if (method === "POST") {
      const input = JSON.parse((await readBody(request)).toString("utf8")) as Record<string, unknown>;
      const { title, description, format } = input;
      if (typeof title !== "string" || typeof description !== "string" || !isFormat(format)) {
        throw new EditorError(400, "a post needs a title, a description and a format");
      }
      return send(response, 201, { dir: await createPost(blogDir, { title, description, format }, new Date()) });
    }
  } else if (sub === undefined) {
    if (method === "GET") {
      return send(response, 200, await readPost(blogDir, dir));
    }
    if (method === "PUT") {
      await writePost(blogDir, dir, (await readBody(request)).toString("utf8"));
      return send(response, 204);
    }
  } else if (sub === "images" && method === "POST") {
    const name = url.searchParams.get("name") ?? "";
    return send(response, 201, { name: await saveImage(blogDir, dir, name, await readBody(request)) });
  }
  throw new EditorError(405, `${method} is not allowed on ${url.pathname}`);
}

// A Connect middleware for Vite's dev server, mounted at apiBase.
export function editorApi(blogDir: string) {
  return (request: IncomingMessage, response: ServerResponse): void => {
    route(blogDir, request, response).catch((error: unknown) => {
      // A SyntaxError is a request body that isn't JSON.
      let status = error instanceof SyntaxError ? 400 : 500;
      if (error instanceof EditorError) {
        status = error.status;
      }
      send(response, status, { error: error instanceof Error ? error.message : String(error) });
    });
  };
}
