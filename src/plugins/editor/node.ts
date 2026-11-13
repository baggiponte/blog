import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";

// Vite's dev server speaks Node's request and response objects; the editor API speaks web Request and Response
// (api.ts). These two functions translate, so the API itself never imports anything from Node's HTTP server.

export function toRequest(incoming: IncomingMessage): Request {
  const url = new URL(incoming.url ?? "/", `http://${incoming.headers.host ?? "localhost"}`);
  const headers = new Headers();
  for (const [name, value] of Object.entries(incoming.headers)) {
    for (const each of Array.isArray(value) ? value : [value]) {
      if (each !== undefined) {
        headers.append(name, each);
      }
    }
  }
  const method = incoming.method ?? "GET";
  const hasBody = method !== "GET" && method !== "HEAD";
  return new Request(url, {
    method,
    headers,
    // Streamed, not buffered: the API stops reading an upload once it's over the limit.
    ...(hasBody ? { body: Readable.toWeb(incoming) as ReadableStream<Uint8Array>, duplex: "half" } : {}),
  });
}

export async function send(outgoing: ServerResponse, response: Response): Promise<void> {
  outgoing.writeHead(response.status, Object.fromEntries(response.headers));
  outgoing.end(new Uint8Array(await response.arrayBuffer()));
}
