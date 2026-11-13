import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";

import { send, toRequest } from "./node.ts";

test("a Node request reaches a web handler whole, and its answer comes back", async () => {
  let seen: { method: string; url: string; header: string | null; body: string } | undefined;
  const server = createServer((incoming, outgoing) => {
    const request = toRequest(incoming);
    void request.text().then(async body => {
      seen = { method: request.method, url: request.url, header: request.headers.get("If-Match"), body };
      await send(outgoing, new Response("done", { status: 201, headers: { ETag: '"v2"' } }));
    });
  });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  try {
    const response = await fetch(`http://127.0.0.1:${port}/posts/hello?x=1`, {
      method: "PUT",
      body: "text",
      headers: { "If-Match": '"v1"' },
    });
    assert.equal(response.status, 201);
    assert.equal(response.headers.get("ETag"), '"v2"');
    assert.equal(await response.text(), "done");
    assert.deepEqual(seen, {
      method: "PUT",
      url: `http://127.0.0.1:${port}/posts/hello?x=1`,
      header: '"v1"',
      body: "text",
    });
  } finally {
    server.close();
  }
});
