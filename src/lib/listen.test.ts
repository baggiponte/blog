import assert from "node:assert/strict";
import { test } from "node:test";

import { listenLink } from "./listen.ts";

const id = "4cOdK2wGLETKBW3PvgPWqT";

test("a shared Spotify link loses its share id and locale", () => {
  assert.deepEqual(listenLink(`https://open.spotify.com/intl-it/track/${id}?si=a1b2c3d4e5f60718`), {
    platform: "spotify",
    url: `https://open.spotify.com/track/${id}`,
  });
  assert.equal(
    listenLink(`https://open.spotify.com/episode/${id}?si=x`)?.url,
    `https://open.spotify.com/episode/${id}`,
  );
});

test("links no platform recognises are refused", () => {
  assert.equal(listenLink(`https://open.spotify.com/playlist/${id}`), undefined);
  assert.equal(listenLink(`https://open.spotify.com.evil.example/track/${id}`), undefined);
  assert.equal(listenLink(`https://open.spotify.com/track/${id}extra`), undefined);
  assert.equal(listenLink("spotify:track:4cOdK2wGLETKBW3PvgPWqT"), undefined);
});
