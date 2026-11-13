import assert from "node:assert/strict";
import { test } from "node:test";

import { listenLink } from "./listen.ts";

const id = "4cOdK2wGLETKBW3PvgPWqT";

test("a shared Spotify link loses its share id and locale, and is listened to", () => {
  assert.deepEqual(listenLink(`https://open.spotify.com/intl-it/track/${id}?si=a1b2c3d4e5f60718`), {
    platform: "spotify",
    verb: "listen",
    url: `https://open.spotify.com/track/${id}`,
  });
  assert.equal(
    listenLink(`https://open.spotify.com/episode/${id}?si=x`)?.url,
    `https://open.spotify.com/episode/${id}`,
  );
});

test("a YouTube link keeps its video and start time, and is watched", () => {
  assert.deepEqual(listenLink("https://youtu.be/dQw4w9WgXcQ?si=abc&t=42"), {
    platform: "youtube",
    verb: "watch",
    url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42",
  });
  assert.equal(
    listenLink("https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=x")?.url,
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  );
});

test("links no platform recognises are refused", () => {
  assert.equal(listenLink(`https://open.spotify.com/playlist/${id}`), undefined);
  assert.equal(listenLink(`https://open.spotify.com.evil.example/track/${id}`), undefined);
  assert.equal(listenLink(`https://open.spotify.com/track/${id}extra`), undefined);
  assert.equal(listenLink("spotify:track:4cOdK2wGLETKBW3PvgPWqT"), undefined);
  assert.equal(listenLink("https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ"), undefined);
  assert.equal(listenLink("http://youtu.be/dQw4w9WgXcQ"), undefined);
  assert.equal(listenLink("not a link"), undefined);
});
