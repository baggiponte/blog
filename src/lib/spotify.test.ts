import assert from "node:assert/strict";
import { test } from "node:test";

import { spotifyLink } from "./spotify.ts";

const id = "4cOdK2wGLETKBW3PvgPWqT";

test("a shared link loses its share id and locale", () => {
  assert.deepEqual(spotifyLink(`https://open.spotify.com/intl-it/track/${id}?si=a1b2c3d4e5f60718`), {
    url: `https://open.spotify.com/track/${id}`,
    kind: "song",
  });
  assert.equal(spotifyLink(`https://open.spotify.com/album/${id}`)?.url, `https://open.spotify.com/album/${id}`);
});

test("episodes and shows are podcasts, tracks and albums are songs", () => {
  assert.equal(spotifyLink(`https://open.spotify.com/episode/${id}?si=x`)?.kind, "podcast");
  assert.equal(spotifyLink(`https://open.spotify.com/show/${id}`)?.kind, "podcast");
  assert.equal(spotifyLink(`https://open.spotify.com/album/${id}`)?.kind, "song");
});

test("anything but a Spotify track, album, episode or show is refused", () => {
  assert.equal(spotifyLink(`https://open.spotify.com/playlist/${id}`), undefined);
  assert.equal(spotifyLink(`https://open.spotify.com.evil.example/track/${id}`), undefined);
  assert.equal(spotifyLink(`https://open.spotify.com/track/${id}extra`), undefined);
  assert.equal(spotifyLink("spotify:track:4cOdK2wGLETKBW3PvgPWqT"), undefined);
});
