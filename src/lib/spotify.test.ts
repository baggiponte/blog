import assert from "node:assert/strict";
import { test } from "node:test";

import { spotifyUrl } from "./spotify.ts";

const id = "4cOdK2wGLETKBW3PvgPWqT";

test("a shared link loses its share id and locale", () => {
  assert.equal(
    spotifyUrl(`https://open.spotify.com/intl-it/track/${id}?si=a1b2c3d4e5f60718`),
    `https://open.spotify.com/track/${id}`,
  );
  assert.equal(spotifyUrl(`https://open.spotify.com/album/${id}`), `https://open.spotify.com/album/${id}`);
});

test("anything but a Spotify track or album is refused", () => {
  assert.equal(spotifyUrl(`https://open.spotify.com/playlist/${id}`), undefined);
  assert.equal(spotifyUrl(`https://open.spotify.com.evil.example/track/${id}`), undefined);
  assert.equal(spotifyUrl(`https://open.spotify.com/track/${id}extra`), undefined);
  assert.equal(spotifyUrl("spotify:track:4cOdK2wGLETKBW3PvgPWqT"), undefined);
});
