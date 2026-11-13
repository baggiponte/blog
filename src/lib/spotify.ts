// A Spotify link as the app shares it: https://open.spotify.com/intl-it/track/<id>?si=<share id>.
// The `si` parameter ties every click back to whoever shared the link, and the locale only picks the page's language,
// so both are dropped: the card links to https://open.spotify.com/track/<id>.
const shared =
  /^https:\/\/open\.spotify\.com\/(?:intl-[a-z]{2}(?:-[a-z]{2})?\/)?(track|album|episode|show)\/([A-Za-z0-9]{22})(?:[/?#]|$)/i;

// Music gets a record sleeve, podcasts a stack of episodes.
export type SpotifyKind = "song" | "podcast";

export interface SpotifyLink {
  url: string;
  kind: SpotifyKind;
}

export function spotifyLink(link: string): SpotifyLink | undefined {
  const match = shared.exec(link.trim());
  if (match === null) {
    return undefined;
  }
  const type = match[1]?.toLowerCase();
  return {
    url: `https://open.spotify.com/${type}/${match[2]}`,
    kind: type === "episode" || type === "show" ? "podcast" : "song",
  };
}
