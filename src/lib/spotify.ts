// A Spotify link as the app shares it: https://open.spotify.com/intl-it/track/<id>?si=<share id>.
// The `si` parameter ties every click back to whoever shared the link, and the locale only picks the page's language,
// so both are dropped: the card links to https://open.spotify.com/track/<id>.
const shared =
  /^https:\/\/open\.spotify\.com\/(?:intl-[a-z]{2}(?:-[a-z]{2})?\/)?(track|album)\/([A-Za-z0-9]{22})(?:[/?#]|$)/i;

export function spotifyUrl(link: string): string | undefined {
  const match = shared.exec(link.trim());
  return match === null ? undefined : `https://open.spotify.com/${match[1]?.toLowerCase()}/${match[2]}`;
}
