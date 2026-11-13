// Where a player card sends the reader. Each platform recognises its own share links and keeps only what the link
// needs: a Spotify link drops `si`, which ties every click back to whoever shared it, and the `intl-xx` locale.
// Another platform (Apple Music, YouTube, a podcast's own site) is one more entry in `platforms`.
export interface Platform {
  // Shown on the card ("listen on spotify") and used to pick its icon.
  name: string;
  canonical(link: string): string | undefined;
}

const spotify: Platform = {
  name: "spotify",
  canonical(link) {
    const match =
      /^https:\/\/open\.spotify\.com\/(?:intl-[a-z]{2}(?:-[a-z]{2})?\/)?(track|album|episode|show)\/([A-Za-z0-9]{22})(?:[/?#]|$)/i.exec(
        link.trim(),
      );
    return match === null ? undefined : `https://open.spotify.com/${match[1]?.toLowerCase()}/${match[2]}`;
  },
};

export const platforms: readonly Platform[] = [spotify];

export interface ListenLink {
  platform: string;
  url: string;
}

export function listenLink(link: string): ListenLink | undefined {
  for (const platform of platforms) {
    const url = platform.canonical(link);
    if (url !== undefined) {
      return { platform: platform.name, url };
    }
  }
  return undefined;
}
