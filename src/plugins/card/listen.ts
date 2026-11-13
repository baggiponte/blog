// Where a song, podcast or video card sends the reader. Each platform recognises its own share links, keeps only what
// the link needs, says what it most likely is (a Spotify track is a song, a YouTube link a video) and how you'd use it
// there: the card reads "listen on spotify" or "watch on youtube".
// Another platform (Apple Music, a podcast's own site) is one more entry in `platforms`, plus its icon in global.css.
export type MediaType = "song" | "podcast" | "video";

export interface Platform {
  // Shown on the card and used to pick its icon (`data-platform`).
  name: string;
  verb: "listen" | "watch";
  canonical(link: string): { url: string; type: MediaType } | undefined;
}

// Drops `si`, which ties every click back to whoever shared the link, and the `intl-xx` locale.
const spotify: Platform = {
  name: "spotify",
  verb: "listen",
  canonical(link) {
    const match =
      /^https:\/\/open\.spotify\.com\/(?:intl-[a-z]{2}(?:-[a-z]{2})?\/)?(track|album|episode|show)\/([A-Za-z0-9]{22})(?:[/?#]|$)/i.exec(
        link,
      );
    if (match === null) {
      return undefined;
    }
    const kind = match[1]?.toLowerCase() ?? "";
    return {
      url: `https://open.spotify.com/${kind}/${match[2]}`,
      type: kind === "episode" || kind === "show" ? "podcast" : "song",
    };
  },
};

// The video in youtu.be/<id> or youtube.com/watch?v=<id> (www. and m. included), else null.
function videoId(url: URL): string | null {
  const host = url.hostname.replace(/^(www|m)\./, "");
  if (host === "youtu.be") {
    return url.pathname.slice(1);
  }
  return host === "youtube.com" && url.pathname === "/watch" ? url.searchParams.get("v") : null;
}

// youtube.com/watch?v=…, m.youtube.com and youtu.be/…, keeping a start time (`t`) and dropping `si` and the rest.
const youtube: Platform = {
  name: "youtube",
  verb: "watch",
  canonical(link) {
    let url: URL;
    try {
      url = new URL(link);
    } catch {
      return undefined;
    }
    const id = videoId(url);
    if (url.protocol !== "https:" || id === null || !/^[\w-]{11}$/.test(id)) {
      return undefined;
    }
    const start = url.searchParams.get("t");
    const time = start !== null && /^\d+s?$/.test(start) ? `&t=${start}` : "";
    return { url: `https://www.youtube.com/watch?v=${id}${time}`, type: "video" };
  },
};

export const platforms: readonly Platform[] = [spotify, youtube];

export interface ListenLink {
  platform: string;
  verb: Platform["verb"];
  url: string;
  // What the link most likely is, for a card that doesn't say.
  type: MediaType;
}

export function listenLink(link: string): ListenLink | undefined {
  for (const platform of platforms) {
    const found = platform.canonical(link.trim());
    if (found !== undefined) {
      return { platform: platform.name, verb: platform.verb, ...found };
    }
  }
  return undefined;
}
