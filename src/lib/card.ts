import { z } from "astro/zod";

import { listenLink, type MediaType, type Platform } from "./listen.ts";

// A card: something worth opening elsewhere, shown as its title and a line about it. The same fields
// fill a ```card block anywhere in a post (src/lib/remark-card.ts) and the `song` a post ends with (frontmatter).
//
// - link: a page to read, drawn as a text bookmark with the site that published it.
// - song, podcast, video: something to play on Spotify or YouTube, drawn as a sleeve with its record, a stack of
//   episodes, or a screen.
//
// Only the link is required. Without a type the link decides it: a Spotify track or album is a song, an episode or
// show a podcast, a YouTube link a video, anything else a link. Without a title the card says what it is, and without
// an image it draws a blank one; a link has no image. Every field is written in the post: the build fetches nothing,
// and a reader's browser asks the other site for nothing until they click.
export const cardTypes = ["link", "song", "podcast", "video"] as const;
export type CardType = (typeof cardTypes)[number];

interface Common {
  url: string;
  title: string;
  // Why it's here, where to start, or what the page is about.
  note?: string | undefined;
}
export interface MediaCard<Image = string> extends Common {
  type: MediaType;
  image?: Image | undefined;
  platform: string;
  verb: Platform["verb"];
}
export interface LinkCard extends Common {
  type: "link";
  // Who published it ("julia evans"), the host when the block doesn't say.
  site: string;
  host: string;
}
export type Card<Image = string> = MediaCard<Image> | LinkCard;

// The picture's size on the page: square for a sleeve or an episode, 16:9 for a video. Astro resizes the image to
// twice this.
export const imageSize = (type: MediaType): { width: number; height: number } =>
  type === "video" ? { width: 160, height: 90 } : { width: 128, height: 128 };

// Tracking parameters a shared link picks up on its way here; the rest of the query is part of the address.
const tracking = /^(utm_\w+|fbclid|gclid|mc_cid|mc_eid)$/;

// A web address without tracking parameters, or undefined when it isn't one.
export function canonicalLink(link: string): URL | undefined {
  let url: URL;
  try {
    url = new URL(link.trim());
  } catch {
    return undefined;
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    return undefined;
  }
  for (const key of [...url.searchParams.keys()]) {
    if (tracking.test(key)) {
      url.searchParams.delete(key);
    }
  }
  return url;
}

const host = (url: URL): string => url.hostname.replace(/^www\./, "");

// jvns.ca/blog/git-config, the way an address reads when nobody gave it a title.
const address = (url: URL): string => `${host(url)}${decodeURI(url.pathname).replace(/\/$/, "")}`;

const defaultTitle = (type: MediaType, verb: Platform["verb"]): string =>
  `${{ song: "a song", podcast: "an episode", video: "a video" }[type]} worth a ${verb}`;

// `image` is Astro's image() in the frontmatter, and a plain path in a block, which the remark plugin turns into a
// Markdown image so Astro resizes it all the same.
export const cardSchema = <Image extends z.ZodType>(image: Image) =>
  z
    .strictObject({
      type: z.enum(cardTypes).optional(),
      url: z.string(),
      title: z.string().min(1).optional(),
      note: z.string().min(1).optional(),
      image: image.optional(),
      site: z.string().min(1).optional(),
    })
    .transform((card, context): Card<z.output<Image>> => {
      const { site, image, ...fields } = card;
      const media = listenLink(card.url);
      const type = card.type ?? media?.type ?? "link";
      if (type === "link") {
        const url = canonicalLink(card.url);
        if (url === undefined) {
          context.addIssue({ code: "custom", path: ["url"], message: "an http or https link" });
          return z.NEVER;
        }
        if (image !== undefined) {
          context.addIssue({ code: "custom", path: ["image"], message: "a link card is text only" });
          return z.NEVER;
        }
        return {
          ...fields,
          type,
          url: url.href,
          title: card.title ?? address(url),
          site: site ?? host(url),
          host: host(url),
        };
      }
      if (media === undefined) {
        context.addIssue({ code: "custom", path: ["url"], message: `a Spotify or YouTube link for a ${type}` });
        return z.NEVER;
      }
      if (site !== undefined) {
        context.addIssue({ code: "custom", path: ["site"], message: "only a link card names its site" });
        return z.NEVER;
      }
      const { platform, verb, url } = media;
      return { ...fields, type, platform, verb, url, image, title: card.title ?? defaultTitle(type, verb) };
    });

// The `song` a post ends with: anything to play, not a page to read.
export const songSchema = <Image extends z.ZodType>(image: Image) =>
  cardSchema(image).transform((card, context): MediaCard<z.output<Image>> => {
    if (card.type === "link") {
      context.addIssue({ code: "custom", path: ["url"], message: "a Spotify or YouTube link to play" });
      return z.NEVER;
    }
    return card;
  });
