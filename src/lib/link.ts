import { z } from "astro/zod";

// A link card: a page worth reading elsewhere, shown as its title, a line about it and its picture, the way Notion
// shows a bookmark. Only the link is required: without a title the card shows the address, without a site the host,
// and without an image the card is text only. Every field is written in the post, so the build never fetches
// anything and a reader's browser asks the other site for nothing until they click.

// Tracking parameters a shared link picks up on its way here; the rest of the query is part of the address.
const tracking = /^(utm_\w+|fbclid|gclid|mc_cid|mc_eid)$/;

// The link without tracking parameters, or undefined when it isn't a web address.
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

// simonwillison.net/2024/dec/31/llms-in-2024, the way an address reads when nobody gave it a title.
const address = (url: URL): string => {
  const path = decodeURI(url.pathname).replace(/\/$/, "");
  return `${host(url)}${path}`;
};

// `image` is a plain path in a block, which the remark plugin turns into a Markdown image so Astro resizes it; a
// picture on the other site would be fetched by every reader's browser, so it has to be a file next to index.md.
export const linkSchema = z
  .strictObject({
    url: z.string(),
    title: z.string().min(1).optional(),
    // What the page is about, in its words or yours.
    description: z.string().min(1).optional(),
    // Who published it, when the host doesn't say: "julia evans" for jvns.ca.
    site: z.string().min(1).optional(),
    image: z
      .string()
      .min(1)
      .refine(path => !/^[a-z][a-z\d+.-]*:/i.test(path), "a file next to index.md, not a link")
      .optional(),
  })
  .transform((link, context) => {
    const url = canonicalLink(link.url);
    if (url === undefined) {
      context.addIssue({ code: "custom", path: ["url"], message: "an http or https link" });
      return z.NEVER;
    }
    return {
      ...link,
      url: url.href,
      host: host(url),
      title: link.title ?? address(url),
      site: link.site ?? host(url),
    };
  });

export type Link = z.output<typeof linkSchema>;
