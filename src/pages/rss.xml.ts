import rss from "@astrojs/rss";
import type { APIRoute } from "astro";

import { getPosts, urlOf } from "../lib/posts.ts";
import { site } from "../site.ts";

export const GET: APIRoute = async context => {
  if (context.site === undefined) {
    throw new Error("rss.xml: set `site` in astro.config.ts so feed links are absolute");
  }
  const posts = await getPosts();
  return rss({
    title: site.title,
    description: site.description,
    site: context.site,
    items: posts.map(post => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: urlOf(post),
    })),
  });
};
