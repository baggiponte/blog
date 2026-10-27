import rss from "@astrojs/rss";
import type { APIContext } from "astro";

import { getPosts, slugOf } from "../lib/posts";
import { site } from "../site";

export async function GET(context: APIContext) {
  const posts = await getPosts();
  return rss({
    title: site.title,
    description: site.description,
    site: context.site!,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: `/blog/${slugOf(post)}/`,
    })),
  });
}
