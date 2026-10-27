import type { APIRoute } from "astro";

import { formatDate } from "../lib/dates.ts";
import { getPosts, markdownOf, urlOf } from "../lib/posts.ts";
import type { SearchEntry } from "../lib/search.ts";

// Small enough to ship whole; the palette fetches it on first open.
export const GET: APIRoute = async () => {
  const posts = await getPosts();
  const index: SearchEntry[] = await Promise.all(
    posts.map(async post => ({
      title: post.data.title,
      description: post.data.description,
      date: formatDate(post.data.date),
      url: urlOf(post),
      text: (await markdownOf(post))
        .replace(/```[\s\S]*?```/g, " ")
        .replace(/\s+/g, " ")
        .toLowerCase(),
    })),
  );
  return Response.json(index);
};
