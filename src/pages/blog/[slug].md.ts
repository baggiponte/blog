import type { APIContext } from "astro";

import { getPosts, markdownOf, slugOf, type Post } from "../../lib/posts";

export async function getStaticPaths() {
  const posts = await getPosts();
  return posts.map((post) => ({ params: { slug: slugOf(post) }, props: { post } }));
}

export async function GET({ props }: APIContext<{ post: Post }>) {
  return new Response(await markdownOf(props.post), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
