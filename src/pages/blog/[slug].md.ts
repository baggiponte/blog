import type { APIRoute, InferGetStaticPropsType } from "astro";

import { getPostPaths, markdownOf } from "../../lib/posts.ts";

export const getStaticPaths = getPostPaths;
type Props = InferGetStaticPropsType<typeof getStaticPaths>;

export const GET: APIRoute<Props> = async ({ props }) =>
  new Response(await markdownOf(props.post), { headers: { "Content-Type": "text/markdown; charset=utf-8" } });
