import type { APIRoute, InferGetStaticPropsType } from "astro";

import { getPostPaths } from "../../lib/posts.ts";
import { displayFonts, lightPalette, renderPreview } from "../../plugins/preview/preview.ts";
import tokens from "../../styles/tokens.css?raw";

export const getStaticPaths = getPostPaths;
type Props = InferGetStaticPropsType<typeof getStaticPaths>;

const palette = lightPalette(tokens);
const fonts = await displayFonts();

export const GET: APIRoute<Props> = async ({ props }) =>
  new Response(new Uint8Array(await renderPreview(props.post.data.title, palette, fonts)), {
    headers: { "Content-Type": "image/png" },
  });
