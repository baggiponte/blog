import { type CollectionEntry, getCollection, getEntry } from "astro:content";
import type { GetStaticPaths } from "astro";

import { plainMarkdown } from "./markdown.ts";

export type Post = CollectionEntry<"blog">;
type Body = Post | CollectionEntry<"notebooks">;

// Bundle directories keep their ordering prefix on disk (01-groundup-...); URLs drop it.
export const slugOf = (post: Post): string => post.id.replace(/^\d+-/, "");
export const urlOf = (post: Post): string => `/blog/${slugOf(post)}/`;
export const markdownUrlOf = (post: Post): string => `/blog/${slugOf(post)}.md`;

// Newest first; drafts only show in `astro dev`.
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection("blog", ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.toSorted((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

// One page per post, for both /blog/<slug>/ and /blog/<slug>.md.
export const getPostPaths = (async () => {
  const posts = await getPosts();
  return posts.map(post => ({ params: { slug: slugOf(post) }, props: { post } }));
}) satisfies GetStaticPaths;

// The entry whose body is the post: the notebook export when there is one, else the bundle itself.
export async function bodyEntry(post: Post): Promise<Body> {
  if (post.data.notebook === undefined) {
    return post;
  }
  const id = `${post.id}/${post.data.notebook.replace(/\.md$/, "")}`;
  const entry = await getEntry("notebooks", id);
  if (entry === undefined) {
    throw new Error(`${post.id}: notebook export ${post.data.notebook} not found; run 'just marimo-export'`);
  }
  return entry;
}

// The post as one Markdown document: title, description and body.
export async function markdownOf(post: Post): Promise<string> {
  const { body = "" } = await bodyEntry(post);
  return `# ${post.data.title}\n\n${post.data.description}\n\n${plainMarkdown(body)}\n`;
}

const wordsPerMinute = 220;
export function readingMinutes(markdown: string): number {
  const words = markdown.split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / wordsPerMinute));
}
