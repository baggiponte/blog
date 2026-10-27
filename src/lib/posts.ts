import { getCollection, getEntry, type CollectionEntry } from "astro:content";

export type Post = CollectionEntry<"blog">;

// Bundle directories keep their ordering prefix on disk; URLs drop it.
export const slugOf = (post: Post) => post.id.replace(/^\d+-/, "");

export async function getPosts() {
  const posts = await getCollection("blog", ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

// The entry whose body is the post: the notebook export when there is one, else the bundle itself.
export async function bodyEntry(post: Post) {
  if (!post.data.notebook) return post;
  const id = `${post.id}/${post.data.notebook.replace(/\.md$/, "")}`;
  const entry = await getEntry("notebooks", id);
  if (!entry) throw new Error(`${post.id}: notebook export ${post.data.notebook} not found; run 'just marimo-export'`);
  return entry;
}

// Plain Markdown for "copy page" and "view as markdown": marimo cell attributes and
// separators are noise outside the notebook.
export async function markdownOf(post: Post) {
  const { body = "" } = await bodyEntry(post);
  const clean = body
    .replace(/^(```\w*)\s*\{[^}\n]*\}/gm, "$1")
    .replace(/^<!---->$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return `# ${post.data.title}\n\n${post.data.description}\n\n${clean}\n`;
}

export function readingMinutes(markdown: string) {
  return Math.max(1, Math.round(markdown.split(/\s+/).length / 220));
}

export const formatDate = (date: Date) =>
  date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).toLowerCase();
