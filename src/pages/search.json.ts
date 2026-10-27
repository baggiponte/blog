import { formatDate, getPosts, markdownOf, slugOf } from "../lib/posts";

// Small enough to ship whole; the ⌘K palette fetches it on first open.
export async function GET() {
  const posts = await getPosts();
  const index = await Promise.all(
    posts.map(async (post) => ({
      title: post.data.title,
      description: post.data.description,
      date: formatDate(post.data.date),
      url: `/blog/${slugOf(post)}/`,
      text: (await markdownOf(post)).replace(/```[\s\S]*?```/g, " ").replace(/\s+/g, " ").toLowerCase(),
    })),
  );
  return new Response(JSON.stringify(index), { headers: { "Content-Type": "application/json" } });
}
