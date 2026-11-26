import { formatSchema } from "../src/lib/post-schema.ts";
import { dirForTitle, newPostMarkdown } from "../src/plugins/editor/post.ts";
import { fileStore } from "../src/plugins/editor/store.ts";

// Writes a draft bundle the way the editor's "new post" button does, so `just new` and the editor name folders and
// fill in frontmatter alike. Prints the folder name for the justfile to open in the editor.
//   node scripts/new-post.ts <post|tidbits> <title words…>

const [format, ...words] = process.argv.slice(2);
const title = words.join(" ");
const parsed = formatSchema.safeParse(format);
if (!parsed.success || title === "") {
  console.error("usage: node scripts/new-post.ts <post|tidbits> <title…>");
  process.exit(2);
}

try {
  const dir = dirForTitle(title);
  await fileStore("content/blog").create(
    dir,
    newPostMarkdown({ title, description: "", format: parsed.data }, new Date()),
  );
  console.log(dir);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
