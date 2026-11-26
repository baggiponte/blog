import { parseArgs } from "node:util";

import { formatSchema } from "../src/lib/post-schema.ts";
import { checkPost, dirForTitle, newPostMarkdown } from "../src/plugins/editor/post.ts";
import { fileStore } from "../src/plugins/editor/store.ts";

// Writes a draft bundle the way the editor's "new post" button does, so `just new`, `just notebook` and the editor
// name folders and fill in frontmatter alike. Prints the folder name for the justfile to open.
//   node scripts/new-post.ts [--notebook <name>.md] <post|tidbits> <title words…>

const usage = "usage: node scripts/new-post.ts [--notebook <name>.md] <post|tidbits> <title…>";

try {
  const { values, positionals } = parseArgs({ options: { notebook: { type: "string" } }, allowPositionals: true });
  const [format, ...words] = positionals;
  const title = words.join(" ");
  const parsed = formatSchema.safeParse(format);
  if (!parsed.success || title === "") {
    console.error(usage);
    process.exit(2);
  }

  const dir = dirForTitle(title);
  const markdown = newPostMarkdown(
    { title, description: "", format: parsed.data, notebook: values.notebook },
    new Date(),
  );
  // Catches a notebook name the content collection would refuse before any folder is made.
  checkPost(markdown);
  await fileStore("content/blog").create(dir, markdown);
  console.log(dir);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
