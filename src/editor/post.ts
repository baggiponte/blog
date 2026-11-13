import { parseFrontmatter } from "@astrojs/markdown-remark";
import { z } from "astro/zod";
import { blockKinds } from "../lib/blocks.ts";
import { checkBlocks } from "../lib/fenced-blocks.ts";
import { formatSchema, postSchema } from "../lib/post-schema.ts";
import { slugify, urlOfDir } from "../lib/slugs.ts";
import { EditorError } from "./failure.ts";

// What a post is to the editor, with no storage and no HTTP: the rules here hold wherever posts are kept.

// A bundle folder name that passed the check below. The brand exists only for the type checker: a plain string can't
// be passed where a BundleDir is expected, so every path built from one has gone through parseBundleDir first.
export type BundleDir = string & { readonly __brand: "BundleDir" };
// An image file name the editor wrote or accepts reading: same idea.
export type ImageName = string & { readonly __brand: "ImageName" };

export type Format = z.infer<typeof formatSchema>;

// A fingerprint of an index.md's content (store.ts computes it). The editor sends back the version it read with every
// save; if the file has changed since, from another tab or another editor, the save is refused instead of overwriting.
export type Version = string & { readonly __brand: "Version" };

// Whether saving text with version `incoming`, over a file now at `current`, would overwrite someone else's change:
// the editor read `expected`, and the file moved on since. The same text as on disk is never a conflict (VS Code's
// rule), and a save without a version always goes through.
export const isConflict = (current: Version, expected: Version | undefined, incoming: Version): boolean =>
  expected !== undefined && current !== expected && current !== incoming;

// What the sidebar shows for each hand-written post.
export interface PostSummary {
  dir: BundleDir;
  // The frontmatter date as written, e.g. "2026-11-10"; empty when there is none.
  date: string;
  title: string;
  draft: boolean;
  url: string;
}

// What the "new post" form sends.
const newPostSchema = z.object({ title: z.string(), description: z.string(), format: formatSchema });
export type NewPost = z.infer<typeof newPostSchema>;

// Lowercase words joined by dashes, like `01-groundup-linear-regression`: no `..`, no slashes, no dotfiles.
const dirPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export const isBundleDir = (text: string): text is BundleDir => dirPattern.test(text);

export function parseBundleDir(text: string): BundleDir {
  if (!isBundleDir(text)) {
    throw new EditorError(`not a post folder name: ${text}`, { failure: "invalid" });
  }
  return text;
}

export function parseNewPost(input: unknown): NewPost {
  const result = newPostSchema.safeParse(input);
  if (!result.success) {
    throw new EditorError("a post needs a title, a description and a format", {
      failure: "invalid",
      cause: result.error,
    });
  }
  return result.data;
}

// The folder a new post goes in: its title, slugified.
export function dirForTitle(title: string): BundleDir {
  const dir = slugify(title);
  if (dir === "") {
    throw new EditorError("the title needs at least one letter or digit", { failure: "invalid" });
  }
  return dir as BundleDir;
}

// The index.md `just new` writes, with the title and description already filled in.
export function newPostMarkdown({ title, description, format }: NewPost, date: Date): string {
  return [
    "---",
    `title: ${JSON.stringify(title.toLowerCase())}`,
    `description: ${JSON.stringify(description)}`,
    `date: ${date.toISOString().slice(0, 10)}`,
    `format: ${format}`,
    "draft: true",
    "---",
    "",
    "",
  ].join("\n");
}

// The frontmatter as YAML data, parsed the way Astro parses it; a syntax error becomes an "invalid" failure.
function frontmatterOf(markdown: string): Record<string, unknown> {
  try {
    return parseFrontmatter(markdown).frontmatter;
  } catch (error) {
    const reason = error instanceof Error ? (error.message.split("\n")[0] ?? "").replace(/:$/, "") : String(error);
    throw new EditorError(`the frontmatter isn't valid YAML: ${reason}`, { failure: "invalid", cause: error });
  }
}

// Refuses an index.md the site couldn't build: the same schema as the content collection (src/content.config.ts),
// and the same check as the build for every ```player and ```link block, so the editor never writes a file that stops
// `astro dev` or the build. Drafts are let through with a capitalised title or no description, as the schema allows.
export function checkPost(markdown: string): void {
  const result = postSchema.safeParse(frontmatterOf(markdown));
  if (!result.success) {
    const issues = result.error.issues.map(issue => `${issue.path.join(".") || "frontmatter"}: ${issue.message}`);
    throw new EditorError(issues.join("; "), { failure: "invalid" });
  }
  try {
    checkBlocks(markdown, blockKinds);
  } catch (error) {
    throw new EditorError(error instanceof Error ? error.message : String(error), { failure: "invalid", cause: error });
  }
}

// YAML reads `date: 2026-03-01` as a Date; the sidebar shows it as written.
function dateText(date: unknown): string {
  if (date instanceof Date) {
    return date.toISOString().slice(0, 10);
  }
  return typeof date === "string" ? date : "";
}

// The sidebar row for a bundle, or undefined for a notebook post, which is written in marimo and not here.
// A post whose frontmatter is broken (hand-edited outside the editor) still gets a row, so it can be opened and fixed.
export function summarize(dir: BundleDir, markdown: string): PostSummary | undefined {
  let data: Record<string, unknown> = {};
  try {
    data = frontmatterOf(markdown);
  } catch {
    // Listed under its folder name.
  }
  const { notebook, title, date, draft } = data;
  if (notebook !== undefined) {
    return undefined;
  }
  return {
    dir,
    title: typeof title === "string" && title !== "" ? title : dir,
    date: dateText(date),
    draft: draft === true,
    url: urlOfDir(dir),
  };
}

const imageTypes: Record<string, string> = {
  ".avif": "image/avif",
  ".gif": "image/gif",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

const extensionOf = (name: string): string => /\.[^.]*$/.exec(name)?.[0].toLowerCase() ?? "";

// The media type of an image the site serves, or undefined for any other file.
export const imageType = (name: string): string | undefined => imageTypes[extensionOf(name)];

// "Screenshot 2026-11-05 at 10.00.PNG" -> "screenshot-2026-11-05-at-10-00.png": the name a dropped image is saved as.
export function imageName(original: string): ImageName {
  const extension = extensionOf(original);
  if (imageTypes[extension] === undefined) {
    throw new EditorError(`not an image: ${original}`, { failure: "unsupported" });
  }
  const base = slugify(original.slice(0, -extension.length)) || "image";
  return `${base}${extension === ".jpeg" ? ".jpg" : extension}` as ImageName;
}

// "plot.png" -> "plot-2.png": the name to try when the first one is taken.
export function numbered(name: ImageName, n: number): ImageName {
  const extension = extensionOf(name);
  return n === 1 ? name : (`${name.slice(0, -extension.length)}-${n}${extension}` as ImageName);
}

// An image name asked for by URL: a plain file name with an image extension, no folders and no dotfiles.
export function parseImageName(text: string): ImageName {
  if (!/^\w[\w.-]*$/.test(text) || imageType(text) === undefined) {
    throw new EditorError(`not an image in the post's folder: ${text}`, { failure: "invalid" });
  }
  return text as ImageName;
}
