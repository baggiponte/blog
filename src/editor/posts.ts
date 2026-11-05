import { access, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";

import { urlOfDir } from "../lib/slugs.ts";

// What the editor lists: every bundle written by hand. Notebook posts are written in marimo, so they're left out.
export interface DraftSummary {
  dir: string;
  title: string;
  draft: boolean;
  url: string;
}

export type Format = "post" | "tidbits";

// Bundle directories and image names: lowercase words joined by dashes, like `01-groundup-linear-regression`.
const dirPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const isBundleDir = (dir: string): boolean => dirPattern.test(dir);

const imageExtensions = new Set([".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"]);

// "Café au lait!" -> "cafe-au-lait".
export function slugify(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// A frontmatter value from index.md, unquoted; enough to list posts without a YAML parser.
function frontmatterValue(markdown: string, key: string): string | undefined {
  const frontmatter = /^---\n([\s\S]*?)\n---/.exec(markdown)?.[1] ?? "";
  const value = new RegExp(`^${key}:[ \\t]*(.*)$`, "m").exec(frontmatter)?.[1]?.trim();
  return value?.replace(/^(["'])(.*)\1$/, "$2");
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export class EditorError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// The index.md of an existing bundle; anything else (a typo, `../`, a notebook's folder name) is refused.
async function bundlePath(blogDir: string, dir: string): Promise<string> {
  if (!isBundleDir(dir)) {
    throw new EditorError(400, `not a post folder name: ${dir}`);
  }
  const path = join(blogDir, dir, "index.md");
  if (!(await exists(path))) {
    throw new EditorError(404, `no post in content/blog/${dir}`);
  }
  return path;
}

// Newest folder first, which is also the newest post for numbered and dated names alike.
export async function listPosts(blogDir: string): Promise<DraftSummary[]> {
  const entries = await readdir(blogDir, { withFileTypes: true });
  const posts: DraftSummary[] = [];
  for (const entry of entries) {
    if (!(entry.isDirectory() && isBundleDir(entry.name))) {
      continue;
    }
    let markdown: string;
    try {
      markdown = await readFile(join(blogDir, entry.name, "index.md"), "utf8");
    } catch {
      continue;
    }
    if (frontmatterValue(markdown, "notebook") !== undefined) {
      continue;
    }
    posts.push({
      dir: entry.name,
      title: frontmatterValue(markdown, "title") || entry.name,
      draft: frontmatterValue(markdown, "draft") === "true",
      url: urlOfDir(entry.name),
    });
  }
  return posts.toSorted((a, b) => b.dir.localeCompare(a.dir));
}

export async function readPost(blogDir: string, dir: string): Promise<string> {
  return readFile(await bundlePath(blogDir, dir), "utf8");
}

export async function writePost(blogDir: string, dir: string, markdown: string): Promise<void> {
  await writeFile(await bundlePath(blogDir, dir), markdown);
}

// The same bundle `just new` writes, with the title and description already filled in.
export function newPostMarkdown(title: string, description: string, format: Format, date: Date): string {
  const day = date.toISOString().slice(0, 10);
  return [
    "---",
    `title: ${JSON.stringify(title)}`,
    `description: ${JSON.stringify(description)}`,
    `date: ${day}`,
    `format: ${format}`,
    "draft: true",
    "---",
    "",
    "",
  ].join("\n");
}

// Creates content/blog/<slug of the title>/index.md and returns the folder name.
export async function createPost(
  blogDir: string,
  { title, description, format }: { title: string; description: string; format: Format },
  date: Date,
): Promise<string> {
  const dir = slugify(title);
  if (dir === "") {
    throw new EditorError(400, "the title needs at least one letter or digit");
  }
  const folder = join(blogDir, dir);
  if (await exists(folder)) {
    throw new EditorError(409, `content/blog/${dir} already exists`);
  }
  await mkdir(folder);
  await writeFile(join(folder, "index.md"), newPostMarkdown(title.toLowerCase(), description, format, date));
  return dir;
}

// "Screenshot 2026-11-05 at 10.00.PNG" -> "screenshot-2026-11-05-at-10-00.png".
export function imageFileName(original: string): string {
  const extension = extname(original).toLowerCase();
  if (!imageExtensions.has(extension)) {
    throw new EditorError(415, `not an image: ${original}`);
  }
  const base = slugify(original.slice(0, -extension.length)) || "image";
  return `${base}${extension === ".jpeg" ? ".jpg" : extension}`;
}

// Saves a dropped image next to the post's index.md, never over another file, and returns its file name.
export async function saveImage(blogDir: string, dir: string, original: string, bytes: Uint8Array): Promise<string> {
  await bundlePath(blogDir, dir);
  const name = imageFileName(original);
  const extension = extname(name);
  const base = name.slice(0, -extension.length);
  for (let n = 1; ; n++) {
    const candidate = n === 1 ? name : `${base}-${n}${extension}`;
    try {
      await writeFile(join(blogDir, dir, candidate), bytes, { flag: "wx" });
      return candidate;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") {
        throw error;
      }
    }
  }
}
