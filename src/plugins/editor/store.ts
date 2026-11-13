import { createHash } from "node:crypto";
import { access, mkdir, readdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";

import { EditorError } from "./failure.ts";
import {
  type BundleDir,
  type ImageName,
  imageName,
  imageType,
  isBundleDir,
  isConflict,
  numbered,
  type PostSummary,
  summarize,
  type Version,
} from "./post.ts";
import { parseSuggestions, type Suggestion, suggestionsJson } from "./suggestion.ts";

// Where posts are kept, as the API sees it. The API only talks to this interface, so its tests can hand it a small
// in-memory store, and the files on disk are tested once, here, against a temporary folder.

export interface Stored {
  markdown: string;
  version: Version;
}

export interface Image {
  bytes: Uint8Array<ArrayBuffer>;
  type: string;
}

// A post's open suggestions, with the version of the suggestions.json they came from.
export interface Suggestions {
  suggestions: Suggestion[];
  version: Version;
}

export interface PostStore {
  // Every hand-written post, newest date first.
  list(): Promise<PostSummary[]>;
  read(dir: BundleDir): Promise<Stored>;
  // Replaces index.md unless that would be a conflict (isConflict in post.ts), and returns the new version.
  write(dir: BundleDir, markdown: string, expected?: Version): Promise<Version>;
  // Makes the bundle folder with this index.md; refused if the folder exists.
  create(dir: BundleDir, markdown: string): Promise<void>;
  // Saves an image next to index.md, never over another file, and returns the name it got.
  addImage(dir: BundleDir, original: string, bytes: Uint8Array): Promise<ImageName>;
  readImage(dir: BundleDir, name: ImageName): Promise<Image>;
  // The proofreading suggestions Claude left in suggestions.json next to index.md; none when there's no file.
  readSuggestions(dir: BundleDir): Promise<Suggestions>;
  // Replaces them unless the file changed since `expected` (Claude wrote new ones); with none left, removes the file.
  writeSuggestions(dir: BundleDir, suggestions: Suggestion[], expected?: Version): Promise<Version>;
}

// A content hash rather than the file's modification time: an editor that rewrites the same text doesn't count as
// a change, and two writes within the same millisecond still do.
export const versionOf = (markdown: string): Version =>
  createHash("sha256").update(markdown).digest("base64url").slice(0, 16) as Version;

// Next to index.md; .gitignore keeps it out of commits, and the site never reads it.
export const suggestionsFile = "suggestions.json";

export const newestFirst = (a: PostSummary, b: PostSummary): number =>
  b.date.localeCompare(a.date) || b.dir.localeCompare(a.dir);

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

// Into a temporary file, then renamed over the real one: a crash mid-write leaves the old file whole, and the dev
// server's watcher never sees a half-written one.
async function replaceFile(path: string, text: string): Promise<void> {
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.tmp`);
  await writeFile(temporary, text);
  await rename(temporary, path);
}

// suggestions.json as text, or "" when there's none: an empty list and no file are the same thing.
async function textOrEmpty(path: string): Promise<string> {
  try {
    return await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return "";
    }
    throw error;
  }
}

function suggestionsIn(text: string): Suggestion[] {
  if (text === "") {
    return [];
  }
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (error) {
    throw new EditorError(`suggestions.json isn't valid JSON: ${(error as Error).message}`, {
      failure: "invalid",
      cause: error,
    });
  }
  return parseSuggestions(data);
}

// The posts in content/blog/<dir>/index.md.
export function fileStore(blogDir: string): PostStore {
  const indexOf = async (dir: BundleDir): Promise<string> => {
    const path = join(blogDir, dir, "index.md");
    if (!(await exists(path))) {
      throw new EditorError(`no post in content/blog/${dir}`, { failure: "missing" });
    }
    return path;
  };
  const read = async (dir: BundleDir): Promise<Stored> => {
    const markdown = await readFile(await indexOf(dir), "utf8");
    return { markdown, version: versionOf(markdown) };
  };

  return {
    async list() {
      const posts: PostSummary[] = [];
      for (const entry of await readdir(blogDir, { withFileTypes: true })) {
        const path = join(blogDir, entry.name, "index.md");
        if (!(entry.isDirectory() && isBundleDir(entry.name) && (await exists(path)))) {
          continue;
        }
        const summary = summarize(entry.name, await readFile(path, "utf8"));
        if (summary !== undefined) {
          posts.push(summary);
        }
      }
      return posts.toSorted(newestFirst);
    },

    read,

    async write(dir, markdown, expected) {
      const path = await indexOf(dir);
      const version = versionOf(markdown);
      if (expected !== undefined && isConflict((await read(dir)).version, expected, version)) {
        throw new EditorError(`content/blog/${dir}/index.md changed since it was opened`, { failure: "conflict" });
      }
      // The check above and this write are two steps, so a write from elsewhere in between is lost; with one person
      // typing, that window is a few milliseconds.
      await replaceFile(path, markdown);
      return version;
    },

    async create(dir, markdown) {
      try {
        await mkdir(join(blogDir, dir));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "EEXIST") {
          throw new EditorError(`content/blog/${dir} already exists`, { failure: "exists", cause: error });
        }
        throw error;
      }
      await writeFile(join(blogDir, dir, "index.md"), markdown);
    },

    async addImage(dir, original, bytes) {
      await indexOf(dir);
      const name = imageName(original);
      for (let n = 1; ; n++) {
        const candidate = numbered(name, n);
        try {
          // "wx" fails if the file exists, so two uploads can't both take the same name.
          await writeFile(join(blogDir, dir, candidate), bytes, { flag: "wx" });
          return candidate;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "EEXIST") {
            throw error;
          }
        }
      }
    },

    async readImage(dir, name) {
      await indexOf(dir);
      const path = join(blogDir, dir, name);
      if (!(await exists(path))) {
        throw new EditorError(`no ${name} in content/blog/${dir}`, { failure: "missing" });
      }
      return { bytes: await readFile(path), type: imageType(name) ?? "application/octet-stream" };
    },

    async readSuggestions(dir) {
      await indexOf(dir);
      const text = await textOrEmpty(join(blogDir, dir, suggestionsFile));
      return { suggestions: suggestionsIn(text), version: versionOf(text) };
    },

    async writeSuggestions(dir, suggestions, expected) {
      await indexOf(dir);
      const path = join(blogDir, dir, suggestionsFile);
      const text = suggestions.length === 0 ? "" : suggestionsJson(suggestions);
      const version = versionOf(text);
      if (expected !== undefined && isConflict(versionOf(await textOrEmpty(path)), expected, version)) {
        throw new EditorError(`content/blog/${dir}/${suggestionsFile} changed since it was read`, {
          failure: "conflict",
        });
      }
      if (text === "") {
        await rm(path, { force: true });
      } else {
        await replaceFile(path, text);
      }
      return version;
    },
  };
}
