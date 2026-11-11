import { createHash } from "node:crypto";
import { access, mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";

import {
  type BundleDir,
  dirForTitle,
  EditorError,
  type ImageName,
  imageName,
  imageType,
  isBundleDir,
  type NewPost,
  newPostMarkdown,
  numbered,
  type PostSummary,
  summarize,
} from "./post.ts";

// Where posts are kept, as the API sees it. The API only talks to this interface, so its tests can hand it a small
// in-memory store, and the files on disk are tested once, here, against a temporary folder.

// A fingerprint of an index.md's content. The editor sends back the version it read with every save; if the file has
// changed since, from another tab or another editor, the save is refused instead of overwriting it.
export type Version = string & { readonly __brand: "Version" };

export interface Stored {
  markdown: string;
  version: Version;
}

export interface Image {
  bytes: Uint8Array<ArrayBuffer>;
  type: string;
}

export interface PostStore {
  // Every hand-written post, newest date first.
  list(): Promise<PostSummary[]>;
  read(dir: BundleDir): Promise<Stored>;
  // Replaces index.md if it's still at `expected`, and returns the new version. Without `expected`, it always writes.
  write(dir: BundleDir, markdown: string, expected?: Version): Promise<Version>;
  create(post: NewPost, date: Date): Promise<BundleDir>;
  // Saves an image next to index.md, never over another file, and returns the name it got.
  addImage(dir: BundleDir, original: string, bytes: Uint8Array): Promise<ImageName>;
  readImage(dir: BundleDir, name: ImageName): Promise<Image>;
}

// A content hash rather than the file's modification time: an editor that rewrites the same text doesn't count as
// a change, and two writes within the same millisecond still do.
export const versionOf = (markdown: string): Version =>
  createHash("sha256").update(markdown).digest("base64url").slice(0, 16) as Version;

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
      if (expected !== undefined) {
        const current = (await read(dir)).version;
        // The same text as on disk is never a conflict, even if the editor's version is stale (VS Code's rule).
        if (current !== expected && current !== version) {
          throw new EditorError(`content/blog/${dir}/index.md changed since it was opened`, { failure: "conflict" });
        }
      }
      // Into a temporary file, then renamed over index.md: a crash mid-write leaves the old file whole, and the dev
      // server's watcher never sees a half-written one. The check above and this rename are still two steps, so a
      // write from elsewhere in between is lost; with one person typing, that window is a few milliseconds.
      const temporary = join(blogDir, dir, `.index.md.${process.pid}.tmp`);
      await writeFile(temporary, markdown);
      await rename(temporary, path);
      return version;
    },

    async create(post, date) {
      const dir = dirForTitle(post.title);
      try {
        await mkdir(join(blogDir, dir));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "EEXIST") {
          throw new EditorError(`content/blog/${dir} already exists`, { failure: "exists", cause: error });
        }
        throw error;
      }
      await writeFile(join(blogDir, dir, "index.md"), newPostMarkdown(post, date));
      return dir;
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
  };
}
