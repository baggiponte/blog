import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { FontProvider } from "astro";

// A font face as an installed Fontsource package declares it in its index.css.
export interface Face {
  // The file, relative to the package.
  file: string;
  // latin, latin-ext, vietnamese…: the part of the file name between the font and its axis.
  subset: string;
  weight: string;
  style: string;
  unicodeRange: string[];
}

// The @font-face rules of a Fontsource variable package's index.css. `font` is the package's name without its scope,
// which starts every file name: "files/instrument-sans-latin-ext-wght-normal.woff2" is instrument-sans's latin-ext.
export function parseFaces(css: string, font: string): Face[] {
  return [...css.matchAll(/@font-face\s*\{([^}]*)\}/g)].map(([, rule = ""]) => {
    const value = (property: string): string => new RegExp(`${property}:\\s*([^;]+);`).exec(rule)?.[1]?.trim() ?? "";
    const file = /url\(\.\/([^)]+)\)/.exec(value("src"))?.[1];
    const prefix = `files/${font}-`;
    const subset = file?.startsWith(prefix)
      ? /^([a-z-]+)-wght-(?:normal|italic)\.woff2$/.exec(file.slice(prefix.length))?.[1]
      : undefined;
    if (file === undefined || subset === undefined) {
      throw new Error(`not a Fontsource variable font face: ${rule.trim()}`);
    }
    return {
      file,
      subset,
      weight: value("font-weight"),
      style: value("font-style"),
      unicodeRange: value("unicode-range").split(","),
    };
  });
}

export function facesFor(faces: readonly Face[], subsets: readonly string[]): Face[] {
  return faces.filter(face => subsets.includes(face.subset));
}

// Astro's font provider for a Fontsource package already installed for the slides theme and the editor: the files come
// from node_modules, so the build fetches nothing. Unlike the built-in `local` provider, each face keeps its subset,
// so <Font preload> can preload the latin files only. (The built-in `npm` provider downloads the same files from
// jsDelivr during the build.)
export function fontsource(packageName: string): FontProvider {
  let directory = "";
  return {
    name: `fontsource:${packageName}`,
    init({ root }) {
      directory = dirname(createRequire(fileURLToPath(root)).resolve(`${packageName}/index.css`));
    },
    async resolveFont({ subsets }) {
      const css = await readFile(join(directory, "index.css"), "utf-8");
      return {
        fonts: facesFor(parseFaces(css, packageName.replace(/^@[^/]+\//, "")), subsets).map(face => ({
          src: [{ url: join(directory, face.file), format: "woff2" }],
          weight: face.weight,
          style: face.style,
          unicodeRange: face.unicodeRange,
          meta: { subset: face.subset },
        })),
      };
    },
  };
}
