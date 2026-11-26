import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

import satori from "satori";
import sharp from "sharp";

// The link preview (Open Graph and X/Twitter card) every post gets: the title in the site's display font on the
// light theme's paper, ending in the accent dot that also closes the "luca baggi." brand.

export const width = 1200;
export const height = 630;

export interface Palette {
  paper: string;
  ink: string;
  accent: string;
}

// The light theme's colours from src/styles/tokens.css, so the previews follow when the site's palette changes.
export function lightPalette(css: string): Palette {
  const root = /:root\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
  const token = (name: string): string => {
    const value = new RegExp(`--${name}:\\s*([^;]+);`).exec(root)?.[1]?.trim();
    if (value === undefined) {
      throw new Error(`tokens.css has no --${name} in its first :root block`);
    }
    return value;
  };
  return { paper: token("paper"), ink: token("ink"), accent: token("accent") };
}

// Long titles step down so they still fit in three lines or so; the post page's h1 uses the same weight and tracking.
export function titleSize(title: string): number {
  if (title.length <= 30) {
    return 120;
  }
  if (title.length <= 60) {
    return 100;
  }
  return 76;
}

export interface Font {
  name: string;
  data: Buffer;
  weight: 800;
  style: "normal";
}

// Satori reads WOFF but not WOFF2 or variable fonts, so the previews use the static 800 cut from @fontsource; the
// latin-ext file covers accented letters the latin one lacks.
export function displayFonts(): Promise<Font[]> {
  const require = createRequire(import.meta.url);
  return Promise.all(
    ["latin", "latin-ext"].map(async subset => ({
      name: "Bricolage Grotesque",
      data: await readFile(
        require.resolve(`@fontsource/bricolage-grotesque/files/bricolage-grotesque-${subset}-800-normal.woff`),
      ),
      weight: 800 as const,
      style: "normal" as const,
    })),
  );
}

interface Node {
  type: string;
  key: string;
  props: { style?: Record<string, string | number>; children: string | Node[] };
}

// One box per word so Satori wraps between words; the dot rides on the last word, so it never starts a line alone.
function words(title: string, size: number, accent: string): Node[] {
  const all = title.split(/\s+/).filter(Boolean);
  return all.map((word, index) => ({
    type: "div",
    key: String(index),
    props: {
      // The post page's h1 tracking, -0.04em. Satori neither passes letter spacing down nor resolves an em against an
      // inherited font size, so each word sets it as a number of pixels.
      style: { display: "flex", letterSpacing: -0.04 * size },
      children:
        index === all.length - 1
          ? [
              { type: "span", key: "word", props: { children: word } },
              { type: "span", key: "dot", props: { style: { color: accent }, children: "." } },
            ]
          : word,
    },
  }));
}

export async function renderPreview(title: string, palette: Palette, fonts: Font[]): Promise<Buffer> {
  const size = titleSize(title);
  const svg = await satori(
    {
      type: "div",
      key: null,
      props: {
        style: {
          // Satori's boxes are content-box by default, which would push the padding outside the card.
          width,
          height,
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          padding: "88px 96px",
          background: palette.paper,
        },
        children: {
          type: "div",
          key: null,
          props: {
            style: {
              display: "flex",
              flexWrap: "wrap",
              columnGap: "0.24em",
              fontFamily: "Bricolage Grotesque",
              fontWeight: 800,
              fontSize: size,
              lineHeight: 1.02,
              color: palette.ink,
            },
            children: words(title, size, palette.accent),
          },
        },
      },
    },
    { width, height, fonts },
  );
  return sharp(Buffer.from(svg)).png().toBuffer();
}
