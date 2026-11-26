import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import sharp from "sharp";

import { displayFonts, height, lightPalette, renderPreview, titleSize, width } from "./preview.ts";

const tokens = await readFile(new URL("../../styles/tokens.css", import.meta.url), "utf8");

test("the palette is the light theme's, not the dark one's", () => {
  assert.deepEqual(lightPalette(tokens), { paper: "#fdf9ee", ink: "#1c1a14", accent: "#e0562b" });
});

test("a tokens file without a colour fails instead of drawing a default", () => {
  assert.throws(() => lightPalette(":root { --paper: #fff; --ink: #000; }"), /--accent/);
});

test("longer titles get smaller type", () => {
  assert.ok(titleSize("how i use agents") > titleSize("how i use agents, november 2026, with a much longer subtitle"));
});

const hex = (r: number, g: number, b: number): string =>
  `#${[r, g, b].map(channel => channel.toString(16).padStart(2, "0")).join("")}`;

test("the preview is a card-sized PNG on paper, with ink for the title and the accent for its dot", async () => {
  const palette = lightPalette(tokens);
  const png = await renderPreview("how i use agents, november 2026", palette, await displayFonts());
  const { data, info } = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  assert.deepEqual([info.width, info.height], [width, height]);
  const colours = new Set<string>();
  for (let i = 0; i < data.length; i += 3) {
    colours.add(hex(data[i] ?? 0, data[i + 1] ?? 0, data[i + 2] ?? 0));
  }
  assert.equal(hex(data[0] ?? 0, data[1] ?? 0, data[2] ?? 0), palette.paper);
  assert.ok(colours.has(palette.ink), "no ink-coloured pixels: the title is missing");
  assert.ok(colours.has(palette.accent), "no accent-coloured pixels: the dot is missing");
});
