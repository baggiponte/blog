// Geometry for the map view: Europe as a grid of dots, and where each city falls on it.
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { geoConicConformal, geoPath } from "d3-geo";
import { feature } from "topojson-client";

export const width = 960;
export const height = 620;
const step = 8;

// Lisbon to Vilnius, Scotland to Sicily: every city must land inside this frame.
const projection = geoConicConformal().parallels([40, 56]).rotate([-8, 0])
  .fitExtent([[40, 30], [width - 40, height - 30]], { type: "MultiPoint", coordinates: [[-10.5, 38.6], [27.5, 56.4], [-6, 56.4], [25, 39]] })
  .clipExtent([[0, 0], [width, height]]);

export function locate(name: string, at: [number, number]) {
  const [x, y] = projection(at) ?? [NaN, NaN];
  if (!(x >= 0 && x <= width && y >= 0 && y <= height)) throw new Error(`site/cities.json: ${name} falls outside the map`);
  return [x, y];
}

// One dot per grid cell whose centre is on land, as a single path of zero-length round-capped strokes.
export async function landDots() {
  const atlas = createRequire(import.meta.url).resolve("world-atlas/land-50m.json");
  const topology = JSON.parse(await readFile(atlas, "utf8"));
  const rings: [number, number][][] = [];
  let ring: [number, number][] = [];
  geoPath(projection, {
    moveTo(x, y) { rings.push(ring = [[x, y]]); },
    lineTo(x, y) { ring.push([x, y]); },
    closePath() {},
    beginPath() {},
    arc() {},
  })(feature(topology, topology.objects.land));
  // Even-odd ray casting against the projected, clipped coastlines.
  const inside = (x: number, y: number) => {
    let hit = false;
    for (const r of rings) for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i], [xj, yj] = r[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
  let d = "";
  for (let y = step / 2; y < height; y += step) for (let x = step / 2; x < width; x += step) {
    if (inside(x, y)) d += `M${x} ${y}h0`;
  }
  return d;
}
