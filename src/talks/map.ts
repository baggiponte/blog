// Geometry for the map view: Europe as a grid of dots, and where each city falls on it.
import { geoConicConformal, geoPath } from "d3-geo";
import { feature } from "topojson-client";
import land from "world-atlas/land-50m.json" with { type: "json" };

export const width = 960;
export const height = 620;
const step = 8;

// Lisbon to Vilnius, Scotland to Sicily: every city must land inside this frame.
const projection = geoConicConformal()
  .parallels([40, 56])
  .rotate([-8, 0])
  .fitExtent(
    [
      [40, 30],
      [width - 40, height - 30],
    ],
    {
      type: "MultiPoint",
      coordinates: [
        [-10.5, 38.6],
        [27.5, 56.4],
        [-6, 56.4],
        [25, 39],
      ],
    },
  )
  .clipExtent([
    [0, 0],
    [width, height],
  ]);

export function locate(name: string, at: [number, number]): [number, number] {
  const [x, y] = projection(at) ?? [NaN, NaN];
  if (!(x >= 0 && x <= width && y >= 0 && y <= height))
    throw new Error(`content/talks/cities.json: ${name} falls outside the map`);
  return [x, y];
}

// One dot per grid cell whose centre is on land, as a single path of zero-length round-capped strokes.
export function landDots() {
  // The JSON import is typed by its contents; topojson-client wants its own Topology type.
  const topology = land as unknown as Parameters<typeof feature>[0];
  const coastlines = topology.objects.land;
  if (coastlines === undefined) {
    throw new Error("world-atlas/land-50m.json has no land object");
  }
  const rings: [number, number][][] = [];
  let ring: [number, number][] = [];
  geoPath(projection, {
    moveTo(x, y) {
      ring = [[x, y]];
      rings.push(ring);
    },
    lineTo(x, y) {
      ring.push([x, y]);
    },
    closePath() {},
    beginPath() {},
    arc() {},
  })(feature(topology, coastlines));
  // Even-odd ray casting against the projected, clipped coastlines.
  const inside = (x: number, y: number) => {
    let hit = false;
    for (const polygon of rings) {
      // Each edge runs from the previous point to this one, starting with the closing edge.
      let [xj, yj] = polygon.at(-1) ?? [0, 0];
      for (const [xi, yi] of polygon) {
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
        [xj, yj] = [xi, yi];
      }
    }
    return hit;
  };
  let d = "";
  for (let y = step / 2; y < height; y += step)
    for (let x = step / 2; x < width; x += step) {
      if (inside(x, y)) d += `M${x} ${y}h0`;
    }
  return d;
}
