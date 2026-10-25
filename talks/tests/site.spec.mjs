import { readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { parser } from "@slidev/cli";
import { expect, test } from "@playwright/test";

const decks = [];
for (const slug of (await readdir("decks")).sort()) {
  const directory = resolve("decks", slug);
  const data = await parser.load({ roots: [directory], userRoot: directory }, resolve(directory, "slides.md"));
  decks.push({ slug, count: data.slides.length });
}

test("catalog links every deck and unknown paths are real 404s", async ({ page, request }) => {
  expect((await page.goto("/")).status()).toBe(200);
  for (const { slug } of decks) {
    await expect(page.locator(`main a[href="/${slug}/"]`).first()).toBeVisible();
  }
  const logos = await page.locator("img.logo").evaluateAll(images => images.map(image => image.getAttribute("src")));
  expect(logos.length).toBeGreaterThan(0);
  for (const src of new Set(logos)) expect((await request.get(src)).status(), src).toBe(200);
  expect((await request.get("/not-a-talk")).status()).toBe(404);
});

test("timeline lights up every appearance of the talk under the pointer", async ({ page }) => {
  await page.goto("/#timeline");
  const dot = page.locator('.axis button[data-talk="0"]').first();
  const count = await page.locator('.axis button[data-talk="0"]').count();
  await dot.hover();
  await expect(page.locator(".axis .on")).toHaveCount(count);
  await expect(page.locator(".timeline li.on")).toHaveCount(count);
  await expect(page.locator(".timeline > article")).toHaveCount(0);
  await expect(page.locator("main > article").first()).toBeHidden();
  const total = await page.locator(".axis button").count();
  await expect(page.locator(".timeline .total strong")).toHaveText(`${total} talks`);
});

test("map shows a city's talks and lights up every city a talk went to", async ({ page }) => {
  await page.goto("/#map");
  const milan = page.locator('.place[data-city="Milan"]');
  await milan.hover();
  const given = await page.locator('.map li[data-city="Milan"]').count();
  await expect(page.locator('.card.on[data-city="Milan"] li')).toHaveCount(given);
  await expect(page.locator(".map li.on")).toHaveCount(given);
  await page.locator('.map li[data-talk="0"]').first().click();
  const editions = await page.locator('.map li[data-talk="0"]').count();
  await expect(page.locator(".map li.on")).toHaveCount(editions);
  await expect(page.locator(".place.on").first()).toBeVisible();
  await expect(page.locator(".timeline")).toBeHidden();
  const total = await page.locator(".map li[data-talk]").count();
  await expect(page.locator(".map .total strong")).toHaveText(`${total} talks`);
});

for (const { slug, count } of decks) {
  test(`${slug}: cover, direct slides, final slide and 404s`, async ({ page, request }) => {
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("response", response => {
      if (new URL(response.url()).origin === new URL(page.url()).origin && response.status() >= 400) {
        errors.push(`${response.status()} ${response.url()}`);
      }
    });
    expect((await page.goto(`/${slug}/`)).status()).toBe(200);
    await expect(page.locator(".slidev-page-1")).toBeVisible();
    expect((await page.goto(`/${slug}/${count}?clicks=1`)).status()).toBe(200);
    await expect(page.locator(`.slidev-page-${count}`)).toBeVisible();
    for (let slide = 1; slide <= count; slide++) {
      expect((await request.get(`/${slug}/${slide}`, { maxRedirects: 0 })).status(), `slide ${slide}`).toBe(200);
    }
    for (const suffix of [`${count + 1}`, "assets/missing.js", "not-a-slide"]) {
      expect((await request.get(`/${slug}/${suffix}`)).status(), suffix).toBe(404);
    }
    expect(errors).toEqual([]);
  });
}
