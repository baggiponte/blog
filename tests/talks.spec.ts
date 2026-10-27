import { expect, test } from "@playwright/test";

import { listDecks } from "../src/talks/decks.ts";
import { slideCount } from "../src/talks/slides.ts";

// Playwright runs from the repository root.
const root = process.cwd();
const decks = await Promise.all(
  (await listDecks(root)).map(async slug => ({ slug, count: await slideCount(root, slug) })),
);

test("catalog links every deck and unknown paths are real 404s", async ({ page, request }) => {
  expect((await page.goto("/talks/#talks"))!.status()).toBe(200);
  for (const { slug } of decks) {
    await expect(page.locator(`.talks a[href="/talks/${slug}/"]`).first()).toBeVisible();
  }
  const logos = await page.locator("img.logo").evaluateAll(images => images.map(image => image.getAttribute("src")!));
  expect(logos.length).toBeGreaterThan(0);
  for (const src of new Set(logos)) expect((await request.get(src)).status(), src).toBe(200);
  expect((await request.get("/talks/not-a-talk")).status()).toBe(404);
});

test("timeline lights up every appearance of the talk under the pointer", async ({ page }) => {
  await page.goto("/talks/#timeline");
  const dot = page.locator(".axis button").first();
  const talk = await dot.getAttribute("data-talk");
  const count = await page.locator(`.axis button[data-talk="${talk}"]`).count();
  await dot.hover();
  await expect(page.locator(".axis .on")).toHaveCount(count);
  await expect(page.locator(".timeline li.on")).toHaveCount(count);
  await expect(page.locator(".talks > article").first()).toBeHidden();
  const total = await page.locator(".axis button").count();
  await expect(page.locator(".timeline .total strong")).toHaveText(`${total} talks`);
});

test("map shows a city's talks and lights up every city a talk went to", async ({ page }) => {
  // The map is the default view.
  await page.goto("/talks/");
  const milan = page.locator('.place[data-city="Milan"]');
  await milan.hover();
  const given = await page.locator('.map li[data-city="Milan"]').count();
  await expect(page.locator('.card.on[data-city="Milan"] li')).toHaveCount(given);
  await expect(page.locator(".map li.on")).toHaveCount(given);
  // The card stays open while the pointer crosses the bare map between the circle and the card.
  const card = page.locator('.card.on[data-city="Milan"]');
  const circle = (await milan.boundingBox())!;
  const edge = (await card.boundingBox())!;
  await page.mouse.move(edge.x + 8, circle.y + circle.height / 2, { steps: 20 });
  await expect(card).toBeVisible();
  const row = page.locator(".map li[data-talk]").first();
  const talk = await row.getAttribute("data-talk");
  await row.click();
  const editions = await page.locator(`.map li[data-talk="${talk}"]`).count();
  await expect(page.locator(".map li.on")).toHaveCount(editions);
  await expect(page.locator(".place.on").first()).toBeVisible();
  await expect(page.locator(".timeline")).toBeHidden();
  const total = await page.locator(".map li[data-talk]").count();
  await expect(page.locator(".map .total strong")).toHaveText(`${total} talks`);
});

test("the map tab does not jump to the top of the page", async ({ page }) => {
  await page.goto("/talks/");
  await page.mouse.wheel(0, 400);
  await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(0);
  const before = await page.evaluate(() => scrollY);
  // Click in place: a locator click would first scroll the tab into view.
  await page.locator('.views a[data-view="map"]').evaluate((tab: HTMLElement) => tab.click());
  await expect(page).toHaveURL(/#map$/);
  expect(await page.evaluate(() => scrollY)).toBe(before);
});

for (const { slug, count } of decks) {
  test(`${slug}: cover, direct slides, final slide and 404s`, async ({ page, request }) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("response", response => {
      if (new URL(response.url()).origin === new URL(page.url()).origin && response.status() >= 400) {
        errors.push(`${response.status()} ${response.url()}`);
      }
    });
    expect((await page.goto(`/talks/${slug}/`))!.status()).toBe(200);
    await expect(page.locator(".slidev-page-1")).toBeVisible();
    expect((await page.goto(`/talks/${slug}/${count}?clicks=1`))!.status()).toBe(200);
    await expect(page.locator(`.slidev-page-${count}`)).toBeVisible();
    for (let slide = 1; slide <= count; slide++) {
      expect((await request.get(`/talks/${slug}/${slide}`, { maxRedirects: 0 })).status(), `slide ${slide}`).toBe(200);
    }
    for (const suffix of [`${count + 1}`, "assets/missing.js", "not-a-slide"]) {
      expect((await request.get(`/talks/${slug}/${suffix}`)).status(), suffix).toBe(404);
    }
    expect(errors).toEqual([]);
  });
}
