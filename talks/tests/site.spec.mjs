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
  expect((await request.get("/not-a-talk")).status()).toBe(404);
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
