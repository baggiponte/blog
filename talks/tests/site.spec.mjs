import { expect, test } from "@playwright/test";

const decks = [
  { slug: "pymi-pyo3", count: 19, cover: "Use Rust in your Python Package", fifth: "Building Python + Rust Packages" },
  { slug: "pymi-polars", count: 31, cover: "A Lightning-fast DataFrame library for Rust and Python", fifth: "modern" },
  { slug: "pymi-cookiecutter", count: 21, cover: "A formidable library to build coding project templates", fifth: "Time for a demo" },
];

test("catalog links to the three decks and unknown paths are real 404s", async ({ page, request }) => {
  expect((await page.goto("/")).status()).toBe(200);
  await expect(page.getByRole("heading", { name: "talks", exact: true })).toBeVisible();
  for (const { slug } of decks) {
    await expect(page.locator(`h2 a[href="/${slug}/"]`)).toBeVisible();
  }
  expect((await request.get("/not-a-talk")).status()).toBe(404);
  expect((await request.get("/scripts/build.mjs")).status()).toBe(404);
  expect((await request.get("/redirect-sites/pymi-pyo3/site/index.html")).status()).toBe(404);
});

for (const { slug, count, cover, fifth } of decks) {
  test(`${slug}: cover, direct slide, refresh, final slide and missing assets`, async ({ page, request }) => {
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("response", response => {
      if (new URL(response.url()).origin === new URL(page.url()).origin && response.status() >= 400) {
        errors.push(`${response.status()} ${response.url()}`);
      }
    });
    expect((await page.goto(`/${slug}/`)).status()).toBe(200);
    await expect(page.locator(".slidev-page-1")).toContainText(cover);
    for (const image of await page.locator(".slidev-page-1 img").all()) {
      await expect(image).toBeVisible();
      await expect.poll(() => image.evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true);
    }
    const route = `/${slug}/5?via=migration`;
    expect((await page.goto(route)).status()).toBe(200);
    await expect(page).toHaveURL(new RegExp(`/${slug}/5\\?via=migration$`));
    await expect(page.locator(".slidev-page-5")).toContainText(fifth);
    expect((await page.reload()).status()).toBe(200);
    await expect(page.locator(".slidev-page-5")).toContainText(fifth);
    // All three third slides have click steps. A static slide may discard an invalid clicks parameter.
    expect((await page.goto(`/${slug}/3?clicks=1&via=migration`)).status()).toBe(200);
    await expect(page.locator(".slidev-page-3")).toBeVisible();
    await expect(page).toHaveURL(new RegExp(`/${slug}/3\\?clicks=1&via=migration$`));
    expect((await page.goto(`/${slug}/${count}`)).status()).toBe(200);
    await expect(page.locator(`.slidev-page-${count}`)).toContainText("Thank you");
    for (let slide = 1; slide <= count; slide++) {
      const response = await request.get(`/${slug}/${slide}`, { maxRedirects: 0 });
      expect(response.status(), `direct slide ${slide}`).toBe(200);
    }
    expect((await request.get(`/${slug}/5/`, { maxRedirects: 0 })).status()).toBe(200);
    for (const suffix of [`${count + 1}`, "assets/missing.js", "missing.svg", "not-a-slide"]) {
      expect((await request.get(`/${slug}/${suffix}`)).status(), suffix).toBe(404);
    }
    expect(errors).toEqual([]);
  });
}
