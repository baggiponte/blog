import { expect, type Page, test } from "@playwright/test";

import { tools } from "../tools/catalog.ts";

// tools.baggiponte.com, deployed or served from tools/dist/ by the second web server in playwright.config.ts.
const base = process.env.TOOLS_BASE_URL ?? "http://localhost:8791";
const url = (path: string): string => new URL(path, base).href;

// Errors a page throws or logs while loading, so a broken script fails the test instead of a blank tool.
function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") {
      errors.push(message.text());
    }
  });
  return errors;
}

test("the index links every tool, and each one loads without errors", async ({ page }) => {
  await page.goto(url("/"));
  const links = page.locator(".tool-list a");
  await expect(links).toHaveCount(tools.length);
  for (const tool of tools) {
    const errors = collectErrors(page);
    await page.goto(url("/"));
    await page.getByRole("link", { name: tool.name }).click();
    await expect(page).toHaveURL(url(`/${tool.slug}/`));
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(`${tool.name}.`);
    await expect(page.getByRole("link", { name: "tools" })).toHaveAttribute("aria-current", "page");
    expect(errors, tool.slug).toEqual([]);
  }
});

test("a tool's address without the slash redirects permanently", async ({ request }) => {
  const response = await request.get(url("/metronomo"), { maxRedirects: 0 });
  expect(response.status()).toBe(301);
  expect(response.headers().location).toMatch(/\/metronomo\/$/);
});

test("the theme choice carries from the index to a tool", async ({ page }) => {
  await page.goto(url("/"));
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "toggle dark mode" }).click();
  await page.goto(url("/accordatore/"));
  await expect(root).toHaveAttribute("data-theme", "dark");
});

test("the formatter previews bold text as the unicode linkedin keeps", async ({ page }) => {
  await page.goto(url("/formatter/"));
  const editor = page.getByRole("textbox", { name: "post text" });
  await editor.click();
  await page.keyboard.press("ControlOrMeta+B");
  await page.keyboard.type("Hi 2");
  await expect(page.locator("#previewBody")).toHaveText("𝗛𝗶 𝟮");
});

test("the metronome's buttons and tap tempo set the bpm", async ({ page }) => {
  await page.goto(url("/metronomo/"));
  const bpm = page.locator("#bpm");
  await expect(bpm).toHaveText("96");
  await page.getByRole("button", { name: "aumenta i bpm" }).click();
  await expect(bpm).toHaveText("97");
  await expect(page.locator("#mark")).toHaveText("andante");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await expect(bpm).toHaveText("95");
});
