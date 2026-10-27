import { type APIRequestContext, expect, test } from "@playwright/test";

import type { SearchEntry } from "../src/lib/search.ts";

// The posts the site was built with, newest first.
async function searchIndex(request: APIRequestContext): Promise<SearchEntry[]> {
  const response = await request.get("/search.json");
  expect(response.ok()).toBe(true);
  return (await response.json()) as SearchEntry[];
}

test("⌘K finds a post by its title and opens it from the keyboard", async ({ page, request }) => {
  const [post] = await searchIndex(request);
  if (post === undefined) {
    throw new Error("no published posts");
  }
  await page.goto("/");
  await page.keyboard.press("ControlOrMeta+K");
  const box = page.getByRole("combobox", { name: "search posts" });
  await expect(box).toBeFocused();

  await box.fill("no post has these words zqxj");
  await expect(page.getByRole("option")).toHaveCount(0);
  await expect(box).not.toHaveAttribute("aria-activedescendant");

  // The title's words in reverse order still match.
  await box.fill(post.title.split(" ").toReversed().join(" "));
  const option = page.getByRole("option", { name: post.title });
  await expect(option).toHaveAttribute("aria-selected", "true");
  await expect(box).toHaveAttribute("aria-activedescendant", (await option.getAttribute("id")) ?? "");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(post.url);
});

test("the theme choice survives a reload", async ({ page }) => {
  await page.goto("/");
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "toggle dark mode" }).click();
  await expect(root).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(root).toHaveAttribute("data-theme", "dark");
});

test("copy page copies the same Markdown as view as markdown", async ({ page, context, request }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const [post] = await searchIndex(request);
  if (post === undefined) {
    throw new Error("no published posts");
  }
  await page.goto(post.url);
  const source = await page
    .getByRole("menuitem", { name: /view as markdown/, includeHidden: true })
    .getAttribute("href");
  const markdown = await request.get(source ?? "");
  expect(markdown.headers()["content-type"]).toBe("text/markdown; charset=utf-8");
  const text = await markdown.text();
  expect(text.startsWith(`# ${post.title}\n`)).toBe(true);

  await page.getByRole("button", { name: "copy page" }).click();
  await expect(page.getByRole("button", { name: "copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(text);
});

test("a directory without its slash redirects permanently, and unknown posts are 404s", async ({ request }) => {
  const about = await request.get("/about", { maxRedirects: 0 });
  expect(about.status()).toBe(301);
  expect(about.headers().location).toBe("/about/");
  expect((await request.get("/blog/not-a-post/")).status()).toBe(404);
});
