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

  // The menu beside it closes on Escape and on a click anywhere else.
  const more = page.getByRole("button", { name: "more ways to read this page" });
  const menu = page.getByRole("menu");
  for (const close of [
    () => page.keyboard.press("Escape"),
    () => page.getByRole("heading", { name: post.title }).click(),
  ]) {
    await more.click();
    await expect(menu).toBeVisible();
    await expect(more).toHaveAttribute("aria-expanded", "true");
    await close();
    await expect(menu).toBeHidden();
    await expect(more).toHaveAttribute("aria-expanded", "false");
  }
});

test("a directory without its slash redirects permanently, and unknown posts are 404s", async ({ request }) => {
  const about = await request.get("/about", { maxRedirects: 0 });
  expect(about.status()).toBe(301);
  expect(about.headers().location).toBe("/about/");
  expect((await request.get("/blog/not-a-post/")).status()).toBe(404);
});

// Every page the sitemap lists, the way a reader gets there: it loads, says whose blog it is, has content, runs its
// scripts without errors, and links only to pages that exist, at their final address.
test("every page in the sitemap loads cleanly and links only to pages that exist", async ({ page, request }) => {
  const locs = async (path: string): Promise<string[]> =>
    [...(await (await request.get(path)).text()).matchAll(/<loc>([^<]+)<\/loc>/g)].map(
      ([, loc]) => new URL(loc ?? "").pathname,
    );
  const pages = (await Promise.all((await locs("/sitemap-index.xml")).map(locs))).flat();
  expect(pages).toContain("/");

  const errors: string[] = [];
  page.on("pageerror", error => errors.push(`${page.url()}: ${error.message}`));
  const links = new Set<string>();
  for (const path of pages) {
    expect((await page.goto(path))?.status(), path).toBe(200);
    await expect(page, path).toHaveTitle(/(^| · )luca baggi's blog$/);
    await expect(page.locator("main"), path).not.toBeEmpty();
    for (const href of await page.locator('a[href^="/"]').evaluateAll(as => as.map(a => a.getAttribute("href")))) {
      links.add(new URL(href ?? "", page.url()).pathname);
    }
  }
  expect(errors).toEqual([]);
  for (const link of links) {
    expect((await request.get(link, { maxRedirects: 0 })).status(), link).toBe(200);
  }
});

test("code blocks wrap long lines and copy their code", async ({ page, context, request }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  let found = false;
  for (const { url } of await searchIndex(request)) {
    await page.goto(url);
    if ((await page.locator("figure.code").count()) > 0) {
      found = true;
      break;
    }
  }
  expect(found, "no post has a code block").toBe(true);
  const figure = page.locator("figure.code").first();

  const wrap = figure.getByRole("button", { name: "wrap" });
  await expect(wrap).toHaveAttribute("aria-pressed", "false");
  await wrap.click();
  await expect(figure).toHaveClass(/\bwrapped\b/);
  await expect(figure.locator("pre")).toHaveCSS("white-space", /pre-wrap/);
  await figure.getByRole("button", { name: "unwrap" }).click();
  await expect(figure).not.toHaveClass(/\bwrapped\b/);

  await figure.getByRole("button", { name: "copy" }).click();
  await expect(figure.getByRole("button", { name: "copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(await figure.locator("pre").innerText());
});

test("the drafting editor only exists in astro dev, never on the built site", async ({ request }) => {
  for (const path of ["/editor/", "/__editor/posts"]) {
    expect((await request.get(path)).status(), path).toBe(404);
  }
});

test("every response carries the security headers", async ({ request }) => {
  for (const path of ["/", "/talks/", "/favicon.svg"]) {
    const headers = (await request.get(path)).headers();
    expect(headers["x-content-type-options"], path).toBe("nosniff");
    expect(headers["content-security-policy"], path).toBe("frame-ancestors 'self'");
    expect(headers["strict-transport-security"], path).toBe("max-age=31536000");
  }
});

test("hashed build files are cached for a year and pages are checked again on every visit", async ({
  page,
  request,
}) => {
  await page.goto("/");
  const stylesheet = await page.locator('link[rel="stylesheet"][href^="/_astro/"]').first().getAttribute("href");
  expect(stylesheet).not.toBeNull();
  expect((await request.get(stylesheet ?? "")).headers()["cache-control"]).toBe("public, max-age=31536000, immutable");
  expect((await request.get("/")).headers()["cache-control"] ?? "").not.toContain("immutable");
});

test("every post's link preview is a card-sized PNG the site serves", async ({ page, request }) => {
  for (const post of await searchIndex(request)) {
    await page.goto(post.url);
    const image = await page.locator('meta[property="og:image"]').getAttribute("content");
    await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute("content", "summary_large_image");
    if (image === null) {
      throw new Error(`${post.url} has no og:image`);
    }
    // og:image points at the live site, so fetch its path from the server under test.
    const response = await request.get(new URL(image).pathname);
    expect(response.ok(), image).toBe(true);
    expect(response.headers()["content-type"]).toBe("image/png");
  }
});
