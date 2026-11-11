import { readFile, rm, writeFile } from "node:fs/promises";

import { expect, type Page, test } from "@playwright/test";

// These tests write real files: each makes its own post in content/blog and removes it afterwards.
const blogDir = new URL("../content/blog/", import.meta.url);
const indexOf = (dir: string): URL => new URL(`${dir}/index.md`, blogDir);

let dir: string | undefined;
test.afterEach(async () => {
  if (dir !== undefined) {
    await rm(new URL(`${dir}/`, blogDir), { recursive: true, force: true });
  }
});

async function newPost(page: Page, title: string): Promise<string> {
  await page.goto("/editor/");
  await page.getByRole("button", { name: "+ new" }).click();
  await page.getByLabel("title").fill(title);
  await page.getByLabel("description").fill("written by the editor tests");
  await page.getByRole("button", { name: "create" }).click();
  dir = title.toLowerCase().replaceAll(" ", "-");
  await expect(page).toHaveURL(new RegExp(`#${dir}$`));
  return dir;
}

const status = (page: Page) => page.locator("#status");

test("a new post is typed into and saved to its index.md", async ({ page }) => {
  const post = await newPost(page, `editor saves ${Date.now()}`);
  await page.locator(".cm-content").press("ControlOrMeta+End");
  await page.keyboard.type("hello from the editor");
  await expect(status(page)).toHaveText("saved");
  expect(await readFile(indexOf(post), "utf8")).toContain("hello from the editor");
});

test("a save never overwrites a file changed elsewhere", async ({ page }) => {
  const post = await newPost(page, `editor conflicts ${Date.now()}`);
  const before = await readFile(indexOf(post), "utf8");
  const theirs = `${before}written in another editor\n`;
  await writeFile(indexOf(post), theirs);

  await page.locator(".cm-content").press("ControlOrMeta+End");
  await page.keyboard.type("mine");
  await expect(status(page)).toContainText("changed elsewhere");
  expect(await readFile(indexOf(post), "utf8")).toBe(theirs);
});

test("frontmatter the site can't build is not saved", async ({ page }) => {
  const post = await newPost(page, `editor validates ${Date.now()}`);
  const before = await readFile(indexOf(post), "utf8");
  await page.locator(".cm-content").press("ControlOrMeta+Home");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("End");
  // An unclosed quote in the title.
  await page.keyboard.press("Backspace");
  await expect(status(page)).toContainText("isn't valid YAML");
  expect(await readFile(indexOf(post), "utf8")).toBe(before);
});

test("a post with no unsaved typing reloads when it changed elsewhere", async ({ page }) => {
  const post = await newPost(page, `editor reloads ${Date.now()}`);
  await writeFile(indexOf(post), `${await readFile(indexOf(post), "utf8")}written in another editor\n`);
  // What the browser fires on coming back to the tab.
  await page.evaluate(() => dispatchEvent(new Event("focus")));
  await expect(page.locator(".cm-content")).toContainText("written in another editor");
  await expect(status(page)).toContainText("reloaded");
});
