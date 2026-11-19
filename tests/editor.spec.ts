import { mkdir, readFile, rm, writeFile } from "node:fs/promises";

import { expect, type Page, test } from "@playwright/test";

// These tests write real files: each makes its own post in content/blog and removes it afterwards.
const blogDir = new URL("../content/blog/", import.meta.url);
const indexOf = (dir: string): URL => new URL(`${dir}/index.md`, blogDir);

let dirs: string[] = [];
test.afterEach(async () => {
  for (const dir of dirs) {
    await rm(new URL(`${dir}/`, blogDir), { recursive: true, force: true });
  }
  dirs = [];
});

async function newPost(page: Page, title: string): Promise<string> {
  await page.goto("/editor/");
  await page.getByRole("button", { name: "+ new" }).click();
  await page.getByLabel("title").fill(title);
  await page.getByLabel("description").fill("written by the editor tests");
  await page.getByRole("button", { name: "create" }).click();
  const dir = title.toLowerCase().replaceAll(" ", "-");
  dirs.push(dir);
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

test("Claude's suggestions show on the text, and accepting or rejecting one updates both files", async ({ page }) => {
  const post = await newPost(page, `editor suggestions ${Date.now()}`);
  await writeFile(indexOf(post), `${await readFile(indexOf(post), "utf8")}they was here and teh end\n`);
  const suggestionsOf = new URL(`${post}/suggestions.json`, blogDir);
  const suggestions = [
    { id: "a", kind: "grammar", quote: "they was here", replacement: "they were here", note: "agreement" },
    { id: "b", kind: "typo", quote: "teh end", replacement: "the end" },
  ];
  await writeFile(suggestionsOf, JSON.stringify({ suggestions }));
  // Back on the tab after Claude wrote them.
  await page.evaluate(() => dispatchEvent(new Event("focus")));
  await expect(page.locator(".cm-sg-old")).toHaveText(["was", "teh"]);

  await page.locator(".cm-sg-new", { hasText: "were" }).click();
  await expect(page.locator(".cm-sg-card")).toContainText("agreement");
  await page.getByRole("button", { name: "accept" }).click();
  await page.locator(".cm-sg-new", { hasText: "the" }).click();
  await page.getByRole("button", { name: "reject" }).click();

  await expect(status(page)).toHaveText("saved");
  await expect.poll(() => readFile(indexOf(post), "utf8")).toContain("they were here and teh end");
  // None left: the file goes away.
  await expect.poll(() => readFile(suggestionsOf, "utf8").catch(() => "gone")).toBe("gone");
});

test("opening a post's suggestions never writes them back", async ({ page }) => {
  const post = await newPost(page, `editor reads suggestions ${Date.now()}`);
  await writeFile(indexOf(post), `${await readFile(indexOf(post), "utf8")}they was here\n`);
  const suggestionsOf = new URL(`${post}/suggestions.json`, blogDir);
  const written = JSON.stringify({
    suggestions: [{ id: "a", kind: "grammar", quote: "they was here", replacement: "they were here" }],
  });
  await writeFile(suggestionsOf, written);
  const saves: string[] = [];
  page.on("request", request => {
    if (request.method() === "PUT" && request.url().endsWith("/suggestions")) {
      saves.push(request.url());
    }
  });
  await page.evaluate(() => dispatchEvent(new Event("focus")));
  await expect(page.locator(".cm-sg-old")).toHaveText(["was"]);
  // Longer than the editor waits before saving a change to the suggestions.
  await page.waitForTimeout(1000);
  expect(saves).toEqual([]);
  expect(await readFile(suggestionsOf, "utf8")).toBe(written);
});

test("a pasted image is saved next to index.md and linked in the post", async ({ page }) => {
  const post = await newPost(page, `editor images ${Date.now()}`);
  await page.locator(".cm-content").press("ControlOrMeta+End");
  // What the browser hands the page when a screenshot is pasted.
  await page.locator(".cm-content").evaluate(content => {
    const data = new DataTransfer();
    data.items.add(new File([new Uint8Array([137, 80, 78, 71])], "Screenshot 1.PNG", { type: "image/png" }));
    content.dispatchEvent(new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true }));
  });
  await expect(status(page)).toHaveText("saved");
  await expect.poll(() => readFile(indexOf(post), "utf8")).toContain("![](./screenshot-1.png)");
  expect(await readFile(new URL(`${post}/screenshot-1.png`, blogDir))).toHaveLength(4);
});

test("the sidebar lists the drafts and opens the post clicked", async ({ page }) => {
  // One post already on disk, then one made in the editor, which opens it.
  const older = `editor sidebar ${Date.now()}`;
  const olderDir = older.replaceAll(" ", "-");
  dirs.push(olderDir);
  await mkdir(new URL(`${olderDir}/`, blogDir));
  await writeFile(
    indexOf(olderDir),
    `---\ntitle: "${older}"\ndescription: "written by the editor tests"\ndate: 2026-01-01\ndraft: true\n---\n\nthe older post\n`,
  );
  await newPost(page, `${older} newer`);

  const sidebar = page.getByRole("complementary", { name: "posts" });
  const row = sidebar.getByRole("link", { name: new RegExp(older) }).filter({ hasNotText: "newer" });
  await expect(row).toContainText("draft");
  await row.click();
  await expect(page).toHaveURL(new RegExp(`#${olderDir}$`));
  await expect(page.locator(".cm-content")).toContainText("the older post");
  await expect(row).toHaveAttribute("aria-current", "page");
});
