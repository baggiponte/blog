import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";

test.beforeAll(() => execFileSync(process.execPath, ["scripts/redirects.mjs"]));

for (const slug of ["pymi-pyo3", "pymi-polars", "pymi-cookiecutter"]) {
  test(`${slug}: legacy URLs retain slide, query and fragment on a fixed destination`, async ({ page }) => {
    const root = `https://talks.baggiponte.com/${slug}/`;
    const index = await readFile(`redirect-sites/${slug}/site/index.html`, "utf8");
    const fallback = await readFile(`redirect-sites/${slug}/site/404.html`, "utf8");
    await page.route("https://baggiponte.github.io/**", route => route.fulfill({
      status: route.request().url().includes("/5") ? 404 : 200,
      contentType: "text/html",
      body: route.request().url().includes("/5") ? fallback : index,
    }));
    await page.route("https://talks.baggiponte.com/**", route => route.fulfill({
      contentType: "text/html", body: "<h1>destination</h1>",
    }));
    for (const [suffix, target] of [["", ""], ["/", ""], ["/index.html", ""], ["/5", "5"], ["/5/", "5/"], ["/presenter/5", "presenter/5"], ["//evil.example/5", "/evil.example/5"]]) {
      const query = "?clicks=2&next=https%3A%2F%2Fevil.example#notes";
      await page.goto(`https://baggiponte.github.io/${slug}${suffix}${query}`);
      await expect(page).toHaveURL(`${root}${target}${query}`);
      await expect(page.getByRole("heading", { name: "destination" })).toBeVisible();
    }
  });

  test(`${slug}: no-JavaScript visitors receive a working fallback`, async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    const page = await context.newPage();
    const html = await readFile(`redirect-sites/${slug}/site/404.html`, "utf8");
    await page.route("https://baggiponte.github.io/**", route => route.fulfill({ contentType: "text/html", body: html, status: 404 }));
    await page.route("https://talks.baggiponte.com/**", route => route.fulfill({ contentType: "text/html", body: "<h1>destination</h1>" }));
    await page.goto(`https://baggiponte.github.io/${slug}/5?clicks=2#notes`);
    await expect(page).toHaveURL(`https://talks.baggiponte.com/${slug}/`);
    await context.close();
  });
}
