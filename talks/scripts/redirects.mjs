import { mkdir, readdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = new URL("../", import.meta.url);
for (const slug of await readdir(new URL("decks/", root))) {
  const destination = `https://talks.baggiponte.com/${slug}/`;
  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>slides have moved</title>
  <link rel="canonical" href="${destination}">
  <script>
    const base = ${JSON.stringify(`/${slug}`)};
    const path = location.pathname;
    let suffix = path === base ? "" : path.slice((base + "/").length);
    if (suffix === "index.html" || suffix === "404.html") suffix = "";
    if (path === base || path.startsWith(base + "/")) {
      const target = new URL(${JSON.stringify(destination)});
      target.pathname += suffix;
      target.search = location.search;
      target.hash = location.hash;
      location.replace(target.href);
    }
  </script>
  <noscript><meta http-equiv="refresh" content="0;url=${destination}"></noscript>
</head>
<body><p>These slides have moved to <a href="${destination}">${destination}</a>.</p></body>
</html>
`;
  const directory = new URL(`redirect-sites/${slug}/`, root);
  await mkdir(new URL("site/", directory), { recursive: true });
  await mkdir(new URL(".github/workflows/", directory), { recursive: true });
  for (const name of ["index.html", "404.html"]) {
    await writeFile(new URL(`site/${name}`, directory), html);
  }
  await writeFile(new URL("site/.nojekyll", directory), "");
  await writeFile(new URL(".github/workflows/pages.yaml", directory), `name: Publish legacy slide redirects
on:
  workflow_dispatch:
permissions:
  contents: read
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  publish:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: \${{ steps.deployment.outputs.page_url }}
    permissions:
      contents: read
      pages: write
      id-token: write
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - uses: actions/deploy-pages@v4
        id: deployment
`);
  console.log(`Prepared only (not published): ${fileURLToPath(directory)}`);
}
