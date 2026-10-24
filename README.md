# Blog

## Preview

With [just](https://github.com/casey/just)

```bash
just serve
```

## Deploy

Production is the Cloudflare Worker `blog` on `baggiponte.com` and
`www.baggiponte.com`. Deployment is manual, from a clean checkout of `main`.
Use Hugo Extended 0.164.0 (matching CI), Node >=22.18, and pnpm 12.15.0:

```bash
pnpm install --frozen-lockfile
pnpm check     # clean production Hugo build and Cloudflare dry-run
pnpm dev       # serve the static assets locally
pnpm exec cf auth login
pnpm deploy
```

The build generates permanent redirects for directory URLs and serves Hugo's
404 page; only `public/` is uploaded. Roll back by restoring a previous Worker
version in Cloudflare.

## Checks

Install local hooks with `uvx --from prek==0.5.7 prek install`: Zizmor audits
workflows and Gitleaks scans staged changes. CI runs Zizmor, a full-history
Gitleaks scan and the Hugo build on PRs and `main`, with read-only permissions.
Rulesets protect `main` from deletion and force pushes.

## Update

With [Just](https://github.com/casey/just)

```bash
just update
```

## Marimo Notebook Posts

Notebook-backed posts are stored as Hugo page bundles under `content/blog/<slug>/`.

Expected bundle layout:

```text
content/blog/<slug>/
  index.md                 # Hugo front matter and include shortcode
  <notebook_name>.py       # Marimo notebook source
  <notebook_name>.md       # Generated Markdown from marimo export
```

`index.md` owns metadata (`title`, `date`, `tags`, `draft`, etc.) and includes the generated body:

```md
{{< include-md "<notebook_name>.md" >}}
```

The shortcode implementation is in `layouts/shortcodes/include-md.html`.
It reads the sibling generated Markdown, strips its front matter, and renders the rest in the page.

### Export Commands

```bash
just marimo-export content/blog/<slug>/<notebook_name>.py  # export one notebook bundle
just marimo-export content/blog/*/*.py                      # export many bundles
just marimo-export                                          # export all bundles under content/blog
```

Notes:

- Keep one notebook `*.py` file per bundle directory.
- Treat generated `*.md` files as build outputs from marimo; regenerate instead of hand-editing.
