# Blog

## Preview

With [just](https://github.com/casey/just)

```bash
just serve
```

## Deploy

Production is the Cloudflare Worker `blog` on `baggiponte.com` and
`www.baggiponte.com`, deployed by Workers Builds on every push to `main`.
Its build settings live in the Cloudflare dashboard:

| setting | value |
| --- | --- |
| Git repository | `baggiponte/blog`, production branch `main` |
| Build command | `node scripts/build.mjs` |
| Deploy command | `pnpm exec cf deploy` |
| Non-production branch builds | off (GitHub CI checks PRs) |
| Build variable `HUGO_VERSION` | `0.168.0`, matching `.github/workflows/check.yaml` |

Node and pnpm come from `package.json`. The build generates permanent redirects
for directory URLs and serves Hugo's 404 page; only `public/` is uploaded. Roll
back by restoring a previous Worker version in Cloudflare.

To build or deploy locally (Hugo 0.168.0, Node >=22.18, pnpm 12.15.0):

```bash
pnpm install --frozen-lockfile
pnpm check     # clean production Hugo build and Cloudflare dry-run
pnpm dev       # serve the static assets locally
```

## Checks

Install local hooks with `uvx --from prek==0.5.7 prek install`: Zizmor audits
workflows. CI runs Zizmor and the Hugo build on PRs, with read-only permissions;
`main` is checked by Workers Builds itself. GitHub secret scanning with push
protection covers secrets.
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
