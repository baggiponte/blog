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
| Deploy command | `pnpm run deploy` |
| Non-production branch builds | off (GitHub CI checks PRs) |

Node and pnpm come from `package.json`. The build runs `astro build` into
`dist/`, generates permanent redirects for directory URLs, then packages `dist/`
as static assets in `.cloudflare/output/`, which `cf deploy --prebuilt` uploads.
`cf build` is not used directly because it detects Astro and expects the
Cloudflare adapter, which a fully static site does not need. Roll back by
restoring a previous Worker version in Cloudflare.

To build or deploy locally (Node >=22.18, pnpm 12.15.0):

```bash
pnpm install --frozen-lockfile
pnpm check     # production build and Cloudflare dry-run
pnpm dev       # astro dev server, drafts included
pnpm preview   # serve the production build locally
```

## Checks

Install local hooks with `uvx --from prek==0.5.7 prek install`: Zizmor audits
workflows and Gitleaks scans staged changes. CI runs Zizmor, the notebook export
check and the production build on PRs, with read-only permissions; `main` is
checked by Workers Builds itself. GitHub secret scanning with push protection
covers secrets on push. Rulesets protect `main` from deletion and force pushes.

## Update

With [Just](https://github.com/casey/just)

```bash
just update
```

## Writing

```bash
just new <slug>              # long post
just tidbit <slug>           # tidbits entry
just notebook <slug> <name>  # marimo notebook post
```

Each entry is a bundle under `content/blog/<slug>/` whose `index.md` holds the
front matter: `title`, `description`, `date`, `format` (`post` or `tidbits`),
`draft` and, for notebook posts, `notebook`. New entries start as drafts, which
show up only in `pnpm dev`. Wrap text in `==double equals==` to highlight it.
Every post is also served as raw Markdown at `/blog/<slug>.md`, which the copy
page button uses.

## Marimo Notebook Posts

```text
content/blog/<slug>/
  index.md                 # front matter, with `notebook: <notebook_name>.md`
  <notebook_name>.py       # Marimo notebook source
  <notebook_name>.md       # Generated Markdown from marimo export
```

The post body is the generated Markdown named by `notebook`; `index.md` has no
body of its own. A missing export fails the build with a hint to run
`just marimo-export`.

### Export Commands

```bash
just marimo-export content/blog/<slug>/<notebook_name>.py  # export one notebook bundle
just marimo-export content/blog/*/*.py                      # export many bundles
just marimo-export                                          # export all bundles under content/blog
```

Notes:

- Keep one notebook `*.py` file per bundle directory.
- Treat generated `*.md` files as build outputs from marimo; regenerate instead of hand-editing.
