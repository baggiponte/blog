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
| Build command | `node scripts/build.ts` |
| Deploy command | `pnpm run deploy` |
| Non-production branch builds | off (GitHub CI checks PRs) |

Node and pnpm come from `package.json`. The build runs `astro build` into
`dist/`, generates permanent redirects for directory URLs, then packages `dist/`
as static assets in `.cloudflare/output/`, which `cf deploy --prebuilt` uploads.
`cf build` is not used directly because it detects Astro and expects the
Cloudflare adapter, which a fully static site does not need. Roll back by
restoring a previous Worker version in Cloudflare.

To build or deploy locally (the Node in `.node-version`, pnpm 12.15.0):

```bash
pnpm install --frozen-lockfile
pnpm check       # production build, decks included, and Cloudflare dry-run
pnpm dev         # astro dev server, drafts included (decks are not served)
pnpm preview     # serve the production build locally
pnpm exec playwright install chromium   # once, for pnpm test
pnpm test        # browser tests against the production build in dist/
```

## Checks

Install local hooks with `uvx --from prek==0.5.7 prek install`: Zizmor audits
workflows, Gitleaks scans staged changes and Biome lints and formats the code
(see `biome.jsonc`; `.astro` files are linted but not formatted yet). `tsc`
checks every `.ts` file with Astro's strictest preset; it can't read `.astro`
files until TypeScript 7.1, so their frontmatter only wires up `.ts` modules.
Unit tests sit next to the code as `src/**/*.test.ts` (`pnpm test:unit`);
browser tests are in `tests/`. CI runs Zizmor, the notebook export check, Biome, tsc, the
unit tests, the production build and the browser tests on PRs, with read-only
permissions; `main` is checked by Workers Builds itself. GitHub secret scanning
with push protection covers secrets on push. Rulesets protect `main` from
deletion and force pushes.

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

## Talks

The talks catalog is at `/talks/`, and each Slidev deck in `decks/<slug>/` is
built to `/talks/<slug>/` by `scripts/build.ts`. The decks are kept as imported
apart from build fixes. To edit one with hot reload:
`pnpm exec slidev decks/<slug>/slides.md`.

To add a talk, add it to `content/talks/talks.json`, or add an edition to an
existing talk's `given` list. `slides` is `/talks/<slug>/` for a deck in
`decks/` or any URL. The build fails with the file and the entry when something
doesn't match `src/talks/data.ts`, a deck isn't listed, or a venue has no city.

A venue's `city` and `logo` come from the longest name prefix in
`content/talks/venues.json` that sets them, so `"PyData London 2026"` takes its
logo from `"PyData"` and its city from `"PyData London"`. Cities and their
coordinates are in `content/talks/cities.json`; `"online"` needs no entry. The
logos in `public/talks/logos/` are the organisers' own marks, shown only to
name the venue.

The talks used to live in `baggiponte/talks` at `talks.baggiponte.com`; their
history is in this repo's log under `talks/`, before the move into this layout.

## Docs

`docs/` holds standalone reference pages that are not part of the site. Open them
straight from disk in a browser.

- `docs/frontend-glossary.html`: names for the UI elements on this site, with
  annotated screenshots and live examples, for describing design feedback.
