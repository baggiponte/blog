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
| Build command | `pnpm build` |
| Deploy command | `pnpm run deploy` |
| Non-production branch builds | off (GitHub CI checks PRs) |

Node and pnpm come from `package.json`. The build runs `astro build` into
`dist/`, where two integrations in `src/plugins/` build the Slidev decks and
write permanent redirects for directory URLs, then packages `dist/`
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
show up only in `pnpm dev`; a draft may keep a capitalised title or an empty
description, which a published post may not. Wrap text in `==double equals==` to highlight it.

A `card` code block becomes a card, anywhere in a post: a bookmark for a page to read, or a song, podcast or video
that opens on Spotify or YouTube. Only `url` is required: the link decides the type (a Spotify track is a song, an
episode a podcast, a YouTube link a video, anything else a link), and without a title or an image the card fills them
in. The build never fetches anything. The editor refuses to save a block the build would refuse.

```card
url: https://jvns.ca/blog/2024/02/16/popular-git-config-options/
type: link                         # optional: link, song, podcast or video
title: popular git config options  # optional
note: the options people set       # optional
image: ./git-config.png            # optional, next to index.md
site: julia evans                  # optional, links only
```

Each line of text is its own paragraph: write one paragraph per line, with no
blank line needed between them (`src/lib/remark-line-paragraphs.ts`), so don't
hard-wrap prose.
Every post is also served as raw Markdown at `/blog/<slug>.md`, which the copy
page button uses.

### Editor

`pnpm dev` (or `just serve`) also serves a drafting editor at
<http://localhost:4321/editor/>. It edits the post's `index.md` in one pane with
a live preview: away from the cursor the Markdown syntax hides and the text
reads like the post (headings, emphasis, `==highlights==`, links, images
inline), while the lines under the cursor show their Markdown to edit. The
"preview" button adds the post as the dev server renders it beside the text,
and "vim" turns on vim keys; both choices are remembered. It saves as you type
(⌘S saves at once). ⌘B, ⌘I, ⌘U and ⌘⇧X toggle bold, italic (`_word_`, as oxfmt writes it), `==highlight==`
and strikethrough on the selection, with or without vim. A sidebar lists the hand-written posts as a file tree,
newest first, with a "draft" tag on drafts; the button left of the title hides
it. "+ new" writes the same draft bundle as `just new`.
Drag or paste images into the text: each one is saved next to `index.md` with
a slugified name and linked as `![](./<name>)`, with the cursor between the
brackets for the alt text. Notebook posts aren't listed, since marimo writes
them. A save never writes frontmatter the site couldn't build (the status bar
says why), and never overwrites an `index.md` changed elsewhere since it was
opened, in another tab or another editor: the status bar says so and saving
stops until the post is reopened. With no unsaved typing, coming back to the
tab reloads a post changed elsewhere.

Proofreading: ask Claude to proofread a post (the `proofread` skill in
`.claude/skills/`) and it writes `suggestions.json` next to `index.md` instead
of editing the post. Back in the editor, each suggestion shows on the text it
quotes, struck through with the replacement after it; the cursor on one opens a
card to accept (⌘↵), edit or reject (⌘⇧↵) it, ⌥] and ⌥[ move between them, and a
panel lists them all. Accepting is an ordinary edit, so ⌘Z undoes it and brings
the suggestion back. Decided suggestions leave the file, which goes away once
none are left; it's gitignored either way.
them. The editor and its API (`src/plugins/editor/`) exist only in the dev server,
never in the built site; nothing is committed, so review and commit the bundle
as usual.
Changes to the editor's own code, page and API included, apply on a browser
refresh, without restarting `astro dev`.

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
built to `/talks/<slug>/` by the `decks()` integration (`src/plugins/decks/`). The decks are kept as imported
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

New decks use the slides theme in `themes/slides/`: the blog's palette, fonts
and code blocks on a slide. It reads the same `src/styles/tokens.css` and Shiki
themes (`src/lib/code-themes.ts`) as the site, so a palette change reaches both.
Start a deck with `just talk <slug>`, then list it in `talks.json`. Layouts:
`cover` (`event`, `date`, and a QR code in the `::aside::` slot), `section`
(`kicker`), `code`, plus Slidev's built-in ones; `footer: false` drops the
brand and slide number from a slide. `just slides-theme` opens the theme's
example deck with hot reload.

The talks used to live in `baggiponte/talks` at `talks.baggiponte.com`; their
history is in this repo's log under `talks/`, before the move into this layout.

## Docs

`docs/` holds standalone reference pages that are not part of the site. Open them
straight from disk in a browser.

- `docs/frontend-glossary.html`: names for the UI elements on this site, with
  annotated screenshots and live examples, for describing design feedback.
