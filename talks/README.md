# talks

Slides for Luca Baggi's talks, published at <https://talks.baggiponte.com/>
with Cloudflare Workers Static Assets.

- `decks/<slug>/`: slide Markdown, CSS and referenced local assets.
- `site/talks.json`: every talk and each time it was given; the build renders it
  into `site/index.html`, newest first, with repeats collapsed into one entry.
- `site/`: catalog template and 404 page.
- `scripts/build.ts`: builds every deck with one pinned Slidev into `dist/<slug>/`.
  Node runs the TypeScript directly (type stripping); `tsc` only checks it.
- `scripts/data.ts`: Zod schemas for the `site/*.json` data files.
- `cloudflare.config.ts`: Worker `talks` on `talks.baggiponte.com`.

Decks are imported as historical material, unchanged apart from build fixes. Each
catalog entry links to the original repository at the imported commit; those
repositories now only redirect their old GitHub Pages URLs here.

## develop

Requires Node >=22.18 and pnpm 12.15.0.

```sh
pnpm install --frozen-lockfile
uvx --from prek==0.5.7 prek install    # Zizmor and Gitleaks hooks
pnpm exec slidev decks/<slug>/slides.md  # edit one deck with hot reload
pnpm typecheck                         # tsc, no emit
pnpm test:unit                         # catalog and data checks, no browser (~2 s)
pnpm check                             # build + cf deploy --dry-run
pnpm exec playwright install chromium
pnpm test                              # browser tests against local Workers runtime
```

The build generates exact rewrites for slide, presenter, overview and notes routes.
There is no SPA fallback: unknown paths and out-of-range slides return 404.

To add a talk, add it to `site/talks.json`: `title`, `description` and a `given`
list of `{date, venue, slides}` with optional `title` (if that edition was named
differently; kept for the record, not shown), `recording` and `source`. `slides` is `/<slug>/` for a
deck in `decks/` or any external URL. Another edition of an existing talk goes in
its `given` list. The build checks every file against the schemas in
`scripts/data.ts` and fails if a deck is missing from `talks.json`, a
`/<slug>/` link has no deck, or two titles give the same talk id (the page
keys talks by a slug of the title). The tests pick up new decks automatically.

Everything about a venue lives in `site/venues.json`, keyed by name prefix: its
`city`, its `logo` in `site/logos/` and where that file came from. Each field
comes from the longest prefix that sets it, so `"PyData London 2026"` takes the
logo from `"PyData"` and the city from `"PyData London"`. The logos are the
organisers' own marks, taken from their public website repositories and resized
to small icons; the trademarks belong to their owners and are shown only to name
the venue. A venue with no logo renders without one.

The map view places each venue in its city; `site/cities.json` holds each city's
coordinates (and which side its label goes on). A venue in `"online"` joins the
online pill. The build fails if a venue has no city, the city is not in
`cities.json`, or it falls outside the map of Europe; the land dots
come from Natural Earth via `world-atlas`.

## deploy

Production is the Cloudflare Worker `talks` on `talks.baggiponte.com`, deployed
by Workers Builds on every push to `main`. Its build settings live in the
Cloudflare dashboard:

| setting | value |
| --- | --- |
| Git repository | `baggiponte/talks`, production branch `main` |
| Build command | `pnpm build` |
| Deploy command | `pnpm exec cf deploy` |
| Non-production branch builds | off (GitHub CI checks PRs) |

Node and pnpm come from `package.json`; only `dist/` is uploaded. CI checks
every PR with read-only permissions and no deploy job: Zizmor, then the frozen
install, dry-run and browser tests. GitHub secret scanning with push protection
(and the local Gitleaks hook) covers secrets. Roll back by restoring a previous Worker version in Cloudflare.

To deploy by hand, or to smoke-test production after a build:

```sh
pnpm exec cf auth login
pnpm deploy
TALKS_BASE_URL=https://talks.baggiponte.com pnpm test
```

The custom domain is attached by the deploy; do not pre-create a DNS record.
