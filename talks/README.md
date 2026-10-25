# talks

Slides for Luca Baggi's talks, published at <https://talks.baggiponte.com/>
with Cloudflare Workers Static Assets.

- `decks/<slug>/`: slide Markdown, CSS and referenced local assets.
- `site/talks.json`: every talk and each time it was given; the build renders it
  into `site/index.html`, newest first, with repeats collapsed into one entry.
- `site/`: catalog template and 404 page.
- `scripts/build.mjs`: builds every deck with one pinned Slidev into `dist/<slug>/`.
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
its `given` list. The build fails if a deck is missing from `talks.json` or a
`/<slug>/` link has no deck. The tests pick up new decks automatically.

Venue logos live in `site/logos/` and are matched to a venue by the longest name
prefix in `site/venues.json` (`"PyData"` covers London, Global and Amsterdam), which
also records where each file came from. They are the organisers' own marks, taken
from their public website repositories and resized to small icons; the trademarks
belong to their owners and are shown only to name the venue. A venue with no
entry renders without a logo.

The map view places each venue in a city by the longest name prefix in
`site/places.json`, which also holds each city's coordinates (and which side its
label goes on). A venue mapped to `"online"` joins the online pill. The build fails
if a venue has no entry or a city falls outside the map of Europe; the land dots
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
