# talks

Slides for Luca Baggi's talks, published at <https://talks.baggiponte.com/>
with Cloudflare Workers Static Assets.

- `decks/<slug>/`: slide Markdown, CSS and referenced local assets.
- `site/`: catalog and 404 page.
- `scripts/build.mjs`: builds every deck with one pinned Slidev into `dist/<slug>/`.
- `cloudflare.config.ts`: Worker `talks` on `talks.baggiponte.com`.

Decks are imported as historical material, unchanged apart from build fixes. Each
catalog entry links to the original repository at the imported commit; those
repositories now only redirect their old GitHub Pages URLs here.

## develop

Requires Node >=22.18 and pnpm 12.15.0.

```sh
pnpm install --frozen-lockfile
prek install                           # Zizmor and Gitleaks hooks
pnpm exec slidev decks/<slug>/slides.md  # edit one deck with hot reload
pnpm check                             # build + cf deploy --dry-run
pnpm exec playwright install chromium
pnpm test                              # browser tests against local Workers runtime
```

The build generates exact rewrites for slide, presenter, overview and notes routes.
There is no SPA fallback: unknown paths and out-of-range slides return 404.

To add a deck, copy it to `decks/<slug>/` and add an entry to `site/index.html`.
The tests pick up new decks automatically.

## deploy

CI checks every PR and push to `main` (Zizmor, frozen install, dry-run, browser
tests) with read-only permissions and no deploy job. Deployment is manual:

```sh
pnpm exec cf auth login
pnpm deploy
TALKS_BASE_URL=https://talks.baggiponte.com pnpm test
```

The custom domain is attached by the deploy; do not pre-create a DNS record.
Roll back by restoring a previous Worker version in Cloudflare.
