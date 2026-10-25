# talks

Slides for Luca Baggi's talks, published at <https://talks.baggiponte.com/> as a static site on Cloudflare Workers.

- `decks/<slug>/`: Slidev decks, kept as imported apart from build fixes.
- `site/`: the catalog page, its data (`talks.json`, `venues.json`, `cities.json`) and the organisers' logos.
- `scripts/`: the build. Node runs the TypeScript directly; the data schemas are in `scripts/data.ts`.

## develop

Needs the Node version in `.node-version` and the pnpm version in `package.json`.

```sh
pnpm install --frozen-lockfile
uvx --from prek==0.5.7 prek install      # Biome, Zizmor and Gitleaks on commit
pnpm exec playwright install chromium    # once, for pnpm test
```

The commands are the `scripts` in `package.json`: `format`, `lint`, `typecheck`, `test:unit`, `build`, `test`
(browser tests against the local Workers runtime) and `check` (build and a dry-run deploy). CI runs them all on every
PR. To edit one deck with hot reload: `pnpm exec slidev decks/<slug>/slides.md`.

## add a talk

Add it to `site/talks.json`, or add an edition to an existing talk's `given` list. `slides` is `/<slug>/` for a deck in
`decks/` or any URL. The build fails with the file and the entry when something doesn't match `scripts/data.ts`, a deck
isn't listed, or a venue has no city.

A venue's `city` and `logo` come from the longest name prefix in `site/venues.json` that sets them, so `"PyData London
2026"` takes its logo from `"PyData"` and its city from `"PyData London"`. Cities and their coordinates are in
`site/cities.json`; `"online"` needs no entry. The logos are the organisers' own marks, shown only to name the venue.

## deploy

Cloudflare Workers Builds deploys `main` to the `talks` Worker on every push. Its settings live in the dashboard:

| setting | value |
| --- | --- |
| Git repository | `baggiponte/talks`, production branch `main` |
| Build command | `pnpm build` |
| Deploy command | `pnpm exec cf deploy` |
| Non-production branch builds | off (CI checks PRs) |

Roll back by restoring a previous Worker version in Cloudflare. To deploy by hand and smoke-test production:

```sh
pnpm exec cf auth login
pnpm deploy
TALKS_BASE_URL=https://talks.baggiponte.com pnpm test
```

The deploy attaches the custom domain itself; don't pre-create a DNS record.
