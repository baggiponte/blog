# Blog

## Preview

With [just](https://github.com/casey/just)

```bash
just serve
```

## Cloudflare preview

GitHub Pages remains production and the rollback host. The Cloudflare Worker is
named `blog-preview`, has no custom domains or routes, and cannot change apex or
`www` DNS through the checked-in configuration.

Use Hugo Extended 0.164.0 (matching Pages CI), Node >=22.18, and pnpm 12.15.0:

```bash
pnpm install --frozen-lockfile
pnpm check           # clean production Hugo build and Cloudflare dry-run
pnpm dev             # serve the static assets locally
pnpm deploy:preview  # deploy only the isolated workers.dev preview
```

`cf auth login` requires browser authorization before deployment. The build keeps
production canonical URLs, generates permanent redirects for directory URLs, and
uses Hugo's custom 404 page instead of an SPA fallback. Only `public/` is uploaded.
Explicit `index.html` requests may normalize to directory URLs on Cloudflare.
Workers.dev responses have `X-Robots-Tag: noindex`; custom domains are unaffected.
Do not publish a worktree containing unfinished content: use a clean checkout of
the reviewed commit for deployment.

Before a production cutover, save DNS/proxy/routing settings, verify edge TLS for
apex and `www`, and test preview content, assets, search, RSS, sitemap, redirects,
and real 404s. Keep Pages' custom domain, HTTPS, and deployment intact. Production
DNS and Worker domain changes require a separate reviewed cutover; this preview
configuration does not perform them. Supersede the sibling zone ADR-0002 then.

To roll back a cutover, restore DNS-only apex A records
`185.199.{108,109,110,111}.153`, AAAA records `2606:50c0:800{0,1,2,3}::153`,
and `www CNAME baggiponte.github.io`, and remove the migration's Cloudflare
routes/redirects/custom domains as needed. Keep the last-good Worker alive while
cached Cloudflare DNS answers expire. Do not disable Pages before the rollback
window closes.

## Repository writes

Active branch and tag rulesets restrict creation, updates, and deletion to the
repository admin role (the owner). Separate rulesets prohibit force pushes even
for the owner: use new branches instead of force-pushing rebased branches. Bots
have no bypass, so dependency updates require owner-managed branches and merges.
Actions remains enabled for Pages with read-only default token permissions and
only GitHub-owned actions allowed. Enable mandatory SHA pinning only after the
pinned Pages workflow reaches `main`, so rollback deployments remain runnable.

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
