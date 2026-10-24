# Blog

## Preview

With [just](https://github.com/casey/just)

```bash
just serve
```

## Cloudflare preview

The default Cloudflare Worker is `blog-preview`, with no custom domains or
routes. Preview commands cannot change production routing. GitHub Pages remains
deployed and configured for the custom domain as the rollback host.

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

## Production and rollback

Explicit production mode deploys a separate `blog` Worker with only the route
`baggiponte.com/*`. It never matches sibling subdomains and has no workers.dev URL:

```bash
pnpm check:production   # production configuration dry-run
pnpm deploy:production  # publishes production assets and the exact apex route
```

There is no automatic Cloudflare production deployment. Deploy only a clean,
reviewed main checkout. Updating Pages alone does not update the Worker.

Cutover order (DNS and redirect rules are separate from the application config):

1. Snapshot all DNS, Worker routes/domains, redirects, and SSL settings. Confirm
   no competing apex route, active Universal SSL for apex/www, and valid TLS
   using forced-edge requests before changing DNS. Verify Pages is healthy
   through the edge while the route is absent. Effective origin SSL must be
   Full or Full (strict), never Flexible, to avoid a Pages HTTPS redirect loop
   during fallback. Do not change sibling hosts' SSL configuration.
2. Deploy the production Worker while the eight apex GitHub records remain
   DNS-only. Test its route using `curl --resolve baggiponte.com:443:<edge-ip>`:
   compare content/assets, RSS, sitemap, search, real404s, redirects, and canonical
   URLs; ensure production has no preview-only `X-Robots-Tag: noindex`.
3. Add exact-host Single Redirects: any `www.baggiponte.com` request → HTTPS apex,
   and HTTP apex → HTTPS apex, both 301 with original path/query preserved. Do
   not enable zone-wide HTTPS redirects or overwrite existing unrelated rules.
4. Proxy the existing `www` CNAME and all eight apex A/AAAA records **in place**.
   Preserve their GitHub targets; do not delete records or attach a conflicting
   managed Custom Domain. Old cached DNS answers continue reaching live Pages.
5. Verify authoritative/public DNS, valid edge TLS, HTTP/HTTPS/www behavior,
   complete content and browser behavior. Confirm every unrelated DNS record,
   Worker domain and route is unchanged. This supersedes the sibling zone
   ADR-0002's original DNS-only blog decision.

Immediate rollback: delete only the exact `baggiponte.com/*` Worker route. Proxied
traffic then reaches the unchanged GitHub Pages origin, even while DNS caches
still contain Cloudflare IPs. Confirm Pages origin TLS and content before doing
so. Restore the nine original records to `proxied:false, ttl:300`: apex A
`185.199.{108,109,110,111}.153`, AAAA `2606:50c0:800{0,1,2,3}::153`, and
`www CNAME baggiponte.github.io`. Leave the host-scoped redirects working during
cache expiry; remove only these migration rules after DNS-only resolution is
established. Keep the Worker, Pages custom domain, HTTPS, and deployment intact
throughout the rollback window. Do not redeploy production mode during rollback,
because doing so reinstalls the route. DNS propagation is not globally atomic.

## Repository writes

Active branch and tag rulesets restrict creation, updates, and deletion to the
repository admin role (the owner). Separate rulesets prohibit force pushes even
for the owner: use new branches instead of force-pushing rebased branches. Bots
have no bypass, so dependency updates require owner-managed branches and merges.
Actions remains enabled for Pages with read-only default token permissions.
Allowed actions are GitHub-owned actions and the exact pinned revision of
`astral-sh/setup-uv` used by CI. Mandatory SHA pinning is enabled, including nested
actions. The Pages upload wrapper also pins its internal upload action.

Install local hooks with `uvx --from prek==0.5.7 prek install`. The pinned Zizmor
hook audits workflows without suppressions; the pinned Gitleaks hook scans staged
changes before commits. CI also scans the complete Git history with Gitleaks and
requires both security checks and the Hugo build before deploying Pages. Hugo's
downloaded installer is checksum-verified before installation. Only main-branch
pushes or main-branch manual dispatches can deploy; PRs cannot publish.

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
