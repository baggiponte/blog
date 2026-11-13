# plugins

Everything the blog adds on top of Astro that can be switched on or off. Each folder is one feature with its code, tests and styles. Astro's own pieces (pages, content collections, the Markdown pipeline, images, sitemap, RSS) stay where Astro expects them.

| plugin | what it does | where it runs | switch |
| --- | --- | --- | --- |
| `decks/` | builds every Slidev deck in `decks/` into `/talks/<slug>/` and writes its rewrites to `_redirects` | build | `decks()` in `astro.config.ts` |
| `redirects/` | a permanent `/x → /x/` redirect for every page, instead of Cloudflare's temporary ones | build | `permanentRedirects()` in `astro.config.ts`, after `decks()` |
| `editor/` | the drafting editor at `/editor/` and its API, with proofreading suggestions | `astro dev` only: browser and dev server | `editor()` in `astro.config.ts` |
| `fonts/` | an Astro font provider that serves the installed Fontsource packages, so Astro preloads the latin files | build | `fonts` in `astro.config.ts` |
| `card/` | a ```` ```card ```` block becomes a link, song, podcast or video card | build (HTML), browser (CSS) | `remarkCard` in `src/lib/markdown.ts`, and `card.css` in `src/layouts/Base.astro` |

Astro 7 takes remark plugins from the Markdown processor in `src/lib/markdown.ts`, not from integrations, so a Markdown feature's switch is its line in that list. Switching the card off leaves a ```` ```card ```` block as a YAML code block. The editor reads `card/` to refuse saving a post whose card block the build would reject.
