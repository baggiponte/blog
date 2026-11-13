# General directives

* Titles, both for the blog and the articles, should be lowercase.
* In posts every line of text is its own paragraph: write one paragraph per line and never hard-wrap prose.
* oxfmt formats the hand-written Markdown in `content/` (pages and each post's `index.md`); Biome formats the code.
* Work on feature branches, never directly on `main`
* Delete branches after merging
* Commit frequently and atomically - Each commit should represent one logical change

# Plugins

* Put each feature the blog adds on top of Astro in its own `src/plugins/<name>/` folder, with its tests and styles, and list its switch in `src/plugins/README.md`.

# Drafting editor

* The editor at `/editor/` lives in `src/plugins/editor/` and only runs in `astro dev`; keep it out of the build.
* It edits hand-written bundles only, writing `index.md` and images next to it; never let it touch notebook posts.

# Marimo notebook posts

* Edit post metadata in the bundle's `index.md`, and notebook content in its `.py`.
* Never hand-edit a notebook's generated `.md`; regenerate it with `just marimo-export`.
* Keep exactly one notebook `.py` per bundle directory.
* Create notebook posts with `just notebook` and add dependencies from marimo's package manager; never write or edit a notebook's PEP 723 header by hand.

# Talks

* Add talks and editions in `content/talks/talks.json`; deck links are `/talks/<slug>/`.
* Keep Slidev decks in `decks/` as imported, apart from build fixes; don't move old decks onto the slides theme unless asked.
* Start new decks with `just talk <slug>` so they use `themes/slides/`; put shared colours and fonts in `src/styles/tokens.css`, not in the theme.
* Run `pnpm lint`, `pnpm typecheck`, `pnpm test:unit` and `pnpm test` after changing the talks page or the build.

# Code

* Run `pnpm lint`, `pnpm typecheck`, `pnpm test:unit` and `pnpm test` (after `pnpm build`) after changing code.
* Keep logic in `.ts` modules and `.astro` frontmatter to wiring: `tsc` doesn't check `.astro` files yet.
* Write relative imports with their `.ts` extension; Node runs the build script and the unit tests without a bundler.
* Fix lint errors rather than silencing them; a `biome-ignore` needs a reason that says why the rule doesn't apply.
* Test behaviour, not the renderer restating itself: a test must be able to fail when the code is wrong.
