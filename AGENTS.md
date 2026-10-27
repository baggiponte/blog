# General directives

* Titles, both for the blog and the articles, should be lowercase.
* Work on feature branches, never directly on `main`
* Delete branches after merging
* Commit frequently and atomically - Each commit should represent one logical change

# Marimo notebook posts

* Edit post metadata in the bundle's `index.md`, and notebook content in its `.py`.
* Never hand-edit a notebook's generated `.md`; regenerate it with `just marimo-export`.
* Keep exactly one notebook `.py` per bundle directory.
* Create notebook posts with `just notebook` and add dependencies from marimo's package manager; never write or edit a notebook's PEP 723 header by hand.

# Talks

* Add talks and editions in `content/talks/talks.json`; deck links are `/talks/<slug>/`.
* Keep Slidev decks in `decks/` as imported, apart from build fixes.
* Run `pnpm lint`, `pnpm typecheck`, `pnpm test:unit` and `pnpm test` after changing the talks page or the build.
