This blog uses hugo with a custom frontend theme.

# General directives

* Titles, both for the blog and the articles, should be lowercase.
* Work on feature branches, never directly on `main`
* Delete branches after merging
* Commit frequently and atomically - Each commit should represent one logical change

# Marimo Notebook Posts

This blog supports notebook-backed posts under `content/blog/<slug>/`.

Each notebook post is a Hugo page bundle with this layout:

```text
content/blog/<slug>/
  index.md                 # Canonical Hugo page metadata (title/date/tags/draft)
  <notebook_name>.py       # Marimo notebook source
  <notebook_name>.md       # Generated from marimo export (do not hand-edit)
```

`index.md` should include the generated Markdown with:

```md
{{< include-md "<notebook_name>.md" >}}
```

The include shortcode lives at:

- `layouts/shortcodes/include-md.html`

It reads the sibling Markdown file, strips its front matter, and renders the remaining Markdown inside the page.

# Export Commands

Use `just` recipes from `justfile`:

- `just serve`: export changed notebooks via pre-commit hook, then run Hugo server
- `just marimo-export content/blog/<slug>/<notebook_name>.py`: export one notebook bundle
- `just marimo-export`: export all notebook bundles under `content/blog`

## Editing Rules

- Edit metadata in `index.md`.
- Edit notebook code/content in `<notebook_name>.py`.
- Do not manually edit `<notebook_name>.md`; regenerate it.
- Keep exactly one notebook `*.py` file per notebook bundle directory.
