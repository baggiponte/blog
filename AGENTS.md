# AGENTS

## Marimo Notebook Posts

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

## Export Commands

Use `just` recipes from `justfile`:

- `just marimo-export <slug>`: export one notebook bundle
- `just marimo-export-all`: export all notebook bundles under `content/blog`
- `just marimo-watch <slug>`: watch one notebook and continuously re-export

The export target is `*.md` with the same basename as the notebook `*.py` file.

## Editing Rules

- Edit metadata in `index.md`.
- Edit notebook code/content in `<notebook_name>.py`.
- Do not manually edit `<notebook_name>.md`; regenerate it.
- Keep exactly one notebook `*.py` file per notebook bundle directory when using `just marimo-export <slug>`.
