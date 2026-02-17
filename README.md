# Blog

## Preview

With [just](https://github.com/casey/just)

```bash
just serve
```

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
