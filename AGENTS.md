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

- `just marimo-export content/blog/<slug>/<notebook_name>.py`: export one notebook bundle
- `just marimo-export content/blog/*/*.py`: export many notebook bundles
- `just marimo-export`: export all notebook bundles under `content/blog`
- `just serve`: export changed notebooks via pre-commit hook, then run Hugo server

The export target is `*.md` with the same basename as the notebook `*.py` file.

## Editing Rules

- Edit metadata in `index.md`.
- Edit notebook code/content in `<notebook_name>.py`.
- Do not manually edit `<notebook_name>.md`; regenerate it.
- Keep exactly one notebook `*.py` file per notebook bundle directory.

## Maintainer Preferences

Treat this section as decision guidance, not command-by-command policy.

- Prefer the simplest solution that preserves intent; avoid clever plumbing when direct composition is enough.
- Preserve tool contracts end-to-end: when one tool already provides the right inputs, pass them through instead of reshaping them.
- Keep a single source of truth: notebook `*.py` files are authoritative, and generated `*.md` files are reproducible artifacts.
- Favor convention over custom behavior: use Hugo/Hextra defaults first, and customize only when a clear UX goal is unmet.
- Bias toward readability in both code and presentation: explicit, understandable automation and legible visual defaults.
- Prefer fewer, composable commands over many specialized variants when one interface can cover both targeted and bulk workflows.
- In automation recipes, prefer fail-fast and loud behavior over deep defensive guard trees for unlikely corner cases.
- Avoid duplicating selection logic across layers: let hooks select targets, and let recipes execute core actions.
- Keep orchestration linear: one clear pre-step and one clear main step beats conditional retries and branching chains.
- Structure commits as logical, atomic units and use Conventional Commits nomenclature; prefer one minimal atomic commit per task (few files and focused hunks), splitting only when needed, so each change is easy to review and revert.
- At the end of each task, review the conversation for new reusable principles and update this file when needed, while avoiding duplicate or overlapping principles.
