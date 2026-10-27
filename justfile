# Pinned so local exports, the pre-commit hook and CI write identical Markdown
marimo := "0.25.2"
prek := "0.5.7"

# Run marimo notebook export hook, then launch the astro dev server (drafts included)
serve:
    uvx --from prek=={{ prek }} prek run marimo-export-notebooks
    pnpm dev

# Update pre-commit hooks (Renovate handles the rest)
update:
    uvx --from prek=={{ prek }} prek autoupdate

# Create a new blog post
new slug:
    just _bundle "{{ slug }}" post && nvim "content/blog/{{ slug }}/index.md"

# Create a new tidbits entry
tidbit slug:
    just _bundle "{{ slug }}" tidbits && nvim "content/blog/{{ slug }}/index.md"

# Create a notebook post and open it in a sandbox; marimo writes the PEP 723 header
notebook slug name:
    just _bundle "{{ slug }}" post "{{ name }}.md"
    uvx marimo=={{ marimo }} edit --sandbox "content/blog/{{ slug }}/{{ name }}.py"

# Write a draft bundle's index.md; the body is the notebook export when one is given
[private]
_bundle slug format notebook="":
    #!/usr/bin/env bash
    set -euo pipefail

    dir="content/blog/{{ slug }}"
    if [ -e "$dir" ]; then
      echo "$dir already exists"
      exit 1
    fi

    mkdir -p "$dir"
    {
      echo "---"
      echo "title: \"{{ replace(slug, '-', ' ') }}\""
      echo "description: \"\""
      echo "date: $(date +%Y-%m-%d)"
      echo "format: {{ format }}"
      if [ -n "{{ notebook }}" ]; then echo "notebook: {{ notebook }}"; fi
      echo "draft: true"
      echo "---"
    } > "$dir/index.md"

# Export marimo notebooks to Markdown in-place; defaults to all blog notebooks
marimo-export *notebooks:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob

    notebooks=({{ notebooks }})
    if [ ${#notebooks[@]} -eq 0 ]; then
      notebooks=(content/blog/*/*.py)
    fi

    if [ ${#notebooks[@]} -eq 0 ]; then
      echo "No notebooks found under content/blog"
      exit 0
    fi

    # No --sandbox: exporting to Markdown never runs the notebook, so it needs no dependencies
    for notebook in "${notebooks[@]}"; do
      uvx marimo=={{ marimo }} export md "$notebook" -o "${notebook%.py}.md" -f
    done
