# Recipes that create content. Running the site (dev, build, tests, slides theme) lives in package.json's scripts.

# Pinned so local exports, the pre-commit hook and CI write identical Markdown
marimo := "0.25.2"

set quiet

# Create a draft post from its title (`just new my first post`) and open it in the editor
[positional-arguments]
new +title:
    just _draft post "$@"

# Create a draft tidbits entry from its title and open it in the editor
[positional-arguments]
tidbit +title:
    just _draft tidbits "$@"

# Write a hand-written draft like the editor's "new post" button, then start astro dev on it at /editor/
[positional-arguments]
[private]
_draft format +title:
    #!/usr/bin/env bash
    set -euo pipefail

    dir=$(node scripts/new-post.ts "$@")
    echo "content/blog/$dir/index.md is ready; opening /editor/#$dir"
    pnpm dev --open "/editor/#$dir"

# Create a draft notebook post (`just notebook analysis my title`) and open it in marimo, which writes the PEP 723 header
[positional-arguments]
notebook name +title:
    #!/usr/bin/env bash
    set -euo pipefail

    shift
    dir=$(node scripts/new-post.ts --notebook "{{ name }}.md" post "$@")
    uvx marimo=={{ marimo }} edit --sandbox "content/blog/$dir/{{ name }}.py"

# Start a deck on the slides theme; list it in content/talks/talks.json before building
talk slug:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="decks/{{ slug }}"
    if [ -e "$dir" ]; then
      echo "$dir already exists"
      exit 1
    fi

    mkdir -p "$dir"
    {
      echo "---"
      echo "theme: ../../themes/slides"
      echo "layout: cover"
      echo "title: \"{{ replace(slug, '-', ' ') }}\""
      echo "event: \"\""
      echo "date: \"\""
      echo "---"
      echo
      echo "# {{ replace(slug, '-', ' ') }}"
    } > "$dir/slides.md"
    pnpm exec slidev "$dir/slides.md"

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
