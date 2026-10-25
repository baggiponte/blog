# Pinned so local exports, the pre-commit hook and CI write identical Markdown
marimo := "0.25.2"
prek := "0.5.7"

# Run marimo notebook export hook, then launch a local hugo server
serve:
    uvx --from prek=={{ prek }} prek run marimo-export-notebooks
    hugo server --buildDrafts --disableFastRender

# Update hugo modules
update:
    hugo mod get -u
    hugo mod get -u github.com/imfing/hextra

    uvx --from prek=={{ prek }} prek autoupdate

# Create a new blog post
new slug:
    hugo new content/{{ slug }}.md && nvim $_

# Create a new TIL
til slug:
    hugo new content/til/{{ slug }}.md && nvim $_

# Create a notebook post with a PEP 723 header, then open it in a sandbox
notebook slug name *packages:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="content/blog/{{ slug }}"
    notebook="$dir/{{ name }}.py"
    if [ -e "$dir" ]; then
      echo "$dir already exists"
      exit 1
    fi

    hugo new "$dir/index.md"
    printf '\n{{{{< include-md "%s" >}}\n' "{{ name }}.md" >> "$dir/index.md"
    printf 'import marimo\n\napp = marimo.App(width="full")\n\n\nif __name__ == "__main__":\n    app.run()\n' > "$notebook"
    uv add --script "$notebook" marimo {{ packages }}
    uvx marimo=={{ marimo }} edit --sandbox "$notebook"

# Add dependencies to a notebook's PEP 723 header
notebook-add notebook +packages:
    uv add --script {{ notebook }} {{ packages }}

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
