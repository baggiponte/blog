# Run marimo notebook export hook, then launch a local hugo server
serve:
    uv run prek run marimo-export-notebooks
    hugo server --buildDrafts --disableFastRender

# Update hugo modules
update:
    hugo mod get -u
    hugo mod get -u github.com/imfing/hextra

    uv sync --upgrade
    uv run prek autoupdate

# Create a new blog post
new slug:
    hugo new content/{{ slug }}.md && nvim $_

# Create a new TIL
til slug:
    hugo new content/til/{{ slug }}.md && nvim $_

# Export marimo notebooks to Markdown in-place; defaults to all blog notebooks
marimo-export *notebooks:
    #!/usr/bin/env zsh
    notebooks="{{ notebooks }}"

    if [ -z "$notebooks" ]; then
      notebooks="$(fd --extension py --min-depth 2 --max-depth 2 --type f . content/blog)"
    fi

    if [ -z "$notebooks" ]; then
      echo "No notebooks found under content/blog";
      exit 0;
    fi

    for notebook in ${(z)notebooks}; do
      out="${notebook%.py}.md";
      uvx marimo export md "$notebook" -o "$out" -f;
    done
