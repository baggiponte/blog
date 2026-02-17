# Launch a local hugo server
start:
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

# Export one or more marimo notebooks to Markdown in-place
marimo-export +notebooks:
    for notebook in {{ notebooks }}; do \
      out="${notebook%.py}.md"; \
      uvx marimo export md "$notebook" -o "$out" -f; \
    done

# Watch and continuously export one marimo notebook bundle
marimo-watch slug:
    notebook=$(fd -e py --max-depth 1 --type f . content/blog/{{ slug }} | head -n 1) && \
      test -n "$notebook" && out="${notebook%.py}.md" && \
      uvx marimo export md "$notebook" -o "$out" --watch -f
