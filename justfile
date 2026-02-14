# Launch a local hugo server
start:
    hugo server --buildDrafts --disableFastRender

# Update hugo modules
update:
    hugo mod get -u
    hugo mod get -u github.com/imfing/hextra

# Create a new blog post
new slug:
    hugo new content/{{ slug }}.md && nvim $_

# Create a new TIL
til slug:
    hugo new content/til/{{ slug }}.md && nvim $_

# Export a single marimo notebook page bundle to Markdown
marimo-export slug:
    notebook=$(fd -e py --max-depth 1 --type f . content/blog/{{ slug }} | head -n 1) && \
      test -n "$notebook" && out="${notebook%.py}.md" && \
      uvx marimo export md "$notebook" -o "$out" -f

# Export all marimo notebook page bundles under content/blog
marimo-export-all:
    fd -e py --min-depth 2 --max-depth 2 --type f . content/blog | while read -r notebook; do \
      out="${notebook%.py}.md"; \
      uvx marimo export md "$notebook" -o "$out" -f; \
    done

# Watch and continuously export one marimo notebook bundle
marimo-watch slug:
    notebook=$(fd -e py --max-depth 1 --type f . content/blog/{{ slug }} | head -n 1) && \
      test -n "$notebook" && out="${notebook%.py}.md" && \
      uvx marimo export md "$notebook" -o "$out" --watch -f
