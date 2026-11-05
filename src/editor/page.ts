// The editor page. The dev server sends it as is, outside Astro and without Vite's HMR client: a save changes
// content/blog, which makes Astro reload every page that's open, and only the preview should reload, never the
// page being typed in. Vite still serves the script and stylesheet it links.
export const editorPage = `
<!doctype html>
<html lang="en" data-theme="light">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="robots" content="noindex" />
    <title>drafts · luca baggi's blog</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <link rel="stylesheet" href="/src/editor/editor.css" />
    <script is:inline>
      try {
        const saved = localStorage.getItem("theme");
        if (saved === "dark" || saved === "light") { document.documentElement.dataset.theme = saved; }
      } catch {}
    </script>
  </head>
  <body>
    <header class="bar">
      <a class="brand" href="/">drafts<i>.</i></a>
      <select id="post" aria-label="post"></select>
      <button id="create" type="button">new post</button>
      <output id="status" aria-live="polite"></output>
      <button id="vim" type="button" aria-pressed="false" title="vim keys">vim</button>
      <button id="show-preview" type="button" aria-pressed="false" title="show the post as the site renders it">preview</button>
      <a id="open" href="/" target="_blank" rel="noopener">open post ↗</a>
    </header>

    <main class="panes">
      <section id="source" aria-label="markdown"></section>
      <iframe id="preview" title="preview" hidden></iframe>
    </main>

    <dialog id="new" aria-labelledby="new-h">
      <form method="dialog">
        <h2 id="new-h">new post</h2>
        <label>title <input name="title" required autocomplete="off" /></label>
        <label>description <textarea name="description" required rows="2"></textarea></label>
        <fieldset>
          <legend>format</legend>
          <label><input type="radio" name="format" value="post" checked /> post</label>
          <label><input type="radio" name="format" value="tidbits" /> tidbits</label>
        </fieldset>
        <p id="new-error" role="alert"></p>
        <div class="actions">
          <button id="cancel" type="button">cancel</button>
          <button type="submit">create</button>
        </div>
      </form>
    </dialog>

    <script type="module" src="/src/editor/client.ts"></script>
  </body>
</html>
`;
