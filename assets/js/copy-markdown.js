document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-markdown]");
  if (!button) return;

  const source = document.querySelector("[data-markdown-source]");
  if (!source) return;

  const original = button.textContent;
  try {
    const markdown = source.value || source.textContent;
    await navigator.clipboard.writeText(markdown.trim() + "\n");
    button.textContent = "Copied";
  } catch {
    button.textContent = "Copy failed";
  }

  window.setTimeout(() => {
    button.textContent = original;
  }, 1600);
});
