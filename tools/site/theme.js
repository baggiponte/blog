// The theme toggle in the header: the page applies the saved theme before paint, this only flips and saves it.
// Tools that draw with the palette listen for the "themechange" event to redraw.
// A block, so its names don't collide with a tool's own top-level ones.
{
  const root = document.documentElement;
  document.querySelector(".theme-toggle")?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Private windows and blocked storage: the choice lasts until the page is left.
    }
    dispatchEvent(new Event("themechange"));
  });
}
