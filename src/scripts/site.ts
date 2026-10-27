// Theme toggle, ⌘K search, code block buttons, copy page, rotating bio.
const root = document.documentElement;

document.querySelector(".theme-toggle")?.addEventListener("click", () => {
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch {}
});

// ---- search -------------------------------------------------------------
type Hit = { title: string; description: string; date: string; url: string; text: string };
const modal = document.querySelector<HTMLElement>("[data-search]")!;
const input = modal.querySelector<HTMLInputElement>("input")!;
const list = modal.querySelector<HTMLUListElement>("ul")!;
let index: Hit[] | null = null;
let selected = 0;

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function render() {
  // Every word must appear in the title or the text, in any order; posts matching more words in the title come first, then newest.
  const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
  const titleScore = (p: Hit) => words.filter((w) => p.title.toLowerCase().includes(w)).length;
  const hits = (index ?? [])
    .filter((p) => words.every((w) => p.title.toLowerCase().includes(w) || p.text.includes(w)))
    .sort((a, b) => titleScore(b) - titleScore(a));
  selected = Math.min(selected, Math.max(hits.length - 1, 0));
  list.innerHTML =
    hits
      .map(
        (p, i) =>
          `<li role="option" aria-selected="${i === selected}"><a href="${p.url}">${esc(p.title)}<small>${p.date}</small></a></li>`,
      )
      .join("") || `<li class="empty">nothing yet. maybe i should write that one.</li>`;
}

async function openSearch() {
  modal.hidden = false;
  input.value = "";
  selected = 0;
  input.focus();
  index ??= await fetch("/search.json").then((r) => r.json());
  render();
}
const closeSearch = () => (modal.hidden = true);

document.querySelector("[data-search-open]")?.addEventListener("click", openSearch);
input.addEventListener("input", () => {
  selected = 0;
  render();
});
modal.addEventListener("click", (e) => {
  if (e.target === modal) closeSearch();
});
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    modal.hidden ? openSearch() : closeSearch();
    return;
  }
  if (modal.hidden) return;
  const items = list.querySelectorAll("a");
  if (e.key === "Escape") closeSearch();
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    selected = (selected + (e.key === "ArrowDown" ? 1 : items.length - 1)) % Math.max(items.length, 1);
    render();
  }
  if (e.key === "Enter" && items[selected]) location.href = items[selected].href;
});

// ---- code blocks --------------------------------------------------------
async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function flash(button: HTMLElement, label: string, done: string) {
  button.textContent = done;
  button.classList.add("done");
  setTimeout(() => {
    button.textContent = label;
    button.classList.remove("done");
  }, 1400);
}

document.querySelectorAll<HTMLElement>("figure.code").forEach((figure) => {
  const pre = figure.querySelector("pre")!;
  const caption = figure.querySelector("figcaption")!;
  const wrap = Object.assign(document.createElement("button"), { type: "button", className: "code-btn", textContent: "wrap" });
  const copy = Object.assign(document.createElement("button"), { type: "button", className: "code-btn", textContent: "copy" });
  wrap.setAttribute("aria-pressed", "false");
  caption.append(wrap, copy);

  // The fade on the right edge shows while there is code left to scroll to.
  const edge = () =>
    figure.classList.toggle(
      "more",
      !figure.classList.contains("wrapped") && pre.scrollLeft + pre.clientWidth < pre.scrollWidth - 2,
    );
  wrap.addEventListener("click", () => {
    const on = figure.classList.toggle("wrapped");
    wrap.setAttribute("aria-pressed", String(on));
    wrap.textContent = on ? "unwrap" : "wrap";
    edge();
  });
  copy.addEventListener("click", async () => {
    if (await copyText(pre.innerText)) flash(copy, "copy", "copied");
  });
  pre.addEventListener("scroll", edge, { passive: true });
  new ResizeObserver(edge).observe(pre);
});

// ---- copy page ----------------------------------------------------------
const cp = document.querySelector<HTMLElement>("[data-copy-page]");
if (cp) {
  const main = cp.querySelector<HTMLButtonElement>(".cp-main")!;
  const label = main.querySelector("span")!;
  const more = cp.querySelector<HTMLButtonElement>(".cp-more")!;
  const menu = cp.querySelector<HTMLElement>(".cp-menu")!;
  const source = cp.dataset.copyPage!;
  // Fetch early so the click handler can write to the clipboard synchronously enough for Safari.
  const markdown = fetch(source).then((r) => r.text());
  const setMenu = (open: boolean) => {
    menu.hidden = !open;
    more.setAttribute("aria-expanded", String(open));
  };
  const copyPage = async () => {
    if (await copyText(await markdown)) flash(label, "copy page", "copied");
  };
  main.addEventListener("click", copyPage);
  more.addEventListener("click", (e) => {
    e.stopPropagation();
    setMenu(Boolean(menu.hidden));
  });
  menu.querySelector("[data-act=copy]")?.addEventListener("click", () => {
    copyPage();
    setMenu(false);
  });
  document.addEventListener("click", (e) => {
    if (!cp.contains(e.target as Node)) setMenu(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setMenu(false);
  });
}

// ---- rotating bio -------------------------------------------------------
const lately = document.querySelector<HTMLElement>("[data-lately]");
if (lately && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  const lines: string[] = JSON.parse(lately.dataset.lately!);
  let i = 0;
  setInterval(() => {
    lately.classList.add("swap");
    setTimeout(() => {
      i = (i + 1) % lines.length;
      lately.textContent = lines[i];
      lately.classList.remove("swap");
    }, 300);
  }, 5000);
}
