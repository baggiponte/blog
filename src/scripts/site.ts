// Theme toggle, ⌘K search, code block buttons, copy page, rotating bio.
import { type SearchEntry, search } from "../lib/search.ts";
import { element, find } from "./dom.ts";

// ---- theme --------------------------------------------------------------
// Base.astro applies the saved theme before the first paint; this only flips and saves it.
const root = document.documentElement;
find(".theme-toggle").addEventListener("click", () => {
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch {
    // Private windows and blocked storage: the choice lasts until the page is left.
  }
});

// ---- search -------------------------------------------------------------
const modal = find("[data-search]");
const input = find<HTMLInputElement>("input", modal);
const list = find<HTMLUListElement>("ul", modal);
// Fetched on first open and kept; dropped on failure so the next open tries again.
let index: Promise<SearchEntry[]> | undefined;
let entries: SearchEntry[] = [];
let selected = 0;

async function loadIndex(): Promise<SearchEntry[]> {
  const response = await fetch("/search.json");
  if (!response.ok) {
    throw new Error(`/search.json: ${response.status}`);
  }
  // Built by pages/search.json.ts from the same type in the same build.
  return (await response.json()) as SearchEntry[];
}

const notice = (text: string): HTMLLIElement => element("li", { className: "empty", textContent: text });

function render(): void {
  const hits = search(entries, input.value);
  selected = Math.min(selected, Math.max(hits.length - 1, 0));
  if (hits.length === 0) {
    list.replaceChildren(notice("nothing yet. maybe i should write that one."));
    return;
  }
  list.replaceChildren(
    ...hits.map((hit, i) => {
      const item = element("li", { role: "option" }, element("a", { href: hit.url }, hit.title, element("small", {}, hit.date)));
      item.setAttribute("aria-selected", String(i === selected));
      return item;
    }),
  );
}

async function openSearch(): Promise<void> {
  modal.hidden = false;
  input.value = "";
  selected = 0;
  input.focus();
  index ??= loadIndex();
  try {
    entries = await index;
  } catch {
    index = undefined;
    list.replaceChildren(notice("search is unavailable right now. try again in a bit."));
    return;
  }
  render();
}

function closeSearch(): void {
  modal.hidden = true;
}

find("[data-search-open]").addEventListener("click", () => {
  void openSearch();
});
input.addEventListener("input", () => {
  selected = 0;
  render();
});
// A click on the backdrop, outside the box, closes the palette.
modal.addEventListener("click", event => {
  if (event.target === modal) {
    closeSearch();
  }
});
document.addEventListener("keydown", event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (modal.hidden) {
      void openSearch();
    } else {
      closeSearch();
    }
    return;
  }
  if (modal.hidden) {
    return;
  }
  const links = list.querySelectorAll("a");
  switch (event.key) {
    case "Escape":
      closeSearch();
      break;
    case "ArrowDown":
    case "ArrowUp":
      event.preventDefault();
      // Wraps around at both ends.
      selected = (selected + (event.key === "ArrowDown" ? 1 : links.length - 1)) % Math.max(links.length, 1);
      render();
      break;
    case "Enter": {
      const link = links[selected];
      if (link !== undefined) {
        location.href = link.href;
      }
      break;
    }
    default:
      break;
  }
});

// ---- code blocks --------------------------------------------------------
async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

// Shows "copied" (or "failed") on the element for a moment, then its label again.
function flash(target: HTMLElement, label: string, copied: boolean): void {
  target.textContent = copied ? "copied" : "failed";
  target.classList.add("done");
  setTimeout(() => {
    target.textContent = label;
    target.classList.remove("done");
  }, 1400);
}

for (const figure of document.querySelectorAll<HTMLElement>("figure.code")) {
  const pre = find("pre", figure);
  const wrap = element("button", { type: "button", className: "code-btn", textContent: "wrap" });
  const copy = element("button", { type: "button", className: "code-btn", textContent: "copy" });
  wrap.setAttribute("aria-pressed", "false");
  find("figcaption", figure).append(wrap, copy);

  // The fade on the right edge shows while there is code left to scroll to.
  const edge = (): void => {
    const hidden = pre.scrollLeft + pre.clientWidth < pre.scrollWidth - 2;
    figure.classList.toggle("more", !figure.classList.contains("wrapped") && hidden);
  };
  wrap.addEventListener("click", () => {
    const wrapped = figure.classList.toggle("wrapped");
    wrap.setAttribute("aria-pressed", String(wrapped));
    wrap.textContent = wrapped ? "unwrap" : "wrap";
    edge();
  });
  copy.addEventListener("click", () => {
    void copyText(pre.innerText).then(copied => flash(copy, "copy", copied));
  });
  pre.addEventListener("scroll", edge, { passive: true });
  new ResizeObserver(edge).observe(pre);
}

// ---- copy page ----------------------------------------------------------
const copyPage = document.querySelector<HTMLElement>("[data-copy-page]");
if (copyPage !== null) {
  const main = find<HTMLButtonElement>(".cp-main", copyPage);
  const label = find("span", main);
  const more = find<HTMLButtonElement>(".cp-more", copyPage);
  const menu = find(".cp-menu", copyPage);
  const source = copyPage.dataset.copyPage ?? "";
  // Fetched early so the click handler can write to the clipboard soon enough for Safari, which only allows it
  // shortly after the click. A failed fetch shows as a failed copy.
  const markdown = fetch(source).then(async response => (response.ok ? response.text() : null));
  const setMenu = (open: boolean): void => {
    menu.hidden = !open;
    more.setAttribute("aria-expanded", String(open));
  };
  const copyMarkdown = async (): Promise<void> => {
    const text = await markdown.catch(() => null);
    flash(label, "copy page", text !== null && (await copyText(text)));
  };
  main.addEventListener("click", () => {
    void copyMarkdown();
  });
  more.addEventListener("click", event => {
    // The document listener below would close the menu again straight away.
    event.stopPropagation();
    // `hidden` can also be "until-found"; any hidden state opens the menu.
    setMenu(menu.hidden !== false);
  });
  find("[data-act=copy]", menu).addEventListener("click", () => {
    setMenu(false);
    void copyMarkdown();
  });
  document.addEventListener("click", event => {
    if (!(event.target instanceof Node && copyPage.contains(event.target))) {
      setMenu(false);
    }
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      setMenu(false);
    }
  });
}

// ---- rotating bio -------------------------------------------------------
const lately = document.querySelector<HTMLElement>("[data-lately]");
// index.astro writes site.lately here as a JSON array of strings.
const lines = JSON.parse(lately?.dataset.lately ?? "[]") as string[];
if (lately !== null && lines.length > 1 && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  let current = 0;
  setInterval(() => {
    lately.classList.add("swap");
    setTimeout(() => {
      current = (current + 1) % lines.length;
      lately.textContent = lines[current] ?? "";
      lately.classList.remove("swap");
    }, 300);
  }, 5000);
}
