import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { yamlFrontmatter } from "@codemirror/lang-yaml";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { languages } from "@codemirror/language-data";
import { Compartment, Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { tags } from "@lezer/highlight";
import { vim } from "@replit/codemirror-vim";
import { basicSetup } from "codemirror";

import { find as $, element } from "../../scripts/dom.ts";
import { apiBase } from "./base.ts";
import { formatKeys } from "./format.ts";
import { imageInsert } from "./insert.ts";
import { livePreview, postImageUrl, revealOnClick } from "./live-preview.ts";
import type { Format, PostSummary, Version } from "./post.ts";
import {
  ApiError,
  addImage,
  createPost,
  listPosts,
  postUrl,
  readPost,
  readSuggestions,
  saveInit,
  writePost,
  writeSuggestions,
} from "./remote.ts";
import { goTo, load, type Review, reject, remaining, review, reviewField } from "./review.ts";
import { type OnDisk, opened, saveStep, shouldReload } from "./saving.ts";
import { textSelection } from "./selection.ts";
import { changeOf, type Suggestion, withContext } from "./suggestion.ts";

const tree = $<HTMLUListElement>("#posts .tree");
const sidebar = $<HTMLElement>("#posts");
const postsToggle = $<HTMLButtonElement>("#show-posts");
const status = $<HTMLElement>("#status");
const preview = $<HTMLIFrameElement>("#preview");
const open = $<HTMLAnchorElement>("#open");
const dialog = $<HTMLDialogElement>("#new");
const form = $<HTMLFormElement>("#new form");
const pane = $<HTMLElement>("#source");
const vimToggle = $<HTMLButtonElement>("#vim");
const previewToggle = $<HTMLButtonElement>("#show-preview");
const panes = $<HTMLElement>(".panes");
const reviewPanel = $<HTMLElement>("#review");
const reviewToggle = $<HTMLButtonElement>("#show-review");

const saveDelay = 600;
let current: PostSummary | undefined;
let posts: PostSummary[] = [];
let timer: ReturnType<typeof setTimeout> | undefined;
// The open post's file as last read or saved: a save sends its version back, and the API refuses it if the file
// changed elsewhere since (another tab, another editor). The decisions on it are in saving.ts.
let disk: OnDisk | undefined;
// The preview reloads on every save; it comes back where it was.
let previewScroll = 0;

function say(text: string, tone: "ok" | "busy" | "error" = "ok"): void {
  status.textContent = text;
  status.setAttribute("data-tone", tone);
}

async function save(): Promise<void> {
  clearTimeout(timer);
  timer = undefined;
  const text = view.state.doc.toString();
  if (current === undefined || disk === undefined) {
    return;
  }
  const step = saveStep(disk, text);
  if (step !== "write") {
    if (step === "unchanged") {
      say("saved");
    }
    return;
  }
  say("saving…", "busy");
  try {
    rememberScroll();
    disk = { ...disk, text, version: await writePost(current.dir, text, disk.version) };
    say("saved");
    // A new title or draft flag shows in the sidebar.
    await refresh();
  } catch (error) {
    if (error instanceof ApiError && error.failure === "conflict") {
      disk = { ...disk, conflicted: true };
      say("not saved: the file changed elsewhere. Copy your text, then reopen the post", "error");
      return;
    }
    say(`not saved: ${(error as Error).message}`, "error");
  }
}

function rememberScroll(): void {
  try {
    previewScroll = preview.contentWindow?.scrollY ?? previewScroll;
  } catch {
    // A cross-origin page (an error overlay never is, but be safe) keeps the last known position.
  }
}

preview.addEventListener("load", () => {
  preview.contentWindow?.scrollTo(0, previewScroll);
});

// Dropped or pasted images are saved next to index.md, then linked where they landed.
async function addImages(files: File[], position: number): Promise<void> {
  const images = files.filter(file => file.type.startsWith("image/"));
  if (current === undefined || images.length === 0) {
    return;
  }
  const dir = current.dir;
  say(images.length === 1 ? "adding the image…" : `adding ${images.length} images…`, "busy");
  try {
    const names: string[] = [];
    for (const image of images) {
      names.push(await addImage(dir, image));
    }
    if (current?.dir !== dir) {
      return;
    }
    const { text, altFrom } = imageInsert(view.state.doc.toString(), position, names);
    view.dispatch({ changes: { from: position, insert: text }, selection: { anchor: position + altFrom } });
    view.focus();
    say(names.length === 1 ? `added ${names[0]}` : `added ${names.length} images`);
  } catch (error) {
    say(`image not added: ${(error as Error).message}`, "error");
  }
}

// Colours from the site's tokens (editor.css), so the source reads in light and dark alike.
const highlight = HighlightStyle.define([
  { tag: [tags.heading], fontWeight: "700", color: "var(--ink)" },
  { tag: [tags.emphasis], fontStyle: "italic" },
  { tag: [tags.strong], fontWeight: "700" },
  { tag: [tags.strikethrough], textDecoration: "line-through" },
  { tag: [tags.link, tags.url], color: "var(--accent)" },
  {
    tag: [tags.monospace],
    fontFamily: "var(--mono)",
    fontSize: "0.88em",
    background: "var(--code-line)",
    borderRadius: "3px",
  },
  { tag: [tags.quote], color: "var(--muted)", fontStyle: "italic" },
  { tag: [tags.processingInstruction, tags.meta, tags.contentSeparator, tags.punctuation], color: "var(--code-muted)" },
  { tag: [tags.propertyName, tags.definition(tags.propertyName)], color: "var(--accent)" },
  { tag: [tags.keyword, tags.operator], color: "var(--accent)" },
  { tag: [tags.string, tags.number, tags.bool, tags.atom], color: "var(--muted)" },
  { tag: [tags.comment], color: "var(--code-muted)", fontStyle: "italic" },
]);

// Per-browser choices: vim keys, and whether the rendered post shows next to the editor.
function remembered(key: string, fallback: boolean): boolean {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value === "true";
  } catch {
    return fallback;
  }
}
function remember(key: string, value: boolean): void {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // Private windows may refuse; the choice just isn't kept.
  }
}

const keys = new Compartment();
let vimOn = remembered("editor-vim", false);
let previewOn = remembered("editor-preview", false);

const view = new EditorView({
  parent: pane,
  extensions: [
    // Vim's keymap goes before the default one, so its keys win.
    keys.of(vimOn ? vim() : []),
    basicSetup,
    // index.md is YAML frontmatter, then the Markdown body.
    yamlFrontmatter({ content: markdown({ base: markdownLanguage, codeLanguages: languages }) }),
    livePreview(url => postImageUrl(apiBase, current?.dir ?? "", url)),
    revealOnClick,
    review(),
    textSelection,
    syntaxHighlighting(highlight),
    EditorView.lineWrapping,
    // Above the default keymap, which has ⌘I and ⌘U for selecting syntax and undoing selections.
    Prec.high(keymap.of(formatKeys)),
    keymap.of([
      {
        key: "Mod-s",
        preventDefault: true,
        run: () => {
          void save();
          return true;
        },
      },
    ]),
    EditorView.updateListener.of(update => {
      const before = update.startState.field(reviewField);
      const after = update.state.field(reviewField);
      if (before !== after || update.selectionSet) {
        renderReview(after);
      }
      if (before !== after) {
        if (idsOf(before) !== idsOf(after)) {
          scheduleSuggestionsSave();
        }
      }
      if (update.docChanged && current !== undefined) {
        clearTimeout(timer);
        timer = setTimeout(() => void save(), saveDelay);
        say("editing…", "busy");
      }
    }),
    EditorView.domEventHandlers({
      dragover: event => {
        if (event.dataTransfer?.types.includes("Files")) {
          event.preventDefault();
          pane.toggleAttribute("data-dropping", true);
        }
        return false;
      },
      dragleave: () => {
        pane.toggleAttribute("data-dropping", false);
        return false;
      },
      drop: (event, editor) => {
        pane.toggleAttribute("data-dropping", false);
        const files = [...(event.dataTransfer?.files ?? [])];
        if (files.length === 0) {
          return false;
        }
        event.preventDefault();
        const position = editor.posAtCoords({ x: event.clientX, y: event.clientY }) ?? editor.state.selection.main.head;
        void addImages(files, position);
        return true;
      },
      paste: (event, editor) => {
        const files = [...(event.clipboardData?.files ?? [])];
        if (files.length === 0) {
          return false;
        }
        event.preventDefault();
        // At the end of the cursor's line: an image is a paragraph of its own, so it shouldn't split a sentence (in
        // vim's normal mode the cursor sits on a character, not between two).
        void addImages(files, editor.state.doc.lineAt(editor.state.selection.main.head).to);
        return true;
      },
    }),
  ],
});

// Claude's suggestions for the open post (review.ts shows them in the text). Accepting or rejecting one saves the
// rest back to suggestions.json, so a reload doesn't bring it back; the version guards a file Claude rewrote meanwhile.
let suggestionsVersion: Version | undefined;
let suggestionsTimer: ReturnType<typeof setTimeout> | undefined;
let reviewOn = remembered("editor-review", true);

const idsOf = (value: Review): string =>
  remaining(value)
    .map(s => s.id)
    .join(" ");

// The open suggestions, each with its context taken from the text as it is now.
function openSuggestions(): Suggestion[] {
  const { placed, lost } = view.state.field(reviewField);
  const text = view.state.doc.toString();
  return [...placed.map(p => withContext(p.suggestion, text, p.from, p.to)), ...lost];
}

async function loadSuggestions(): Promise<void> {
  if (current === undefined) {
    return;
  }
  try {
    const { suggestions, version } = await readSuggestions(current.dir);
    suggestionsVersion = version;
    view.dispatch({ effects: load.of(suggestions) });
    renderReview(view.state.field(reviewField));
  } catch (error) {
    say(`suggestions not shown: ${(error as Error).message}`, "error");
  }
}

function scheduleSuggestionsSave(): void {
  clearTimeout(suggestionsTimer);
  suggestionsTimer = setTimeout(() => void saveSuggestions(), 300);
}

async function saveSuggestions(): Promise<void> {
  clearTimeout(suggestionsTimer);
  suggestionsTimer = undefined;
  if (current === undefined || suggestionsVersion === undefined) {
    return;
  }
  try {
    suggestionsVersion = await writeSuggestions(current.dir, openSuggestions(), suggestionsVersion);
  } catch (error) {
    if (error instanceof ApiError && error.failure === "conflict") {
      // Claude wrote a new batch while these were open: show those.
      await loadSuggestions();
      say("new suggestions from Claude");
      return;
    }
    say(`suggestions not saved: ${(error as Error).message}`, "error");
  }
}

// Back on the tab after asking Claude to proofread: the new suggestions show up, unless a save of ours is pending.
async function reloadSuggestionsIfNew(): Promise<void> {
  if (current === undefined || suggestionsTimer !== undefined) {
    return;
  }
  const { version } = await readSuggestions(current.dir);
  if (version !== suggestionsVersion) {
    await loadSuggestions();
  }
}

function suggestionRow(suggestion: Suggestion, lost: boolean): HTMLLIElement {
  const { start, end, insert } = changeOf(suggestion.quote, suggestion.replacement);
  const change = element(
    "span",
    { className: "review-change" },
    suggestion.quote.slice(0, start),
    end > start ? element("del", {}, suggestion.quote.slice(start, end)) : "",
    // One line break per paragraph break: the panel is narrow.
    insert === "" ? "" : element("ins", {}, insert.replace(/\n\s*\n/g, "\n")),
    suggestion.quote.slice(end),
  );
  const row = element(
    "li",
    { className: `review-item review-${suggestion.kind}` },
    element("span", { className: "tag" }, suggestion.kind),
    change,
  );
  if (suggestion.note !== undefined) {
    row.append(element("span", { className: "review-note" }, suggestion.note));
  }
  if (lost) {
    const dismiss = element("button", { type: "button", className: "review-dismiss" }, "dismiss");
    dismiss.addEventListener("click", () => reject(view, suggestion.id));
    row.append(dismiss);
  } else {
    row.tabIndex = 0;
    row.addEventListener("click", () => goTo(view, suggestion.id));
    row.addEventListener("keydown", event => {
      if (event.key === "Enter") {
        goTo(view, suggestion.id);
      }
    });
  }
  return row;
}

function renderReview(value: Review): void {
  const count = value.placed.length + value.lost.length;
  reviewToggle.hidden = count === 0;
  $<HTMLElement>(".count", reviewToggle).textContent = String(count);
  reviewToggle.setAttribute("aria-pressed", String(reviewOn));
  reviewPanel.hidden = !(reviewOn && count > 0);
  $<HTMLElement>(".review-list", reviewPanel).replaceChildren(
    ...value.placed.map(p => suggestionRow(p.suggestion, false)),
  );
  $<HTMLElement>(".review-lost", reviewPanel).hidden = value.lost.length === 0;
  $<HTMLElement>(".review-lost-list", reviewPanel).replaceChildren(...value.lost.map(s => suggestionRow(s, true)));
  const active = view.state.field(reviewField).placed.find(p => {
    const { head } = view.state.selection.main;
    return head >= p.from && head <= p.to;
  });
  for (const row of reviewPanel.querySelectorAll(".review-list .review-item")) {
    row.toggleAttribute("data-active", false);
  }
  if (active !== undefined) {
    const index = value.placed.indexOf(active);
    reviewPanel.querySelectorAll(".review-list .review-item")[index]?.toggleAttribute("data-active", true);
  }
}
reviewToggle.addEventListener("click", () => {
  reviewOn = !reviewOn;
  remember("editor-review", reviewOn);
  renderReview(view.state.field(reviewField));
});

function showVim(): void {
  vimToggle.setAttribute("aria-pressed", String(vimOn));
  view.dispatch({ effects: keys.reconfigure(vimOn ? vim() : []) });
}
function showPreview(): void {
  previewToggle.setAttribute("aria-pressed", String(previewOn));
  panes.toggleAttribute("data-preview", previewOn);
  preview.hidden = !previewOn;
  if (previewOn && current !== undefined && preview.getAttribute("src") !== current.url) {
    preview.src = current.url;
  }
}
vimToggle.addEventListener("click", () => {
  vimOn = !vimOn;
  remember("editor-vim", vimOn);
  showVim();
  view.focus();
});
previewToggle.addEventListener("click", () => {
  previewOn = !previewOn;
  remember("editor-preview", previewOn);
  showPreview();
});

// The sidebar: one row per post folder, its title under the folder name, and a tag on drafts.
function renderList(): void {
  tree.replaceChildren(
    ...posts.map(post => {
      const row = element(
        "a",
        { className: "tree-item", href: `#${post.dir}` },
        element("span", { className: "tree-dir" }, `${post.dir}/`),
        element("span", { className: "tree-title" }, post.title),
      );
      if (post.draft) {
        row.append(element("span", { className: "tag" }, "draft"));
      }
      if (post.dir === current?.dir) {
        row.setAttribute("aria-current", "page");
      }
      row.addEventListener("click", event => {
        event.preventDefault();
        void openPost(post.dir);
      });
      return element("li", {}, row);
    }),
  );
}

let postsOn = remembered("editor-posts", true);
function showPosts(): void {
  postsToggle.setAttribute("aria-pressed", String(postsOn));
  sidebar.hidden = !postsOn;
}
postsToggle.addEventListener("click", () => {
  postsOn = !postsOn;
  remember("editor-posts", postsOn);
  showPosts();
});

async function openPost(dir: string): Promise<void> {
  await save();
  const post = posts.find(candidate => candidate.dir === dir) ?? posts[0];
  if (post === undefined) {
    say("no posts yet: start one with “new post”");
    return;
  }
  await saveSuggestions();
  disk = opened(await readPost(post.dir));
  current = post;
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: disk.text } });
  await loadSuggestions();
  renderList();
  open.href = post.url;
  previewScroll = 0;
  // A hidden preview loads once it's shown.
  if (previewOn) {
    preview.src = post.url;
  }
  history.replaceState(null, "", `#${post.dir}`);
  say("saved");
}

// Back on the tab after editing the post somewhere else (shouldReload in saving.ts).
async function reloadIfClean(): Promise<void> {
  if (current === undefined || disk === undefined) {
    return;
  }
  const latest = await readPost(current.dir);
  if (!shouldReload(disk, view.state.doc.toString(), latest.version)) {
    return;
  }
  disk = opened(latest);
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: disk.text } });
  await loadSuggestions();
  say("reloaded: the file changed elsewhere");
}
addEventListener("focus", () => {
  void reloadIfClean()
    .then(reloadSuggestionsIfNew)
    .catch(() => undefined);
});

// A new post's page exists once the dev server has picked up its folder, a moment after it's written.
async function waitForPage(url: string): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt++) {
    if ((await fetch(url, { method: "HEAD" })).ok) {
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
}

async function refresh(): Promise<void> {
  posts = await listPosts();
  current = posts.find(post => post.dir === current?.dir) ?? current;
  renderList();
}

$<HTMLButtonElement>("#create").addEventListener("click", () => dialog.showModal());
$<HTMLButtonElement>("#cancel").addEventListener("click", () => dialog.close());
form.addEventListener("submit", event => {
  event.preventDefault();
  const data = new FormData(form);
  const body = {
    title: String(data.get("title") ?? ""),
    description: String(data.get("description") ?? ""),
    format: String(data.get("format") ?? "post") as Format,
  };
  void (async () => {
    try {
      const dir = await createPost(body);
      dialog.close();
      form.reset();
      await refresh();
      await waitForPage(posts.find(post => post.dir === dir)?.url ?? "/");
      await openPost(dir);
      view.dispatch({ selection: { anchor: view.state.doc.length } });
      view.focus();
    } catch (error) {
      $<HTMLElement>("#new-error").textContent = (error as Error).message;
    }
  })();
});
// Closing the tab mid-sentence still writes the last words.
addEventListener("pagehide", () => {
  const text = view.state.doc.toString();
  if (current !== undefined && disk !== undefined && saveStep(disk, text) === "write") {
    void fetch(postUrl(current.dir), { ...saveInit(text, disk.version), keepalive: true });
  }
  if (current !== undefined && suggestionsTimer !== undefined && suggestionsVersion !== undefined) {
    void fetch(`${postUrl(current.dir)}/suggestions`, {
      method: "PUT",
      body: JSON.stringify({ suggestions: openSuggestions() }),
      headers: { "If-Match": `"${suggestionsVersion}"` },
      keepalive: true,
    });
  }
});

showPreview();
showPosts();
vimToggle.setAttribute("aria-pressed", String(vimOn));
await refresh();
await openPost(decodeURIComponent(location.hash.slice(1)));
