import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { yamlFrontmatter } from "@codemirror/lang-yaml";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { languages } from "@codemirror/language-data";
import { Compartment, Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { tags } from "@lezer/highlight";
import { vim } from "@replit/codemirror-vim";
import { basicSetup } from "codemirror";

import { find as $, element } from "../scripts/dom.ts";
import { apiBase } from "./base.ts";
import { formatKeys } from "./format.ts";
import { imageInsert } from "./insert.ts";
import { livePreview, postImageUrl, revealOnClick } from "./live-preview.ts";
import type { Format, PostSummary } from "./post.ts";
import { ApiError, addImage, createPost, listPosts, postUrl, readPost, saveInit, writePost } from "./remote.ts";
import type { Version } from "./store.ts";

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

const saveDelay = 600;
let current: PostSummary | undefined;
let posts: PostSummary[] = [];
let timer: ReturnType<typeof setTimeout> | undefined;
// The text and version last read or saved: a save sends the version back, and the API refuses it if the file changed
// elsewhere since (another tab, another editor).
let saved = "";
let version = "" as Version;
// Set when a save was refused for that reason; autosave stops until the post is opened again.
let conflicted = false;
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
  if (current === undefined || text === saved || conflicted) {
    return;
  }
  say("saving…", "busy");
  try {
    rememberScroll();
    version = await writePost(current.dir, text, version);
    saved = text;
    say("saved");
    // A new title or draft flag shows in the sidebar.
    await refresh();
  } catch (error) {
    if (error instanceof ApiError && error.conflict) {
      conflicted = true;
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
  const { markdown: text, version: read } = await readPost(post.dir);
  current = post;
  saved = text;
  version = read;
  conflicted = false;
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
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

// Back on the tab after editing the post somewhere else: with no unsaved typing here, take the file as it is now on
// disk, like Zed and VS Code reload a clean buffer. With unsaved typing, the next save reports the conflict instead.
async function reloadIfClean(): Promise<void> {
  if (current === undefined || view.state.doc.toString() !== saved) {
    return;
  }
  const latest = await readPost(current.dir);
  if (latest.version === version || view.state.doc.toString() !== saved) {
    return;
  }
  saved = latest.markdown;
  version = latest.version;
  conflicted = false;
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: latest.markdown } });
  say("reloaded: the file changed elsewhere");
}
addEventListener("focus", () => void reloadIfClean().catch(() => undefined));

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
  if (current !== undefined && !conflicted && view.state.doc.toString() !== saved) {
    void fetch(postUrl(current.dir), { ...saveInit(view.state.doc.toString(), version), keepalive: true });
  }
});

showPreview();
showPosts();
vimToggle.setAttribute("aria-pressed", String(vimOn));
await refresh();
await openPost(decodeURIComponent(location.hash.slice(1)));
