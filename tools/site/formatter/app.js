// LinkedIn's composer only takes plain text, so the formatting is carried as Unicode: sans-serif bold and italic
// letters, and combining marks for underline and strikethrough.

/* ---------------------------------------------------------------- tables */

const _SANS = { u: 0x1d5a0, l: 0x1d5ba, d: 0x1d7e2 };
const BOLD = { u: 0x1d5d4, l: 0x1d5ee, d: 0x1d7ec }; // Mathematical Sans-Serif Bold
const ITALIC = { u: 0x1d608, l: 0x1d622, d: null }; // Mathematical Sans-Serif Italic
const BOLD_ITALIC = { u: 0x1d63c, l: 0x1d656, d: null }; // Mathematical Sans-Serif Bold Italic

const PLAIN = Object.freeze({ bold: false, italic: false, underline: false, strike: false });

const COMBINING_UNDERLINE = "\u0332";
const COMBINING_STRIKE = "\u0336";

const BLOCK_TAGS = new Set([
  "DIV",
  "P",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "BLOCKQUOTE",
  "PRE",
  "SECTION",
  "ARTICLE",
  "HEADER",
  "FOOTER",
  "FIGURE",
  "FIGCAPTION",
  "ADDRESS",
  "TABLE",
  "HR",
]);

/* ------------------------------------------------------------- styling */

function stylize(text, st) {
  let table = st.italic ? ITALIC : null;
  if (st.bold) {
    table = st.italic ? BOLD_ITALIC : BOLD;
  }
  const marks = (st.underline ? COMBINING_UNDERLINE : "") + (st.strike ? COMBINING_STRIKE : "");
  let out = "";

  for (const ch of text) {
    const cp = ch.codePointAt(0);
    let c = ch;

    if (cp === 0xa0) {
      c = " ";
    } else if (table) {
      if (cp >= 0x41 && cp <= 0x5a) {
        c = String.fromCodePoint(table.u + cp - 0x41);
      } else if (cp >= 0x61 && cp <= 0x7a) {
        c = String.fromCodePoint(table.l + cp - 0x61);
      } else if (cp >= 0x30 && cp <= 0x39 && table.d) {
        c = String.fromCodePoint(table.d + cp - 0x30);
      }
    }

    if (marks && !/\s/.test(ch)) {
      c += marks;
    }
    out += c;
  }

  return out;
}

function styleOf(el, inherited) {
  const tag = el.tagName;
  let s = inherited;

  if (tag === "B" || tag === "STRONG") {
    s = { ...s, bold: true };
  } else if (tag === "I" || tag === "EM") {
    s = { ...s, italic: true };
  } else if (tag === "U") {
    s = { ...s, underline: true };
  } else if (tag === "S" || tag === "STRIKE" || tag === "DEL") {
    s = { ...s, strike: true };
  }

  const inline = el.style;
  if (inline) {
    const weight = inline.fontWeight;
    if (weight === "bold" || weight === "bolder" || Number.parseInt(weight, 10) >= 600) {
      s = { ...s, bold: true };
    }
    if (inline.fontStyle === "italic") {
      s = { ...s, italic: true };
    }

    const decoration = (inline.textDecorationLine || inline.textDecoration || "").toLowerCase();
    if (decoration.includes("underline")) {
      s = { ...s, underline: true };
    }
    if (decoration.includes("line-through")) {
      s = { ...s, strike: true };
    }
  }

  return s;
}

/* ----------------------------------------------------------- serialize */

// Turns the editor DOM (or a selected fragment of it) into the plain text a
// LinkedIn post can carry: line structure preserved, styling encoded as
// Unicode sans-serif alphanumerics and combining marks.
function serialize(root) {
  const lines = [{ runs: [] }];
  let line = lines[0];
  let pendingBreak = false;

  function newLine() {
    line = { runs: [] };
    lines.push(line);
    pendingBreak = false;
  }

  function addRun(text, style) {
    if (!text) {
      return;
    }
    if (pendingBreak) {
      pendingBreak = false;
      if (line.runs.length > 0) {
        newLine();
      }
    }
    line.runs.push({ text, style });
  }

  function beginBlock() {
    pendingBreak = false;
    if (line.runs.length > 0) {
      newLine();
    }
  }

  function walkAll(nodes, style, depth) {
    for (const node of nodes) {
      walk(node, style, depth);
    }
  }

  function walk(node, style, depth) {
    if (node.nodeType === 3) {
      addRun(node.data, style);
      return;
    }
    if (node.nodeType === 11) {
      walkAll(node.childNodes, style, depth);
      return;
    }
    if (node.nodeType !== 1) {
      return;
    }

    const tag = node.tagName;

    if (tag === "BR") {
      newLine();
      return;
    }

    if (tag === "UL" || tag === "OL") {
      beginBlock();
      const ordered = tag === "OL";
      let n = ordered ? Number.parseInt(node.getAttribute("start") || "1", 10) || 1 : 1;

      for (const item of node.children) {
        if (!item.tagName || item.tagName !== "LI") {
          walk(item, style, depth);
          continue;
        }
        beginBlock();
        const indent = "  ".repeat(depth);
        const marker = ordered ? `${n++}. ` : "• ";
        addRun(indent + marker, PLAIN);
        walkAll(item.childNodes, style, depth + 1);
      }

      pendingBreak = true;
      return;
    }

    if (BLOCK_TAGS.has(tag)) {
      beginBlock();
      walkAll(node.childNodes, style, depth);
      pendingBreak = true;
      return;
    }

    walkAll(node.childNodes, styleOf(node, style), depth);
  }

  walkAll(root.childNodes, PLAIN, 0);

  // Trailing blank lines are editor scaffolding, not post content.
  while (lines.length > 1 && lines.at(-1).runs.every(r => !r.text.trim())) {
    lines.pop();
  }

  return lines
    .map(l => l.runs.map(r => stylize(r.text, r.style)).join(""))
    .join("\n")
    .replace(/[ \t]+$/gm, "");
}

/* ---------------------------------------------------------------- setup */

const editor = document.getElementById("editor");
const countEl = document.getElementById("count");
const copyBtn = document.getElementById("copy");
const toastEl = document.getElementById("toast");
const formatButtons = Array.from(document.querySelectorAll(".fmt button"));

const isMac = /mac|iphone|ipad|ipod/i.test(navigator.platform || navigator.userAgent);
const MOD = isMac ? "⌘" : "Ctrl+";
const LIMIT = 3000;
const DRAFT_KEY = "linkedin-formatter.draft.v1";

const NAMES = {
  bold: "bold",
  italic: "italic",
  underline: "underline",
  strikeThrough: "strikethrough",
  insertUnorderedList: "bulleted list",
  insertOrderedList: "numbered list",
};
const KEYS = {
  bold: "B",
  italic: "I",
  underline: "U",
  strikeThrough: "⇧X",
  insertUnorderedList: "⇧8",
  insertOrderedList: "⇧7",
};

for (const el of document.querySelectorAll("[data-mod]")) {
  el.textContent = isMac ? "⌘" : "Ctrl+";
}

for (const btn of formatButtons) {
  const cmd = btn.dataset.cmd;
  btn.title = `${NAMES[cmd]} (${MOD}${KEYS[cmd]})`;
}

/* ------------------------------------------------------------- preview */

const PREVIEW_KEY = "linkedin-formatter.preview";
const previewBtn = document.getElementById("previewBtn");
const previewBody = document.getElementById("previewBody");
const pageEl = document.querySelector(".fmt-page");

let previewOpen = window.matchMedia("(min-width: 1000px)").matches;
try {
  const saved = localStorage.getItem(PREVIEW_KEY);
  if (saved === "0") {
    previewOpen = false;
  }
  if (saved === "1") {
    previewOpen = true;
  }
} catch (_) {
  /* storage unavailable */
}

function setPreview(open) {
  previewOpen = open;
  previewBtn.setAttribute("aria-pressed", open ? "true" : "false");
  previewBtn.title = open ? "hide the linkedin preview" : "show the linkedin preview";
  pageEl.classList.toggle("has-preview", open);
  try {
    localStorage.setItem(PREVIEW_KEY, open ? "1" : "0");
  } catch (_) {
    /* storage unavailable */
  }
}

previewBtn.addEventListener("click", () => setPreview(!previewOpen));

// The preview shows the exact text that lands on LinkedIn, Unicode included.
function renderPreview(text) {
  if (!text.trim()) {
    previewBody.innerHTML = '<span class="li-empty">your post will appear here.</span>';
    return;
  }
  previewBody.textContent = text;
}

/* --------------------------------------------------------------- state */

let saveTimer = null;
let toastTimer = null;

function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1700);
}

function textLength(text) {
  return Array.from(text).length;
}

function refresh() {
  const text = serialize(editor);
  const n = textLength(text);

  countEl.textContent = `${n} / ${LIMIT}`;
  countEl.classList.toggle("warn", n > LIMIT * 0.9 && n <= LIMIT);
  countEl.classList.toggle("over", n > LIMIT);
  copyBtn.disabled = n === 0;
  syncPressed();
  renderPreview(text);

  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify({ html: editor.innerHTML }));
    } catch (_) {
      /* private mode / quota: the page still works */
    }
  }, 250);
}

function syncPressed() {
  for (const btn of formatButtons) {
    let on = false;
    try {
      on = document.queryCommandState(btn.dataset.cmd);
    } catch (_) {
      /* no selection */
    }
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  }
}

/* ---------------------------------------------------------------- copy */

function currentText() {
  const sel = window.getSelection();
  if (sel?.rangeCount && !sel.isCollapsed) {
    const range = sel.getRangeAt(0);
    if (editor.contains(range.commonAncestorContainer)) {
      return serialize(range.cloneContents());
    }
  }
  return serialize(editor);
}

async function copyPost() {
  const text = currentText();
  if (!text.trim()) {
    toast("nothing to copy yet");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    try {
      document.execCommand("copy");
    } catch (_) {
      /* ignore */
    }
    scratch.remove();
  }

  toast("copied for linkedin");
}

/* ------------------------------------------------------------ commands */

function runCommand(cmd) {
  editor.focus();
  document.execCommand(cmd, false, null);
  refresh();
  syncPressed();
}

for (const btn of formatButtons) {
  btn.addEventListener("mousedown", e => e.preventDefault()); // keep the selection
  btn.addEventListener("click", () => runCommand(btn.dataset.cmd));
}

editor.addEventListener("input", refresh);
editor.addEventListener("keyup", syncPressed);
editor.addEventListener("mouseup", syncPressed);
document.addEventListener("selectionchange", syncPressed);

editor.addEventListener("paste", e => {
  e.preventDefault();
  const clipboard = e.clipboardData || window.clipboardData;
  const text = clipboard ? clipboard.getData("text/plain") : "";
  if (text) {
    document.execCommand("insertText", false, text);
  }
  refresh();
});

// Cmd/Ctrl+C always hands out LinkedIn-ready text, selection or whole post.
document.addEventListener("copy", e => {
  if (document.activeElement !== editor) {
    return;
  }
  const text = currentText();
  if (!text) {
    return;
  }
  e.preventDefault();
  e.clipboardData.setData("text/plain", text);
  toast("copied for linkedin");
});

document.addEventListener("keydown", e => {
  const mod = isMac ? e.metaKey : e.ctrlKey;
  if (!mod || e.altKey) {
    return;
  }

  const key = e.key.toLowerCase();

  if (key === "enter") {
    e.preventDefault();
    copyPost().catch(error => console.error(error));
    return;
  }

  let cmd = null;
  if (!e.shiftKey && key === "b") {
    cmd = "bold";
  } else if (!e.shiftKey && key === "i") {
    cmd = "italic";
  } else if (!e.shiftKey && key === "u") {
    cmd = "underline";
  } else if (e.shiftKey && key === "x") {
    cmd = "strikeThrough";
  } else if (e.shiftKey && e.code === "Digit8") {
    cmd = "insertUnorderedList";
  } else if (e.shiftKey && e.code === "Digit7") {
    cmd = "insertOrderedList";
  }

  if (!cmd) {
    return;
  }
  e.preventDefault();
  runCommand(cmd);
});

copyBtn.addEventListener("mousedown", e => e.preventDefault()); // keep a text selection alive
copyBtn.addEventListener("click", copyPost);

/* --------------------------------------------------------- persistence */

try {
  const raw = localStorage.getItem(DRAFT_KEY);
  if (raw) {
    const draft = JSON.parse(raw);
    if (draft && typeof draft.html === "string") {
      editor.innerHTML = draft.html;
    }
  }
} catch (_) {
  /* corrupt or unavailable storage */
}

setPreview(previewOpen);
refresh();
syncPressed();
editor.focus();
