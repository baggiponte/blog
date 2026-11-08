// A story editor: the preview is the same 1080×1920 canvas the png is exported from, so what you see is what you get.
const $ = id => document.getElementById(id);
const canvas = $("story");
const ctx = canvas.getContext("2d");
const presets = [
  { name: "Tramonto", top: "#ff1658", bottom: "#ff8508", glow: true },
  { name: "After hours", top: "#7135d6", bottom: "#ed56a0", glow: false },
  { name: "Oceano", top: "#133fa0", bottom: "#3acdd0", glow: false },
  { name: "Pesca", top: "#ed6f86", bottom: "#ffc88c", glow: false },
  { name: "Aurora", top: "#123c46", bottom: "#65b58c", glow: false },
  { name: "Notte", top: "#242144", bottom: "#8463a0", glow: false },
];
const defaults = {
  ...presets[0],
  title: "Preferiresti:",
  options: [
    "Uscire con un GymBro che al primo appuntamento porta la bilancia e pesa ogni proteina nella sua caesar salad guardandoti giudizioso mentre mangi le tue patatine fritte con un velato senso di colpevolezza",
    "Lasciare questa terra decapitato da un cartello stradale piombato sulla tua testa mentre attraversi la strada",
  ],
  size: 68,
};
const fields = ["top", "bottom", "glow", "title", "size"];
const freshState = () => ({ ...defaults, options: [...defaults.options] });
let state = freshState();
let fontFamily = "Story";
let customFace = null;
let ready = false;
let saveTimer;
try {
  localStorage.removeItem("preferiresti.v1");
  const saved = JSON.parse(localStorage.getItem("preferiresti.v2"));
  if (saved && typeof saved === "object") {
    for (const k of fields) {
      if (typeof saved[k] === typeof defaults[k]) {
        state[k] = saved[k];
      }
    }
    if (
      Array.isArray(saved.options) &&
      saved.options.length >= 2 &&
      saved.options.every(text => typeof text === "string")
    ) {
      state.options = saved.options.map(text => text.slice(0, 2500));
    }
  }
} catch {}
for (const k of ["top", "bottom"]) {
  if (!/^#[\da-f]{6}$/i.test(state[k])) {
    state[k] = defaults[k];
  }
}
state.size = Math.min(84, Math.max(30, Number(state.size) || 68));
state.title = state.title.slice(0, 160);
function buildOptions() {
  $("options").replaceChildren();
  state.options.forEach((text, index) => {
    const field = document.createElement("label");
    field.className = "field";
    const heading = document.createElement("span");
    heading.className = "option-heading";
    const caption = document.createElement("span");
    caption.textContent = `Opzione ${index + 1}`;
    heading.append(caption);
    if (state.options.length > 2) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-option";
      remove.textContent = "Rimuovi";
      remove.setAttribute("aria-label", `Rimuovi opzione ${index + 1}`);
      remove.addEventListener("click", event => {
        event.preventDefault();
        state.options.splice(index, 1);
        buildOptions();
        render();
        save();
      });
      heading.append(remove);
    }
    const input = document.createElement("textarea");
    input.rows = 3;
    input.maxLength = 2500;
    input.value = text;
    input.setAttribute("aria-label", `Opzione ${index + 1}`);
    input.addEventListener("input", () => {
      state.options[index] = input.value;
      render();
      save();
    });
    field.append(heading, input);
    $("options").append(field);
  });
}
function sync() {
  for (const key of fields) {
    if (key === "glow") {
      $(key).checked = state[key];
    } else {
      $(key).value = state[key];
    }
  }
  buildOptions();
  render();
}
function updateControls() {
  $("topHex").textContent = state.top.toUpperCase();
  $("bottomHex").textContent = state.bottom.toUpperCase();
  $("sizeValue").textContent = `${state.size} px`;
  for (const [i, el] of document.querySelectorAll(".preset").entries()) {
    const preset = presets[i];
    el.setAttribute(
      "aria-pressed",
      String(preset.top === state.top && preset.bottom === state.bottom && preset.glow === state.glow),
    );
  }
}
function wrap(text, width) {
  const result = [];
  for (const paragraph of text.split("\n")) {
    if (!paragraph.trim()) {
      result.push("");
      continue;
    }
    let line = "";
    for (const word of paragraph.trim().split(/\s+/u)) {
      if (ctx.measureText(line ? `${line} ${word}` : word).width <= width) {
        line = line ? `${line} ${word}` : word;
        continue;
      }
      if (line) {
        result.push(line);
        line = "";
      }
      for (const char of word) {
        if (line && ctx.measureText(line + char).width > width) {
          result.push(line);
          line = "";
        }
        line += char;
      }
    }
    result.push(line);
  }
  return result;
}
function optionText(text) {
  return text.trim() ? `- ${text.trim().replace(/^-\s*/, "")}` : "";
}
function storyTexts() {
  return [state.title, ...state.options.map(optionText)];
}
function layout(size) {
  ctx.font = `400 ${size}px "${fontFamily}"`;
  const texts = storyTexts();
  const lineHeight = size * 1.22;
  const blocks = texts.map(text => (text.trim() ? wrap(text, 816) : []));
  const gap = size * 1.35;
  const visible = blocks.filter(lines => lines.length);
  return {
    blocks,
    lineHeight,
    gap,
    total: visible.reduce((sum, lines) => sum + lines.length * lineHeight, 0) + Math.max(0, visible.length - 1) * gap,
  };
}
function render() {
  updateControls();
  const gradient = ctx.createLinearGradient(0, 0, 0, 1920);
  gradient.addColorStop(0, state.top);
  gradient.addColorStop(1, state.bottom);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 1080, 1920);
  if (state.glow) {
    const light = ctx.createRadialGradient(1190, 900, 0, 1190, 900, 1170);
    light.addColorStop(0, "rgba(255,222,55,.98)");
    light.addColorStop(0.38, "rgba(255,174,0,.65)");
    light.addColorStop(1, "rgba(255,150,0,0)");
    ctx.fillStyle = light;
    ctx.fillRect(0, 0, 1080, 1920);
    const red = ctx.createRadialGradient(-80, 1730, 0, -80, 1730, 850);
    red.addColorStop(0, "rgba(222,35,84,.7)");
    red.addColorStop(1, "rgba(222,35,84,0)");
    ctx.fillStyle = red;
    ctx.fillRect(0, 0, 1080, 1920);
  }
  if (!ready) {
    return;
  }
  let size = Number(state.size);
  let plan = layout(size);
  while (plan.total > 1390 && size > 4) {
    size -= 0.5;
    plan = layout(size);
  }
  ctx.font = `400 ${size}px "${fontFamily}"`;
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.shadowColor = "rgba(65,25,20,.23)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetY = 3;
  let y = Math.max(220, (1920 - plan.total) / 2 - 70);
  let drawn = false;
  for (const lines of plan.blocks) {
    if (lines.length === 0) {
      continue;
    }
    if (drawn) {
      y += plan.gap;
    }
    for (const line of lines) {
      ctx.fillText(line, 540, y + plan.lineHeight / 2);
      y += plan.lineHeight;
    }
    drawn = true;
  }
  ctx.shadowColor = "transparent";
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;
  canvas.setAttribute("aria-label", storyTexts().filter(Boolean).join(" "));
  $("sizeValue").textContent = `${String(state.size)} px${size < Number(state.size) ? ` → ${size} px` : ""}`;
}
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem("preferiresti.v2", JSON.stringify(state));
    } catch {
      $("status").textContent = "Salvataggio automatico non disponibile in questo browser.";
    }
  }, 200);
}
presets.forEach(preset => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "preset";
  button.title = preset.name;
  button.setAttribute("aria-label", preset.name);
  button.style.background = `linear-gradient(155deg,${preset.top},${preset.bottom})`;
  button.addEventListener("click", () => {
    Object.assign(state, { top: preset.top, bottom: preset.bottom, glow: preset.glow });
    sync();
    save();
  });
  $("presets").append(button);
});
// What each field holds: glow is a checkbox, size a number, the rest text.
function fieldValue(key) {
  if (key === "glow") {
    return $(key).checked;
  }
  return key === "size" ? Number($(key).value) : $(key).value;
}
for (const key of fields) {
  $(key).addEventListener("input", () => {
    state[key] = fieldValue(key);
    render();
    save();
  });
}
$("addOption").addEventListener("click", () => {
  state.options.push("");
  buildOptions();
  $("options").lastElementChild.querySelector("textarea").focus();
  render();
  save();
});
$("swap").addEventListener("click", () => {
  [state.top, state.bottom] = [state.bottom, state.top];
  sync();
  save();
});
$("reset").addEventListener("click", () => {
  state = freshState();
  sync();
  save();
  $("status").textContent = "Esempio ripristinato.";
});
$("font").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  try {
    const face = await new FontFace(`CustomStory${Date.now()}`, await file.arrayBuffer()).load();
    if (customFace) {
      document.fonts.delete(customFace);
    }
    customFace = face;
    document.fonts.add(face);
    fontFamily = face.family;
    $("fontName").textContent = `${file.name} · ricaricalo alla prossima apertura`;
    render();
    $("status").textContent = "Font caricato.";
  } catch {
    $("status").textContent = "Impossibile leggere questo font. Prova un file .ttf, .otf o .woff.";
  }
  event.target.value = "";
});
$("resetFont").addEventListener("click", () => {
  fontFamily = "Story";
  if (customFace) {
    document.fonts.delete(customFace);
    customFace = null;
  }
  $("fontName").textContent = "Barlow Condensed · incluso";
  render();
});
$("download").addEventListener("click", () => {
  render();
  canvas.toBlob(blob => {
    if (!blob) {
      $("status").textContent = "Esportazione non riuscita. Riprova.";
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `preferiresti-${new Date().toISOString().slice(0, 10)}.png`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    $("status").textContent = "PNG pronto · 1080 × 1920 px.";
  }, "image/png");
});
sync();
document.fonts
  .load("62px Story")
  .then(() => {
    ready = true;
    render();
    $("download").disabled = false;
  })
  .catch(() => {
    $("status").textContent = "Il font incluso non è disponibile. Carica un font per continuare.";
  });
