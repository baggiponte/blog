// A guitar tuner: each string button plays its reference note, and with the microphone on, the pitch is found by
// autocorrelation and shown as cents away from the nearest (or the chosen) string.
const $ = id => document.getElementById(id);
const STRINGS = [
  { note: "E", name: "mi grave", frequency: 82.41 },
  { note: "A", name: "la", frequency: 110.0 },
  { note: "D", name: "re", frequency: 146.83 },
  { note: "G", name: "sol", frequency: 196.0 },
  { note: "B", name: "si", frequency: 246.94 },
  { note: "E", name: "mi cantino", frequency: 329.63 },
];
const NOTES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const IDLE = "tocca una corda per sentire la nota di riferimento";

let auto = true;
let selected = null;
let audio = null;
let analyser = null;
let samples = null;
let listening = false;
let tone = null;
const tuned = new Set();

// The page's palette, read when drawing so the scale and headstock follow the theme toggle.
const color = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---- the needle's scale ----------------------------------------------------
function drawTicks() {
  const svg = $("ticks");
  const width = svg.clientWidth;
  const height = svg.clientHeight;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  let marks = "";
  for (let cents = -50; cents <= 50; cents += 5) {
    const x = width / 2 + (cents / 50) * (width / 2 - 30);
    const big = cents % 25 === 0;
    const stroke = cents === 0 ? "var(--accent)" : "var(--rule)";
    marks += `<line x1="${x}" x2="${x}" y1="${big ? 140 : 148}" y2="162" stroke="${stroke}" stroke-width="${big ? 2 : 1}"/>`;
    if (big && cents !== 0) {
      marks += `<text x="${x}" y="134" fill="var(--muted)" font-size="10" font-family="ui-monospace, monospace" text-anchor="middle">${cents > 0 ? `+${cents}` : cents}</text>`;
    }
  }
  svg.innerHTML = marks;
}

// ---- the headstock, drawn on a canvas ----------------------------------------
function drawHead() {
  const canvas = $("headstock");
  const ratio = devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const g = canvas.getContext("2d");
  g.scale(ratio, ratio);
  const cx = width / 2;
  const top = 6;
  const half = Math.min(width * 0.42, 90);
  g.beginPath();
  g.moveTo(cx, top);
  g.bezierCurveTo(cx + half * 0.6, top, cx + half * 0.7, top + 30, cx + half, top + 40);
  g.lineTo(cx + half * 0.92, height * 0.7);
  g.quadraticCurveTo(cx + half * 0.6, height * 0.82, cx + half * 0.55, height);
  g.lineTo(cx - half * 0.55, height);
  g.quadraticCurveTo(cx - half * 0.6, height * 0.82, cx - half * 0.92, height * 0.7);
  g.lineTo(cx - half, top + 40);
  g.bezierCurveTo(cx - half * 0.7, top + 30, cx - half * 0.6, top, cx, top);
  const wood = g.createLinearGradient(0, 0, 0, height);
  wood.addColorStop(0, "#7a4329");
  wood.addColorStop(1, "#4a2516");
  g.fillStyle = wood;
  g.fill();
  g.strokeStyle = "#e9dcc4";
  g.lineWidth = 3;
  g.stroke();
  // the nut
  g.fillStyle = "#efe6d4";
  g.fillRect(cx - half * 0.55, height - 14, half * 1.1, 6);
  // pegs and strings, three a side
  const pegY = [height * 0.2, height * 0.42, height * 0.64];
  const nutX = i => cx - half * 0.42 + i * ((half * 0.84) / 5);
  const accent = color("--accent");
  for (let row = 0; row < 3; row++) {
    for (const [side, string] of [
      [-1, 2 - row],
      [1, 3 + row],
    ]) {
      const px = cx + side * half * 0.62;
      const py = pegY[row];
      const active = selected === string;
      g.strokeStyle = active ? accent : "rgba(240, 234, 220, 0.8)";
      g.lineWidth = active ? 2.4 : 1 + (5 - string) * 0.25;
      g.beginPath();
      g.moveTo(px, py);
      g.lineTo(nutX(string), height);
      g.stroke();
      g.beginPath();
      g.arc(px, py, 9, 0, Math.PI * 2);
      const peg = g.createRadialGradient(px - 3, py - 3, 1, px, py, 9);
      peg.addColorStop(0, "#fffdf7");
      peg.addColorStop(1, "#8e8874");
      g.fillStyle = peg;
      g.fill();
    }
  }
}

function select(i) {
  selected = i;
  for (const button of document.querySelectorAll(".str")) {
    button.classList.toggle("sel", Number(button.dataset.i) === i);
  }
  drawHead();
}

// ---- reference notes ----------------------------------------------------------
async function ensureAudio() {
  audio ??= new AudioContext();
  await audio.resume();
}
function playTone(frequency) {
  try {
    tone?.stop();
  } catch {
    // Already stopped on its own.
  }
  const t = audio.currentTime;
  const base = audio.createOscillator();
  const octave = audio.createOscillator();
  const gain = audio.createGain();
  const octaveGain = audio.createGain();
  base.type = "triangle";
  base.frequency.value = frequency;
  octave.type = "sine";
  octave.frequency.value = frequency * 2;
  octaveGain.gain.value = 0.3;
  base.connect(gain);
  octave.connect(octaveGain).connect(gain);
  gain.connect(audio.destination);
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(0.35, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 2.6);
  base.start(t);
  octave.start(t);
  base.stop(t + 2.7);
  octave.stop(t + 2.7);
  tone = base;
}

// ---- pitch detection (autocorrelation) -----------------------------------------
function detect(buffer, sampleRate) {
  let rms = 0;
  for (const value of buffer) {
    rms += value * value;
  }
  if (Math.sqrt(rms / buffer.length) < 0.012) {
    return -1;
  }
  // Trim the quiet ends so the correlation starts and stops near zero crossings.
  let start = 0;
  let end = buffer.length - 1;
  const threshold = 0.2;
  for (let i = 0; i < buffer.length / 2; i++) {
    if (Math.abs(buffer[i]) < threshold) {
      start = i;
      break;
    }
  }
  for (let i = 1; i < buffer.length / 2; i++) {
    if (Math.abs(buffer.at(-i)) < threshold) {
      end = buffer.length - i;
      break;
    }
  }
  const s = buffer.slice(start, end);
  const n = s.length;
  const c = new Float32Array(n);
  for (let lag = 0; lag < n; lag++) {
    let sum = 0;
    for (let j = 0; j < n - lag; j++) {
      sum += s[j] * s[j + lag];
    }
    c[lag] = sum;
  }
  let dip = 0;
  while (dip < n - 1 && c[dip] > c[dip + 1]) {
    dip++;
  }
  let max = -1;
  let peak = -1;
  for (let i = dip; i < n; i++) {
    if (c[i] > max) {
      max = c[i];
      peak = i;
    }
  }
  if (peak < 1 || peak >= n - 1) {
    return -1;
  }
  // Parabolic interpolation around the peak for a sub-sample period.
  const [x1, x2, x3] = [c[peak - 1], c[peak], c[peak + 1]];
  const a = (x1 + x3 - 2 * x2) / 2;
  const b = (x3 - x1) / 2;
  const period = a ? peak - b / (2 * a) : peak;
  return sampleRate / period;
}

let smooth = null;
let okSince = 0;
let lastHeard = 0;
const distance = (frequency, string) => Math.abs(Math.log2(frequency / STRINGS[string].frequency));

function listen() {
  if (!listening) {
    return;
  }
  analyser.getFloatTimeDomainData(samples);
  const frequency = detect(samples, audio.sampleRate);
  const now = performance.now();
  if (frequency > 60 && frequency < 1000) {
    lastHeard = now;
    smooth = smooth ? smooth * 0.75 + frequency * 0.25 : frequency;
    let target = selected;
    if (auto || target === null) {
      target = STRINGS.reduce((best, _, i) => (distance(smooth, i) < distance(smooth, best) ? i : best), 0);
      if (target !== selected) {
        select(target);
      }
    }
    show(smooth, 1200 * Math.log2(smooth / STRINGS[target].frequency), target, now);
  } else if (now - lastHeard > 1200) {
    smooth = null;
    okSince = 0;
    idle();
  }
  requestAnimationFrame(listen);
}

function noteName(frequency) {
  const midi = Math.round(12 * Math.log2(frequency / 440) + 69);
  return NOTES[midi % 12] + (Math.floor(midi / 12) - 1);
}
function advice(cents) {
  if (Math.abs(cents) > 45) {
    return cents < 0 ? "troppo bassa: stringi di più la corda" : "troppo alta: allenta di più la corda";
  }
  return cents < 0 ? "un po' bassa: stringi la corda" : "un po' alta: allenta la corda";
}
function show(frequency, cents, string, now) {
  const clamped = Math.max(-50, Math.min(50, cents));
  const needle = $("needle");
  needle.style.transform = `translateX(${(clamped / 50) * ($("ticks").clientWidth / 2 - 30)}px)`;
  needle.textContent = STRINGS[string].note;
  const ok = Math.abs(cents) < 5;
  needle.className = ok ? "needle ok" : "needle off";
  $("readout").textContent =
    `${frequency.toFixed(1)} Hz · ${noteName(frequency)} · ${cents > 0 ? "+" : ""}${cents.toFixed(0)} cent`;
  const status = $("status");
  if (ok) {
    okSince ||= now;
    status.textContent = "accordata!";
    status.className = "status ok";
    if (now - okSince > 700 && !tuned.has(string)) {
      tuned.add(string);
      document.querySelector(`.str[data-i="${string}"]`).classList.add("done");
    }
  } else {
    okSince = 0;
    status.className = "status";
    status.textContent = advice(cents);
  }
}
function idle() {
  const needle = $("needle");
  needle.style.transform = "translateX(0)";
  needle.className = "needle";
  needle.textContent = selected === null ? "–" : STRINGS[selected].note;
  $("readout").innerHTML = "&nbsp;";
  $("status").className = "status";
  $("status").textContent = listening ? "suona una corda qualsiasi" : IDLE;
}

async function startMic() {
  try {
    await ensureAudio();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    analyser = audio.createAnalyser();
    analyser.fftSize = 4096;
    samples = new Float32Array(analyser.fftSize);
    audio.createMediaStreamSource(stream).connect(analyser);
    listening = true;
    $("mic").hidden = true;
    idle();
    listen();
  } catch {
    $("status").textContent = "il microfono non è disponibile qui. tocca le corde e accorda a orecchio.";
    $("mic").hidden = true;
  }
}

async function pick(i) {
  await ensureAudio();
  if (!listening) {
    select(i);
    playTone(STRINGS[i].frequency);
    $("needle").textContent = STRINGS[i].note;
    $("readout").textContent = `${STRINGS[i].frequency.toFixed(2)} Hz · ${STRINGS[i].name}`;
    $("status").className = "status";
    $("status").textContent =
      `sto suonando ${STRINGS[i].name} (${STRINGS[i].note}). regola la corda finché i suoni coincidono.`;
  } else {
    // With the microphone on, picking a string locks the tuner to it.
    auto = false;
    $("auto").setAttribute("aria-pressed", "false");
    select(i);
  }
}

for (const button of document.querySelectorAll(".str")) {
  button.addEventListener("click", () => {
    pick(Number(button.dataset.i)).catch(error => console.error(error));
  });
}
$("auto").addEventListener("click", () => {
  auto = !auto;
  $("auto").setAttribute("aria-pressed", String(auto));
});
$("mic").addEventListener("click", () => {
  startMic().catch(error => console.error(error));
});
if (!navigator.mediaDevices?.getUserMedia) {
  $("mic").hidden = true;
}

addEventListener("resize", () => {
  drawTicks();
  drawHead();
});
addEventListener("themechange", drawHead);
drawTicks();
drawHead();
