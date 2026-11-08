// A metronome scheduled on the Web Audio clock: clicks are queued 100ms ahead so timers can't make them drift, and
// the pendulum and beat dots follow the queue on animation frames.
const $ = id => document.getElementById(id);
const MARKS = [
  [40, "grave"],
  [60, "largo"],
  [66, "larghetto"],
  [76, "adagio"],
  [108, "andante"],
  [120, "moderato"],
  [156, "allegro"],
  [176, "vivace"],
  [200, "presto"],
  [Number.POSITIVE_INFINITY, "prestissimo"],
];
const SUBDIVISIONS = [
  [1, "♩ semiminime"],
  [2, "♫ crome"],
  [3, "terzine"],
  [4, "semicrome"],
];
const MIN = 30;
const MAX = 250;

let bpm = 96;
let beats = 4;
let subdivision = 0;
let accent = true;
try {
  const saved = JSON.parse(localStorage.getItem("metro") || "{}");
  if (Number.isFinite(saved.bpm)) {
    bpm = saved.bpm;
  }
  if ([2, 3, 4, 6].includes(saved.beats)) {
    beats = saved.beats;
  }
  if (saved.subIdx in SUBDIVISIONS) {
    subdivision = saved.subIdx;
  }
  if (typeof saved.accent === "boolean") {
    accent = saved.accent;
  }
} catch {
  // No storage or a broken save: start from the defaults.
}
function save() {
  try {
    localStorage.setItem("metro", JSON.stringify({ bpm, beats, subIdx: subdivision, accent }));
  } catch {
    // Private windows: settings last until the page is left.
  }
}

let audio = null;
let playing = false;
let nextTime = 0;
let tick = 0;
let timer = 0;
let wake = null;
const queue = [];

function setBpm(value) {
  bpm = Math.max(MIN, Math.min(MAX, Math.round(value)));
  $("bpm").textContent = String(bpm);
  $("slider").value = String(bpm);
  $("slider").style.setProperty("--p", `${((bpm - MIN) / (MAX - MIN)) * 100}%`);
  $("mark").textContent = MARKS.find(([below]) => bpm < below)[1];
  save();
}
function renderBeats() {
  const dots = Array.from({ length: beats }, (_, i) => {
    const dot = document.createElement("div");
    dot.className = i === 0 && accent ? "dot first" : "dot";
    return dot;
  });
  $("beats").replaceChildren(...dots);
  for (const button of document.querySelectorAll("[data-beats]")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.beats) === beats));
  }
}
function renderOptions() {
  $("sub-label").textContent = SUBDIVISIONS[subdivision][1];
  $("accent-label").textContent = accent ? "attivo" : "spento";
}

// kind 0 is the accented downbeat, 1 any other beat, 2 a subdivision between beats.
function click(time, kind) {
  const oscillator = audio.createOscillator();
  const gain = audio.createGain();
  oscillator.frequency.value = [1760, 1100, 800][kind];
  gain.gain.setValueAtTime(kind === 2 ? 0.25 : 0.7, time);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.05);
  oscillator.connect(gain).connect(audio.destination);
  oscillator.start(time);
  oscillator.stop(time + 0.06);
}
function schedule() {
  const per = SUBDIVISIONS[subdivision][0];
  const step = 60 / bpm / per;
  while (nextTime < audio.currentTime + 0.1) {
    const beat = Math.floor(tick / per) % beats;
    const onBeat = tick % per === 0;
    let kind = 2;
    if (onBeat) {
      kind = beat === 0 && accent ? 0 : 1;
    }
    click(nextTime, kind);
    if (onBeat) {
      queue.push({ time: nextTime, beat, count: Math.floor(tick / per) });
    }
    nextTime += step;
    tick++;
  }
}
function draw() {
  if (!playing) {
    return;
  }
  while (queue.length > 0 && queue[0].time <= audio.currentTime) {
    const { beat, count } = queue.shift();
    for (const [i, dot] of Array.from($("beats").children).entries()) {
      dot.classList.toggle("on", i === beat);
    }
    const arm = $("arm");
    const side = count % 2 ? -1 : 1;
    arm.style.transition = `transform ${60 / bpm}s ease-in-out`;
    arm.style.transform = `rotate(${side * -28}deg)`;
    arm.classList.add("flash");
    setTimeout(() => arm.classList.remove("flash"), 90);
  }
  requestAnimationFrame(draw);
}

const playIcon = '<path d="M7 4.5v15l13-7.5z"></path>';
const stopIcon =
  '<rect x="6" y="5" width="4.5" height="14" rx="1"></rect><rect x="13.5" y="5" width="4.5" height="14" rx="1"></rect>';

async function start() {
  audio ??= new AudioContext();
  await audio.resume();
  playing = true;
  tick = 0;
  queue.length = 0;
  nextTime = audio.currentTime + 0.06;
  $("arm").style.transition = "none";
  $("arm").style.transform = "rotate(28deg)";
  timer = setInterval(schedule, 25);
  schedule();
  requestAnimationFrame(draw);
  $("play-label").textContent = "stop";
  $("icon").innerHTML = stopIcon;
  try {
    wake = await navigator.wakeLock.request("screen");
  } catch {
    // No Wake Lock (or it was refused): the screen may dim while it plays.
  }
}
function stop() {
  playing = false;
  clearInterval(timer);
  for (const dot of $("beats").children) {
    dot.classList.remove("on");
  }
  const arm = $("arm");
  arm.style.transition = "transform .4s ease-out";
  arm.style.transform = "rotate(0deg)";
  $("play-label").textContent = "avvia";
  $("icon").innerHTML = playIcon;
  wake?.release().catch(() => undefined);
  wake = null;
}
function toggle() {
  if (playing) {
    stop();
  } else {
    start().catch(error => console.error(error));
  }
}
function restartIfPlaying() {
  if (playing) {
    stop();
    toggle();
  }
}

$("play").addEventListener("click", toggle);
$("slider").addEventListener("input", event => setBpm(Number(event.target.value)));

// Holding − or + keeps stepping after a short pause.
let hold = 0;
for (const [id, delta] of [
  ["minus", -1],
  ["plus", 1],
]) {
  const button = $(id);
  button.addEventListener("pointerdown", () => {
    setBpm(bpm + delta);
    const repeat = () => {
      setBpm(bpm + delta);
      hold = setTimeout(repeat, 60);
    };
    hold = setTimeout(repeat, 400);
  });
  for (const type of ["pointerup", "pointerleave", "pointercancel"]) {
    button.addEventListener(type, () => clearTimeout(hold));
  }
}
for (const button of document.querySelectorAll("[data-beats]")) {
  button.addEventListener("click", () => {
    beats = Number(button.dataset.beats);
    renderBeats();
    save();
    restartIfPlaying();
  });
}
$("sub").addEventListener("click", () => {
  subdivision = (subdivision + 1) % SUBDIVISIONS.length;
  renderOptions();
  save();
});
$("accent").addEventListener("click", () => {
  accent = !accent;
  renderOptions();
  renderBeats();
  save();
});

// Tap tempo: the average gap of the last taps, forgotten after two seconds without one.
let taps = [];
$("tap").addEventListener("click", () => {
  const now = performance.now();
  if (taps.length > 0 && now - taps.at(-1) > 2000) {
    taps = [];
  }
  taps.push(now);
  if (taps.length > 6) {
    taps.shift();
  }
  if (taps.length >= 2) {
    setBpm(60000 / ((taps.at(-1) - taps[0]) / (taps.length - 1)));
  }
});
document.addEventListener("keydown", event => {
  if (event.target instanceof HTMLInputElement || event.metaKey || event.ctrlKey) {
    return;
  }
  if (event.code === "Space" && !(event.target instanceof HTMLButtonElement)) {
    event.preventDefault();
    toggle();
  } else if (event.key === "ArrowUp" || event.key === "ArrowRight") {
    setBpm(bpm + 1);
  } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
    setBpm(bpm - 1);
  }
});

setBpm(bpm);
renderBeats();
renderOptions();
