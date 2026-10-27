// The talks page's views: switching between them and lighting up talks and cities. The theme toggle is in site.ts.
import { find, targetOf } from "./dom.ts";

// The map is the default view (also "#map"); "#talks" and "#timeline" switch to the others. They match no element id, so the page does not scroll.
const page = find(".talks");
const views = ["map", "talks", "timeline"] as const;
type View = (typeof views)[number];
const isView = (name: string): name is View => (views as readonly string[]).includes(name);
const route = (): void => {
  const name = location.hash.slice(1);
  page.dataset.view = isView(name) ? name : "map";
  for (const view of document.querySelectorAll<HTMLElement>(".views a"))
    view.toggleAttribute("aria-current", view.dataset.view === page.dataset.view);
};
window.addEventListener("hashchange", route);
route();

// A mark is anything with data-talk or data-city; show(mark) lights it up, or resets the view for null.
type Show = (mark: HTMLElement | null) => void;

// Hovering or focusing a mark previews it, a click or tap keeps it, Escape clears it.
// linger (ms) delays the fall back when the pointer moves onto bare space inside the section.
const highlight = (section: HTMLElement, show: Show, { linger = 0 } = {}) => {
  const markOf = (event: Event) => targetOf(event)?.closest<HTMLElement>("[data-talk], [data-city]") ?? null;
  const key = (mark: HTMLElement | null) => mark && (mark.dataset.talk ?? `city ${mark.dataset.city}`);
  let pinned: HTMLElement | null = null;
  let fallback: ReturnType<typeof setTimeout> | undefined;
  const preview = (event: Event) => show(markOf(event) ?? pinned);
  const restore = () => {
    clearTimeout(fallback);
    show(pinned);
  };
  section.addEventListener("pointerover", event => {
    clearTimeout(fallback);
    const mark = markOf(event);
    if (mark || !linger) show(mark ?? pinned);
    else fallback = setTimeout(() => show(pinned), linger);
  });
  section.addEventListener("pointerleave", restore);
  section.addEventListener("focusin", preview);
  section.addEventListener("focusout", restore);
  section.addEventListener("click", event => {
    // Links navigate, and a city's card stays open while you use it.
    if (targetOf(event)?.closest("a, .card")) return;
    const mark = markOf(event);
    pinned = mark && key(mark) !== key(pinned) ? mark : null;
    show(pinned);
  });
  window.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    pinned = null;
    show(null);
  });
};
// Puts "<title>, given N times<suffix>" in the section's caption, or its hint back for null.
const captioner = (section: HTMLElement) => {
  const caption = find(".caption", section);
  const hint = caption.textContent;
  return (talk: string | null, suffix = "") => {
    if (talk === null) {
      caption.textContent = hint;
      return;
    }
    const rows = `li[data-talk="${CSS.escape(talk)}"]`;
    const count = section.querySelectorAll(rows).length;
    const title = document.createElement("strong");
    title.textContent = find(`${rows} .talk`, section).textContent;
    caption.replaceChildren(title, `, given ${["once", "twice"][count - 1] ?? `${count} times`}${suffix}`);
  };
};

// "Milan", "Milan and Turin", "Milan, Turin and Rome".
const listOf = (names: string[]): string =>
  names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : (names[0] ?? "");

// Timeline: a talk lights up every dot and row of its appearances.
const timeline = find(".timeline", page);
const timelineCaption = captioner(timeline);
highlight(timeline, mark => {
  const talk = mark?.dataset.talk ?? null;
  timeline.classList.toggle("active", talk !== null);
  for (const each of timeline.querySelectorAll<HTMLElement>("[data-talk]"))
    each.classList.toggle("on", each.dataset.talk === talk);
  timelineCaption(talk);
});

// Map: a city shows the talks given there; a talk lights up every city it went to.
// Rows carry data-talk; city circles and their cards carry only data-city.
// Crossing the bare map between a circle and its card lingers, so the card stays reachable.
const map = find(".map", page);
const mapCaption = captioner(map);
highlight(
  map,
  mark => {
    const talk = mark?.dataset.talk ?? null;
    const city = talk === null ? (mark?.dataset.city ?? null) : null;
    map.classList.toggle("active", mark !== null);
    for (const row of map.querySelectorAll<HTMLElement>("li[data-talk]"))
      row.classList.toggle("on", row.dataset.talk === talk || row.dataset.city === city);
    for (const place of map.querySelectorAll<HTMLElement>(".place"))
      place.classList.toggle(
        "on",
        place.dataset.city === city || (talk !== null && (place.dataset.talks ?? "").split(" ").includes(talk)),
      );
    for (const card of map.querySelectorAll<HTMLElement>(".card"))
      card.classList.toggle("on", card.dataset.city === city);
    const places = [...map.querySelectorAll<HTMLElement>(".place.on")].map(place => place.dataset.city ?? "");
    mapCaption(talk, ` in ${listOf(places)}`);
  },
  { linger: 300 },
);
