const newlines = (count: number): string => "\n".repeat(Math.max(0, count));
const leadingNewlines = (text: string): number => /^\n*/.exec(text)?.[0].length ?? 0;
const trailingNewlines = (text: string): number => /\n*$/.exec(text)?.[0].length ?? 0;

// The Markdown for images dropped at `position` in `doc`: one paragraph each, linked relative to index.md so Astro
// picks them up from the post's folder. A blank line separates them from the text around; the file still ends in
// a single newline.
export function imageInsert(doc: string, position: number, names: string[]): { text: string; altFrom: number } {
  const before = doc.slice(0, position);
  const after = doc.slice(position);
  const lead = newlines(before === "" ? 0 : 2 - trailingNewlines(before));
  const trail = newlines((after === "" ? 1 : 2) - leadingNewlines(after));
  const images = names.map(name => `![](./${name})`).join("\n\n");
  // The cursor goes between the first image's brackets, ready for its alt text.
  return { text: `${lead}${images}${trail}`, altFrom: lead.length + 2 };
}
