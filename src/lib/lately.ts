// The homepage's rotating phrases come from the bullet list in content/lately.md.
// Only top-level "- " / "* " bullets count, so the file can carry notes around the list.
const bullet = /^[-*+]\s+(.+)$/;

export function parseLately(markdown: string): [string, ...string[]] {
  const lines = markdown
    .split("\n")
    .map(line => bullet.exec(line.trimEnd())?.[1]?.trim())
    .filter((line): line is string => line !== undefined && line !== "");
  const [first, ...rest] = lines;
  if (first === undefined) {
    throw new Error("content/lately.md has no bullet points to rotate");
  }
  return [first, ...rest];
}
