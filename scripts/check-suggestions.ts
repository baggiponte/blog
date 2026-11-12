import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { parseSuggestions, place } from "../src/editor/suggestion.ts";

// Checks a post's suggestions.json the way the editor will read it: the format, and whether each quote is found once
// in index.md. Claude runs it after writing the file (.claude/skills/proofread/SKILL.md).
//   node scripts/check-suggestions.ts content/blog/<dir>

const dir = process.argv[2];
if (dir === undefined) {
  console.error("usage: node scripts/check-suggestions.ts content/blog/<dir>");
  process.exit(2);
}
const text = await readFile(join(dir, "index.md"), "utf8");
const suggestions = parseSuggestions(JSON.parse(await readFile(join(dir, "suggestions.json"), "utf8")));
let failed = 0;
for (const suggestion of suggestions) {
  const at = place(text, suggestion);
  if (!at.found) {
    failed++;
    console.error(`${suggestion.id}: the quote is ${at.why}: ${JSON.stringify(suggestion.quote)}`);
  }
}
console.log(`${suggestions.length - failed} of ${suggestions.length} suggestions placed`);
process.exit(failed === 0 ? 0 : 1);
