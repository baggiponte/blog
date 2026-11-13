import { z } from "astro/zod";

import { EditorError } from "./failure.ts";

// Proofreading suggestions for a post, with no editor and no files: what a suggestion is, how it's read from
// suggestions.json, and how it finds its place in the text. Claude writes the file next to index.md (see
// .claude/skills/proofread/SKILL.md); the editor shows each suggestion on the text it quotes (review.ts).

// What kind of fix it is, so the editor can tell a typo from a matter of taste.
export const kinds = ["typo", "grammar", "style", "clarity", "fact"] as const;
export type Kind = (typeof kinds)[number];

// A suggestion names the text it changes by quoting it, never by line or offset: a quote still finds its place after
// the text around it moved, and is lost only when the quoted words themselves change. `prefix` and `suffix` are a few
// words around the quote, for when the same words appear more than once. This is how annotation tools that outlive
// edits anchor to text (the W3C TextQuoteSelector, Hypothesis).
const suggestionSchema = z.object({
  id: z.string().min(1),
  kind: z.enum(kinds),
  // Exactly as in index.md, Markdown included.
  quote: z.string().min(1),
  // What replaces the quote; empty to delete it.
  replacement: z.string(),
  // Why, in a sentence.
  note: z.string().optional(),
  prefix: z.string().optional(),
  suffix: z.string().optional(),
});
export type Suggestion = z.infer<typeof suggestionSchema>;

const fileSchema = z.object({
  suggestions: z.array(suggestionSchema).refine(list => new Set(list.map(s => s.id)).size === list.length, {
    message: "two suggestions have the same id",
  }),
});

// The suggestions in a suggestions.json, or an "invalid" failure saying what's wrong with it.
export function parseSuggestions(input: unknown): Suggestion[] {
  const result = fileSchema.safeParse(input);
  if (!result.success) {
    const issues = result.error.issues.map(issue => `${issue.path.join(".") || "file"}: ${issue.message}`);
    throw new EditorError(`suggestions.json: ${issues.join("; ")}`, { failure: "invalid" });
  }
  return result.data.suggestions;
}

export const suggestionsJson = (suggestions: Suggestion[]): string => `${JSON.stringify({ suggestions }, null, 2)}\n`;

// Where a suggestion goes in a text: the quote's range, or why it has none. "missing": the quoted words aren't in the
// text any more. "ambiguous": they appear more than once and the context doesn't tell which.
export type Placement = { found: true; from: number; to: number } | { found: false; why: "missing" | "ambiguous" };

const sharedEnd = (a: string, b: string): number => {
  let n = 0;
  while (n < a.length && n < b.length && a.at(-1 - n) === b.at(-1 - n)) {
    n++;
  }
  return n;
};
const sharedStart = (a: string, b: string): number => {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) {
    n++;
  }
  return n;
};

// Every place the quote appears, scored by how much of the prefix and suffix match around it; the best one wins, and
// a tie is ambiguous rather than a guess.
export function place(text: string, { quote, prefix = "", suffix = "" }: Suggestion): Placement {
  const scored: { from: number; score: number }[] = [];
  for (let at = text.indexOf(quote); at !== -1; at = text.indexOf(quote, at + 1)) {
    const before = text.slice(Math.max(0, at - prefix.length), at);
    const after = text.slice(at + quote.length, at + quote.length + suffix.length);
    scored.push({ from: at, score: sharedEnd(before, prefix) + sharedStart(after, suffix) });
  }
  const [best, second] = scored.toSorted((a, b) => b.score - a.score);
  if (best === undefined) {
    return { found: false, why: "missing" };
  }
  if (second !== undefined && second.score === best.score) {
    return { found: false, why: "ambiguous" };
  }
  return { found: true, from: best.from, to: best.from + quote.length };
}

const contextLength = 32;

// The suggestion with its prefix and suffix taken afresh from the text around its range, so it's found again after
// the editor saves it back and the text around it has changed.
export const withContext = (suggestion: Suggestion, text: string, from: number, to: number): Suggestion => ({
  ...suggestion,
  prefix: text.slice(Math.max(0, from - contextLength), from),
  suffix: text.slice(to, to + contextLength),
});

// The part of the quote that actually changes, in whole words, and what it becomes: "Small tasks, one commit each,
// is the only" -> "are the only" shows as "is" -> "are". The editor strikes only this part; accepting still writes the
// whole replacement over the whole quote.
export interface Change {
  // Offsets into the quote.
  start: number;
  end: number;
  insert: string;
}

const isSpace = (char: string | undefined): boolean => char === undefined || /\s/.test(char);

export function changeOf(quote: string, replacement: string): Change {
  let start = 0;
  while (start < quote.length && start < replacement.length && quote[start] === replacement[start]) {
    start++;
  }
  // Back to the start of the word the difference begins in.
  const splitsWord = (text: string, at: number): boolean => !(isSpace(text[at - 1]) || isSpace(text[at]));
  while (start > 0 && (splitsWord(quote, start) || splitsWord(replacement, start))) {
    start--;
  }
  let tail = 0;
  const room = Math.min(quote.length, replacement.length) - start;
  while (tail < room && quote.at(-1 - tail) === replacement.at(-1 - tail)) {
    tail++;
  }
  // Forward to the end of the word the difference ends in.
  while (tail > 0 && (splitsWord(quote, quote.length - tail) || splitsWord(replacement, replacement.length - tail))) {
    tail--;
  }
  return { start, end: quote.length - tail, insert: replacement.slice(start, replacement.length - tail) };
}
