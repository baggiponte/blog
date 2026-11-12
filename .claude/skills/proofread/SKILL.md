---
name: proofread
description: Proofread a blog post and leave the corrections as suggestions the local editor shows inline, to accept, edit or reject. Use when Luca asks to proofread, review or correct a post in content/blog.
---

# proofread a post

Write the corrections to `content/blog/<dir>/suggestions.json`, next to the post's `index.md`. Never edit `index.md` yourself: Luca accepts, edits or rejects each suggestion in the editor at `/editor/` (`pnpm dev`), which shows them on the text.

Only hand-written posts: skip bundles whose frontmatter has `notebook:`.

## the file

```json
{
  "suggestions": [
    {
      "id": "s1",
      "kind": "grammar",
      "quote": "they was writing",
      "replacement": "they were writing",
      "note": "\"they\" takes \"were\"."
    }
  ]
}
```

- `id`: unique in the file, e.g. `s1`, `s2`, …
- `kind`: one of `typo`, `grammar`, `style`, `clarity`, `fact`.
- `quote`: the exact text to replace, copied character for character from `index.md`, Markdown included (`**`, backticks, links). Quote a few words around the change rather than a single word, so the quote appears only once in the file; the editor marks only the words that differ.
- `replacement`: what the quote becomes; `""` deletes it.
- `note`: optional, one short sentence on why. Leave it out for plain typos.
- `prefix` / `suffix`: optional, only when a quote can't be made unique: the text just before and after it.

One suggestion per change, never a whole rewritten paragraph: Luca decides on each one. Frontmatter (title, description) can be suggested too; keep titles lowercase.

## what to look for

Typos and grammar first, then clarity, then style. Keep Luca's voice: suggest a style change only when it clearly reads better, and mark it `style` so he can skip those. Use `fact` for claims that look wrong or outdated, with the reason in `note`. One line of the file is one paragraph: don't suggest hard-wrapping.

## before you finish

If `suggestions.json` already exists, those are suggestions Luca hasn't decided on yet: add to them with new ids rather than overwriting them, unless he asked for a fresh pass.

Then run `node scripts/check-suggestions.ts content/blog/<dir>`. It fails on a quote that isn't in `index.md` or appears more than once: fix those quotes and run it again. Tell Luca how many suggestions you left, by kind, and that they're waiting in the editor.

`suggestions.json` is gitignored: never commit it.
