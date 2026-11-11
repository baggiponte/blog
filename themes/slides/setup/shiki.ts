import { codeThemes } from "../../../src/lib/code-themes.ts";

// The same syntax colours as the blog's code blocks. Slidev's defineShikiSetup only returns its argument, so a plain
// function does the same job without depending on @slidev/types.
export default () => ({ themes: codeThemes });
