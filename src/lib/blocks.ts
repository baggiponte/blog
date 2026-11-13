import { type Block, block } from "./fenced-blocks.ts";
import { playerBlock } from "./remark-player.ts";

// The fenced blocks a post can hold instead of code: the build renders them (src/lib/markdown.ts) and the editor
// checks them before saving (src/editor/post.ts). A new kind of card is one more entry here.
export const blockKinds: readonly Block[] = [block(playerBlock)];
