import type { Version } from "./post.ts";
import type { Stored } from "./store.ts";

// What the page decides on its own when saving the open post, kept apart from the DOM and the network so each
// decision is a plain function with a test (saving.test.ts). client.ts does what these return.

// The open post's index.md as the page last read or saved it.
export interface OnDisk {
  text: string;
  version: Version;
  // A save was refused because the file changed elsewhere: saving stops until the post is opened again.
  conflicted: boolean;
}

export const opened = ({ markdown, version }: Stored): OnDisk => ({ text: markdown, version, conflicted: false });

// What a save does with the editor's text: nothing after a conflict, only report "saved" when the text is back to
// what's on disk (an undo, say), and otherwise write it.
export type SaveStep = "blocked" | "unchanged" | "write";

export function saveStep(disk: OnDisk, text: string): SaveStep {
  if (disk.conflicted) {
    return "blocked";
  }
  return text === disk.text ? "unchanged" : "write";
}

// Whether coming back to the tab takes the file as it is now on disk, like Zed and VS Code reload a clean buffer:
// only with no unsaved typing, and only if the file changed. With unsaved typing, the next save reports the conflict.
export const shouldReload = (disk: OnDisk, text: string, latest: Version): boolean =>
  text === disk.text && latest !== disk.version;
