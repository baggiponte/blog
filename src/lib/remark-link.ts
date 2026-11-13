import type { Paragraph, PhrasingContent } from "mdast";

import { type BlockKind, part, text } from "./fenced-blocks.ts";
import { type Link, linkSchema } from "./link.ts";

// A ```link block becomes a link card, anywhere in a post; several in a row read as a reading list:
//
//   ```link
//   url: https://jvns.ca/blog/2024/02/16/popular-git-config-options/   # the only field required
//   title: popular git config options
//   description: the options people actually set, and why
//   site: julia evans                                                 # the host when left out
//   image: ./git-config.png                                           # a file next to index.md
//   ```
//
// The fields are checked as the player block's are (src/lib/link.ts): a typo stops the build with file and line, and
// the editor refuses to save it (src/editor/post.ts).

// The card as Markdown nodes that turn into HTML elements; the image stays a Markdown image, resized by Astro.
export function linkCard(link: Link): Paragraph {
  // Twice the widest picture the card shows, in the 1.91:1 shape sites draw their previews in.
  const image: PhrasingContent[] =
    link.image === undefined
      ? []
      : [
          part("span", "link-img", [
            { type: "image", url: link.image, alt: "", data: { hProperties: { width: 480, height: 252 } } },
          ]),
        ];
  return {
    type: "paragraph",
    data: { hName: "div", hProperties: { className: ["link"], ...(image.length > 0 ? { dataImage: "" } : {}) } },
    children: [
      {
        type: "link",
        url: link.url,
        data: { hProperties: { className: ["link-card"], target: "_blank", rel: "noopener" } },
        children: [
          part("span", "link-t", [
            text("b", "link-title", link.title),
            ...(link.description === undefined ? [] : [text("span", "link-desc", link.description)]),
            part("span", "link-site", [
              text("span", "link-mark", link.site.slice(0, 1)),
              text("span", "link-name", link.site),
              ...(link.site === link.host ? [] : [text("span", "link-host", link.host)]),
              { type: "text", value: " ↗" },
            ]),
          ]),
          ...image,
        ],
      },
    ],
  };
}

export const linkBlock: BlockKind<Link> = { lang: "link", schema: linkSchema, render: linkCard };
