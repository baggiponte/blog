import type { Image, Link, Paragraph, PhrasingContent, Root } from "mdast";
import { visit } from "unist-util-visit";

import { spotifyLink } from "./spotify.ts";

// A line holding only a cover linked to Spotify becomes a card, wherever it sits in the post:
//
//   [![lose yourself](./cover.jpg "eminem")](https://open.spotify.com/track/…)
//
// The alt text is the title and the image's title the artist or show. It stays plain Markdown, so "copy page", the
// editor and GitHub still show a linked cover. The image stays a Markdown image, which Astro resizes like any other.
export interface Part {
  type: "spotifyPart";
  data: { hName: string; hProperties?: Record<string, string | string[]> };
  children: PhrasingContent[];
}
declare module "mdast" {
  interface PhrasingContentMap {
    spotifyPart: Part;
  }
}

const part = (hName: string, className: string, children: PhrasingContent[] = []): Part => ({
  type: "spotifyPart",
  data: { hName, hProperties: { className: [className] } },
  children,
});
const vinyl: Part = {
  type: "spotifyPart",
  data: { hName: "span", hProperties: { className: ["vinyl"], ariaHidden: "true" } },
  children: [],
};
const text = (hName: string, className: string, value: string): Part =>
  part(hName, className, [{ type: "text", value }]);

// The linked cover, when the paragraph holds nothing else (whitespace aside).
function linkedCover(paragraph: Paragraph): { link: Link; image: Image } | undefined {
  const [link, ...rest] = paragraph.children.filter(child => child.type !== "text" || child.value.trim() !== "");
  if (link?.type !== "link" || rest.length > 0 || link.children.length !== 1) {
    return undefined;
  }
  const [image] = link.children;
  return image?.type === "image" ? { link, image } : undefined;
}

export function remarkSpotify(): (tree: Root) => void {
  return tree => {
    visit(tree, "paragraph", paragraph => {
      const found = linkedCover(paragraph);
      const spotify = found && spotifyLink(found.link.url);
      if (found === undefined || spotify === undefined) {
        return;
      }
      const { link, image } = found;
      const title = image.alt ?? "";
      const by = image.title ?? "";
      // The link's text names it for screen readers, so the cover itself is decorative.
      // 2× the largest sleeve, so Astro resizes the cover instead of shipping it at full size.
      const cover: Image = {
        type: "image",
        url: image.url,
        alt: "",
        data: { hProperties: { width: 256, height: 256 } },
      };
      const sleeve = spotify.kind === "song" ? [vinyl, cover] : [cover];

      paragraph.data = { hName: "div", hProperties: { className: ["spotify"], dataKind: spotify.kind } };
      link.url = spotify.url;
      link.data = { hProperties: { className: ["spotify-card"], target: "_blank", rel: "noopener" } };
      link.children = [
        part("span", "sleeve", sleeve),
        part("span", "spotify-t", [
          text("span", "spotify-k", spotify.kind),
          text("b", "spotify-title", title),
          ...(by === "" ? [] : [text("span", "spotify-by", by)]),
          text("span", "spotify-go", "listen on spotify ↗"),
        ]),
      ];
    });
  };
}
