interface Link {
  label: string;
  href: string;
}

interface Site {
  name: string;
  title: string;
  description: string;
  bio: { lead: string; rest: string };
  nav: Link[];
  links: Link[];
  footnote: string;
}

// Everything a visitor reads outside of posts lives here, so copy edits never touch templates.
export const site: Site = {
  name: "luca baggi",
  // The browser tab and feed name; a page's own title goes in front, as in "about · luca baggi's blog".
  title: "luca baggi's blog",
  description: "product-minded data scientist writing about ai, genai and the engineering around them.",
  bio: {
    lead: "product-minded data scientist",
    rest: ", using ai and genai for problems that actually pay the bills. these days i'm mostly",
  },
  nav: [
    { label: "writing", href: "/" },
    { label: "about", href: "/about/" },
    { label: "talks", href: "/talks/" },
  ],
  links: [
    { label: "rss", href: "/rss.xml" },
    { label: "github", href: "https://github.com/baggiponte" },
  ],
  footnote: "written by a human, mostly",
};
