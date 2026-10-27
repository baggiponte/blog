// Everything a visitor reads outside of posts lives here, so copy edits never touch templates.
export const site = {
  name: "luca baggi",
  title: "luca baggi",
  description: "product-minded data scientist writing about ai, genai and the engineering around them.",
  bio: {
    lead: "product-minded data scientist",
    rest: ", using ai and genai for problems that actually pay the bills. these days i'm mostly",
  },
  // Cycles slowly under the bio; one line per entry, lowercase.
  lately: [
    "turning slop into concrete solutions.",
    "making llms do the boring parts of the job.",
    "shipping models that survive monday morning.",
    "reading the docs so you don't have to.",
  ],
  nav: [
    { label: "writing", href: "/" },
    { label: "about", href: "/about/" },
    { label: "talks", href: "https://talks.baggiponte.com/", external: true },
  ],
  links: [
    { label: "rss", href: "/rss.xml" },
    { label: "github", href: "https://github.com/baggiponte" },
  ],
  footnote: "written by a human, mostly",
};
