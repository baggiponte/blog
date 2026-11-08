// The tools on tools.baggiponte.com, in the order the index lists them. Each one is a folder of static files in
// tools/site/<slug>/ and is served at /<slug>/.
export interface Tool {
  slug: string;
  name: string;
  description: string;
}

export const tools: Tool[] = [
  {
    slug: "metronomo",
    name: "metronomo",
    description: "a metronome with time signatures, subdivisions, an accent on the downbeat and tap tempo.",
  },
];

interface Link {
  label: string;
  href: string;
}

// The blog's own sections, so the header reads like baggiponte.com's, plus the tools index itself.
export const nav: Link[] = [
  { label: "writing", href: "https://baggiponte.com/" },
  { label: "about", href: "https://baggiponte.com/about/" },
  { label: "talks", href: "https://baggiponte.com/talks/" },
  { label: "tools", href: "/" },
];
