// Bundle directories keep their ordering prefix on disk (01-groundup-...); URLs drop it.
export const slugOfDir = (dir: string): string => dir.replace(/^\d+-/, "");
export const urlOfDir = (dir: string): string => `/blog/${slugOfDir(dir)}/`;

// "Café au lait!" -> "cafe-au-lait": lowercase letters and digits joined by single dashes. Accents are dropped rather
// than the letters they sit on.
export function slugify(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
