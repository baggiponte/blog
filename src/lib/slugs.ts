// Bundle directories keep their ordering prefix on disk (01-groundup-...); URLs drop it.
export const slugOfDir = (dir: string): string => dir.replace(/^\d+-/, "");
export const urlOfDir = (dir: string): string => `/blog/${slugOfDir(dir)}/`;
