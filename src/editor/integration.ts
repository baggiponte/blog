import { fileURLToPath } from "node:url";

import type { AstroIntegration } from "astro";

import { editorApi } from "./api.ts";
import { apiBase } from "./base.ts";
import { editorPage } from "./page.ts";

// The drafting editor at /editor/: only `astro dev` serves it, so the built site never has the page or its API.
export function editor(): AstroIntegration {
  return {
    name: "editor",
    hooks: {
      // Vite only finds dependencies to pre-bundle by crawling pages it serves, and it never serves this one. Without
      // this list, the first visit after a change makes Vite re-bundle mid-load and the page fails until reloaded.
      "astro:config:setup": ({ updateConfig }) => {
        updateConfig({
          vite: {
            optimizeDeps: {
              include: [
                "@codemirror/lang-markdown",
                "@codemirror/lang-yaml",
                "@codemirror/language",
                "@codemirror/language-data",
                "@codemirror/state",
                "@codemirror/view",
                "@lezer/highlight",
                "@replit/codemirror-vim",
                "codemirror",
              ],
            },
          },
        });
      },
      "astro:server:setup": ({ server }) => {
        const blogDir = fileURLToPath(new URL("../../content/blog/", import.meta.url));
        server.middlewares.use(apiBase, editorApi(blogDir));
        server.middlewares.use("/editor", (request, response, next) => {
          if (request.method !== "GET" || !["/", ""].includes(request.url ?? "")) {
            next();
            return;
          }
          if (request.originalUrl === "/editor") {
            response.writeHead(301, { Location: "/editor/" }).end();
            return;
          }
          response.setHeader("Content-Type", "text/html; charset=utf-8");
          response.end(editorPage);
        });
      },
    },
  };
}
