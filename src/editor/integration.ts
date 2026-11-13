import { fileURLToPath } from "node:url";

import type { AstroIntegration } from "astro";

import { apiBase } from "./base.ts";
import { send, toRequest } from "./node.ts";

// Loaded through Vite on each request, not imported, so edits to the editor's API, store and page apply on the next
// request instead of after restarting `astro dev`.
type Api = typeof import("./api.ts");
type Store = typeof import("./store.ts");
type Page = typeof import("./page.ts");
const moduleAt = (name: string): string => fileURLToPath(new URL(name, import.meta.url));

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
                "@codemirror/commands",
                "@codemirror/lang-markdown",
                "@codemirror/lang-yaml",
                "@codemirror/language",
                "@codemirror/language-data",
                "@codemirror/state",
                "@codemirror/view",
                "@lezer/highlight",
                "@replit/codemirror-vim",
                "astro/zod",
                "codemirror",
              ],
            },
          },
        });
      },
      "astro:server:setup": ({ server }) => {
        const blogDir = fileURLToPath(new URL("../../content/blog/", import.meta.url));
        server.middlewares.use(apiBase, (request, response, next) => {
          Promise.all([server.ssrLoadModule(moduleAt("./api.ts")), server.ssrLoadModule(moduleAt("./store.ts"))])
            .then(([api, store]) => (api as Api).editorApi((store as Store).fileStore(blogDir))(toRequest(request)))
            .then(answer => send(response, answer))
            .catch(next);
        });
        server.middlewares.use("/editor", (request, response, next) => {
          if (request.method !== "GET" || !["/", ""].includes(request.url ?? "")) {
            next();
            return;
          }
          if (request.originalUrl === "/editor") {
            response.writeHead(301, { Location: "/editor/" }).end();
            return;
          }
          server
            .ssrLoadModule(moduleAt("./page.ts"))
            .then(module => {
              response.setHeader("Content-Type", "text/html; charset=utf-8");
              response.end((module as Page).editorPage);
            })
            .catch(next);
        });
      },
    },
  };
}
