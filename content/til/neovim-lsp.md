---
date: 2025-06-13
draft: true
title: 'Setting up basedpyright for neovim'
tags: ['neovim']
---

Neovim is my daily driver, though recently I picked up Zed for its AI capabilities. It all started since I was working on a project with a fairly complicated setup (a monorepo with also a uv workspace).

```text
├── .github
├── python
│   ├── agents
│   ├── core
│   ├── flows
│   ├── webservice
│   ├── .gitignore
│   ├── .python-version
│   ├── justfile
│   ├── pyproject.toml
│   ├── README.md
│   └── uv.lock
├── frontend
│   └── ...
├── .gitignore
└── README.md
```

Notice how the uv workspace, inside the `python` folder, is not at the top level of the repository.

Shortly after I migrated to LSP settings 0.11, my LSP stopped working. After some thinking, I realised I made a very minor change that alone broke my setup:

```lua
---@type vim.lsp.Config
return {
  cmd = { 'uvx', '--from=basedpyright', '--', 'basedpyright-langserver', '--stdio' },
  root_markers = {
    '.git', -- the error is here!
    'uv.lock',
    '.venv',
    'pyproject.toml',
  },
  filetypes = { 'python' },
  settings = {
    basedpyright = {
      verboseOutput = true,
      disableOrganizeImports = true,
      analysis = {
        diagnosticMode = 'workspace',
        typeCheckingMode = 'standard',
        inlayHints = {
          variableTypes = true,
          -- callArgumentNames = true,
          functionReturnTypes = true,
          genericTypes = true,
        },
      },
    },
  },
}

```

The error is precisely that I had a `.git` folder as the *first* root marker. This meant that `basedpyright`'s root was in the top level, where there was no virtual environment to discover. Order matters: once I swapped for `uv.lock` first, everything was working smoothly as ever.

One more thing. I am placing `pyproject.toml` last, as with uv workspaces you might have more such files spread across your repo. You need to find a unique file first: the `uv.lock` is a good candidate, followed by `.venv` I guess (since you would usually never bother creating a venv with a different name from the default - the less you touch those things, the better).
