# Instructions

This is my personal blog.

## Just Commands
This Hugo blog uses [just](https://github.com/casey/just) for task automation:

* `just start` - Start Hugo development server with drafts
* `just update` - Update Hugo modules and theme

## Git Workflow
Follow these Git best practices when implementing features:

### Branch Management
* Create feature branches from `main` using conventional naming:
  - `feature/description` for new features
  - `bugfix/description` for bug fixes
  - `hotfix/description` for urgent fixes
* Work on feature branches, never directly on `main`
* Delete branches after merging

### Commit Guidelines
* Use conventional commit messages with prefixes:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `test:` for adding tests
  - `docs:` for documentation
  - `refactor:` for code refactoring
  - `chore:` for maintenance tasks

* Make atomic commits (one logical change per commit)
* Include descriptive commit messages

### What to Commit
* **DO commit**: Source code, configuration files, tests, documentation
* **DON'T commit**: API keys, secrets, build artifacts, dependencies, large binaries

### Pull Request Workflow
* **NEVER push directly to `main` branch** - Always create PRs for all changes
* Use `gh pr create` for creating pull requests
* Include summary and test plan in PR description
* Provide PR link for user review - DO NOT auto-merge unless explicitly requested
* Only use auto-merge when all checks pass and user has approved

## Self-Improvement
At the end of each conversation, review the exchange for any mistakes or corrections made by the user. Suggest specific additions or improvements to these rules in CLAUDE.md to prevent similar mistakes in future conversations.
