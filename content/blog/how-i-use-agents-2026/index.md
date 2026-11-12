---
title: "How I use agents, November 2026"
description: "Five key aspects about how my agent workflow changed in the past months"
date: 2026-11-05
format: post
draft: true
---

The way I work with agents has changed dramatically after the summer, and I think it's finally time to sit down and write about it.

Before I get started: this was inspired a lot from [Thorsten's latest post about it](https://x.com/thorstenball/status/2114386050638336082?a=ntwTZikRT1ZWxWKAJEs&s=20). Make sure to read it!

Here's the key takeaway: **be more ambitious, dare to ask**. Models can one shot features and work alone for at least half an hour; they can design A/B tests to verify how you had them change prompts and skills; comfortably navigate across complex and messy data platforms. Stop babysitting them. ==_Ask more_==.
## Cloud agents
Agents in the cloud are becoming the default everywhere. Amp has had it for a while, with [Orbs](https://ampcode.com/what-are-orbs). Claude and ChatGPT have it too: after all, [OpenAI acquired Ona](https://ona.com/stories/ona-joins-openai), formerly Gitpod, precisely for this. Companies like Paradev (formerly Superset) offer cloud environments to for their workspace-like products, building on primitives like [durable objects](https://earendil.com/posts/pi-durable-on-cloudflare/).

This makes a lot of sense. I can start a session from my phone while on the metro, and check the results from my desk when I hit the office. Cloud sessions run in sandboxes, if you are concerned about agents touching files on your personal machine - which still needs to happen, from time to time.

What gets me excited about this is **where it's all headed**. Amp Orbs can already be shared: a teammate can send me a thread to collaborate or finish the job. People work on common threads via Slack, while [Buzz 1.0 was just released](https://buzz.xyz/): these workflows should just happen in cloud environments. I'm excited for a future where teams share and commit their sessions, so we can record how software gets built, more faithfully then ever.

The only downside I see is when your cloud agents don't run the same version of programs on your local machine: this happened too many times with Claude threads. Just prompt the agent to setup a script that does that 🙃  (Nix users are gonna love this.)
## Farewell, TUI
Everyone who knows me knows how much I love(d) working from the terminal, using ~~obscure~~ well-designed CLIs. Neovim has _always_ been my editor: in fact, I started writing this post inside Neovim, while Claude was drafting a bespoke blog post editor. I haven't had an IDE installed on my machine well before it was mainstream. Things changed.
Last year I wrote that "coding agents belong to the terminal". It was not wrong at the time, but it's not how agents should be used now. If you're not using a GUI today, you're doing yourself a disservice.

I need a GUI for the same reason I used Neovim when we mainly used to write code: *speed*. Desktop apps make it easier to work with multiple threads and review HTML artifacts; there's an integrated browser to manage pull requests and annotate web pages to provide precise feedback. And don't forget about **plugins** (basically, MCPs and MCP UI)! The first time I used the Miro plugin inside ChatGPT I wasn't expecting to see a fully functioning canvas pop out in the middle of the session.

If you look at it, this isn't just a matter of " I just don't want to juggle between the browser and the terminal". ==Agents are moving where work happens==. Companies run agents via Slack. Matt Pocock has agents run inside channels in his private Discord server. Interfaces are evolving: inside a GUI you can _also_ have interfaces superior to chats and sessions to oversee agent work. Though you can definitley mod Claude Desktop or DeepSeek Harness to add a kanban view, it can also be something new like [Intent](https://intentapp.dev/).
## HTML, HTML everywhere
Markdown artifacts are done for. They were hard to read, _when_ we read them. There's a reason the web is not just plain text: we grok concepts more easily when they are visual, better yet interactive.

Now I ask agents to build bespoke, one-off HTML files: mostly for interactive explainers and planning documents _à la_ [Plannotator](https://plannotator.ai/), with clear "Yes/No/Later" buttons and more ergonomic ways to add feedback or make precise edits.

Asking the agent to put previews, interactive diagrams, or short screencasts, helps me *choose* and iterate on the design better and faster. It's much easier to reason about three mock-ups and pick one, rather then reading about them or - for some tasks - describing them to the agent. These are files you can throw away after, or attach to the pull request description when you are done, as to keep a history of the feature (I don't like committing them).
## One orchestrator, individual project threads
Models are smart enough now not to get lost. I have a long-running Claude session to manage this blog and threads cache hit ratios are rarely below 98%. I don't even know if it did auto compaction. At work, I am running GitHub copilot via OpenCode Desktop and I have been working on a 400k-tokens-long orchestrator session since Friday. No performance drops.

To be fair, a colleague of mine pointed out that this might entail higher costs: for example, when I get back to the session after the lunch break, the whole session needs to be written to the cache again. However, I don't see how starting a new session from zero can be cheaper. What I like the most is that I don't have to juggle across so many different sessions.
## Code reviews: artifacts, artifacts, artifacts
I changed my view about code reviews, too. I don't think we need to validate the code we write as much as we did in the past. To be fair: there have always been pull requests where code was never meant to be reviewed -- think of exploratory data analysis.
Most of the time, when it comes to my personal projects, I don't really care about the code as much as the outcome. I never reviewed a single line of code for this blog (though I took a lot of pleasure into asking the coding agent to refactor it): instead, I care about the code blocks being readable. I don't really read the amount of typed Lua code that I ask from a coding agent

## What's next

Planning

Small sw factory: standalone review loop. use cheaper sub like deepseek v4.1 to correct lints and nits.
