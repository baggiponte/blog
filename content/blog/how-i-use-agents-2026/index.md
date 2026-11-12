---
title: "How I use agents, November 2026"
description: "Five key aspects about how my agent workflow changed in the past months"
date: 2026-11-05
format: post
draft: true
---

The way I work with agents has changed dramatically after the summer, and I think it's finally time to sit down and write about it.
Before I get started: this was inspired a lot from [Thorsten's latest post about it](https://x.com/thorstenball/status/2114386050638336082?a=ntwTZikRT1ZWxWKAJEs&s=20). Make sure to read it!
The rest of the post goes like this: a paragraph for each of the key points, and a closing section about what I think is coming next, or I am excited about. Here's the key takeaway: **be more ambitious, dare to ask**. Models can one shot features and work alone for at least half an hour; they can design A/B tests to verify how you had them change prompts and skills; comfortably navigate across complex and messy data platforms. You should update how you use them to match these new capabilities. And stop babysitting them: ==ask more==.

## Cloud agents

Agents in the cloud are becoming the default everywhere. Amp has had it for a while, with [Orbs](https://ampcode.com/what-are-orbs). Claude and ChatGPT have it too: after all, [OpenAI acquired Ona](https://ona.com/stories/ona-joins-openai), formerly Gitpod, precisely for this. Companies like [Paradev](https://paradev.com/) (formerly Superset), [Capy](https://capy.ai/) and [Conductor](https://www.conductor.build/) (acquired by Meta) offer cloud environments for their workspace-like products, building on primitives like [durable objects](https://earendil.com/posts/pi-durable-on-cloudflare/).
This unlocks many use cases, the simplest one being that I can start a session from my phone while on the metro, and check the results once I hit my desk at the office. Besides, cloud sessions run in sandboxes, if you are concerned about agents touching files on your personal machine (which still needs to happen, from time to time).
This is nothing new: people have been doing that for at least one year, back when Mac Minis did not cost a fortune. But now it's gone mainstream, and the biggest players offer it for free in their subscription.
What gets me excited about this is **where it's all headed**. Amp Orbs can already be shared: a teammate can send me a thread to collaborate or finish the job. People work on common threads via Slack, and new products like Buzz [are becoming more stable](https://buzz.xyz/). It just makes sense for these workflows to run in cloud environments.
The only downside I see is when your cloud agents don't run the same version of programs on your local machine: this happened too many times with Claude threads. Just prompt the agent to setup a script that does that 🙃 (Nix users are gonna love this.)

## Farewell, TUI

Everyone who knows me knows how much I love(d) working from the terminal, using ~~obscure~~ well-designed CLIs. I can say I didn't have an IDE installed on my machine well before it was mainstream: Neovim has _always_ been my favourite editor. In fact, I started writing this post inside Neovim, while Claude was drafting a bespoke blog post editor. Well, things changed.
Last year I wrote that "coding agents belong to the terminal". It was not wrong at the time, but it's not how agents should be used now.
I need a GUI for the same reason I used Neovim when we mainly used to write code: _speed_. Desktop apps make it _easier_ to work with multiple projects with multiple threads; there's an integrated browser to manage pull requests or review HTML artifacts; ChatGPT has a great annotation tool to provide precise feedback about web pages. And don't forget about plugins AKA MCPs apps! The first time I used the Miro plugin inside ChatGPT I wasn't expecting to see a fully functioning canvas pop out in the middle of the session.
If you look at it, this isn't just a matter of " I don't want to juggle between the browser and the terminal". ==Agents are moving where work happens==. Companies run agents via Slack. Matt Pocock has agents run inside channels in his private Discord server. I don't mean to dunk on the terminal or TUIs: ==it's just that we are developing better interfaces that match the new and improved capabilities of LLMs, and you might not want to end up stuck in the 2026-equivalent of tab-completion==. We moved past that. Keep trying out new things!

## HTML, HTML everywhere

Markdown artifacts are done for. They were never the right abstraction to begin with. They were hard to read, _when_ we read them. There's a reason the web is not just plain text: we grok concepts more easily when they are visual, better yet **interactive**.
Now I ask agents to build bespoke, one-off HTML files: mostly for interactive explainers and planning documents _à la_ [Plannotator](https://plannotator.ai/), with clear "Yes/No/Later" buttons and more ergonomic ways to add feedback or make precise edits. HTML files enable more back and forth and more collaborative design.
Plan mode might be dead, but planning isn't. Asking the agent to put previews, interactive diagrams, or short screencasts, helps me _choose_ and iterate on the design better and faster. It's much easier to reason about three mock-ups and pick one, rather then reading about them in abstract. You can throw away these files afterwards, or attach to the pull request description, as to keep a history of the feature (I don't like committing them).

## One orchestrator, individual project threads

Models have become smart enough not to get lost. I use Claude projects, with a long-running coordinator session to manage something as broad as "this blog". Threads cache hit ratios rarely fall below 98%. I don't know when, or how, Claude does auto compaction. I don't think I have memories turned on: the coordinator thread remembers the preferences that surface from the thread.
On the job, I reproduced a similar setup: I currenlty running GitHub Copilot's Opus 5.5 on OpenCode Desktop and I have been working on a 400k-tokens-long orchestrator session since Friday. No performance drops.
To be fair, a colleague of mine cleverly pointed out that this might entail higher costs: for example, when I get back to the session after the lunch break, the whole session needs to be written to the cache again. However, I don't see how starting a new session from zero can be cheaper. What I like the most is that I don't have to juggle between many sessions.
I am not in love with Claude main thread. I find it mostly routes my messages to existing and working threads. With OpenCode Dekstop I _feel_ I could customise this behaviour more easily. To be honest, I don't really think this has an impact on the model outcomes to fix now.
On the bright side: I don't need AGENTS.md files. Nor use any skill, beside custom ones to instruct Claude on the format I want it to use for pull request descriptions. Sometimes I run /grill-me when I want planning to be more thorough.

## Code reviews: intent, artifacts and tests

I changed my view about code reviews, too. I don't think we need to validate the code we write as much as we did in the past. To be fair: there have always been pull requests where code was never meant to be reviewed - think of exploratory data analysis.
Most of the time, when it comes to my personal projects, I don't really care about the code as much as the outcome. I never reviewed a single line of code for this blog, though I routinely ask what it could be done to improve the codebase coherence. It's also a nice way to learn a bit of TypeScript in the meantime.
What I do care about is for the agent to provide an informative description. Above all, I want to see visual proof of the changes. [visual-pr](https://github.com/humanlayer/skills#visual-pr) from HumanLayer helps. When it's possible, seeing a screenshot or screencast helps a lot to narrow down the scope of the review. At first, I had the agent upload a screen recording of the changes to the blog. It was very expensive in terms of tokens, so now I just check the preview build and have the agent perform the fixes: it's just faster.
This first step is about confirming whether what was built matches my intent. Then I focus on the how, and how it was tested. I want the agent to tell me _why_ something was done and not _what_ (in the PR description it's just noise): the trade-offs, the alternatives and why they were discarded. This is more of a recap and a proof of the planning session, but some things might come up when implementing so I review those there.
These two steps give me ==guidance== on how to perform the review: requirements first, architecture second, and finally the code. It's much easier to perform a _layered_ review that it's ever been.
At this point I can decide whether I need to review the code. Demos serve also serve as a proof that something works, moreso than tests, which agents are still bad at writing and are my favourite entrypoint for code review. The most successful prompt I use for refactoring goes a bit like this:

> Look for tangled tests that need heavy mocking or reach into internals instead of going through a clean interface, and use them as a signal of tightly coupled abstractions or weak encapsulation.

In this way, the model finds a lot of coupled interfaces and suggests a better architecture that follows Uncle Bob's Good Ol' SOLID Principles. In [this refactoring session for the blog](https://github.com/baggiponte/blog/pull/75), the model discovered a lot of tests were reading the same mocked JSON file and split the interface in two.
In general, I am okay with the first implementation being sketchy and have another round of refactoring. If you force the agent to write clean code from the start, it might lose sight of the goal, but most importantly it will eat up tokens and take more time, so your iterations will be slower.
In general, I try to apply the "tiered review" approach that [Matt Pockock presented at AI Engineer in Paris](https://www.youtube.com/watch?v=LlgiOCmFG_w). I won't ~~delve~~ dig into the details here, but I think it's ok to be more lenient when the changes have a low blast radius and are a one-way door, i.e. they can be easily reverted. Of course, if I am writing a migration for the production database I would read every pixel of it.

## What's next

I don't think the points I discussed are quite novel, nor groundbreaking. They have simply become mainstream. I think I'd like to try to set up automations (the infamous "loop engineering"), for example to trigger automatic code reviews.
For my job, I am mostly looking forward to "goal engineering", i.e. let the agent build a harness to hillclimb against a well-defined metric. I think we can build simple and effective loops to improve machine learning models. My experience is still quite limited as of today: I had the agent play with my [Neovim config startup time](https://github.com/baggiponte/dotfiles/pull/9) and I was simply baffled. The agent broke down the startuptime, build framegraphs, suggested changes, and benchmarked the improvements.
On the user experience side, I'm intrigued by the new wave of multiplayer products like Buzz collaborative chat or [Multica](https://multica.ai/)'s kanban view (which I haven't tried), or [Intent](https://intentapp.dev/) approach that replaces chat sessions with "spaces".

I'm genuinely fascinated about building bespoke interfaces with plugin-first architectures. Pi came first and then came OpenCode V2, and I guess this philosophy came to Desktop apps with DeepSeek Harness. Now you can ask OpenCode Desktop, and even Claude (!), to mod its UI.
I still think the most important unsolved problem is _allow a team to collaborate and record (architectural) choices_ when a project evolves, so I am looking forward to [Chopin](https://githubnext.com/projects/chopin/) by Maggie Appleton's team at GitHub Next to become GA. The blog post is worth reading, and you can listen to the extended version of it in this episode of [The Pragmatic Engineer](https://open.spotify.com/episode/1gdAYoMg9qnu3KRlh9b0KY?si=c450e5730fc64730). I'm excited for a future where teams can share and commit their sessions, so we can record how software gets built, more faithfully then ever.
