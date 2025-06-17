---
date: '2025-06-17T18:31:13+02:00'
draft: true
title: 'Using AI Assistants'
---

Today's recommended read comes from Peter Steinberger. He tells how, during a 24-hour long hackathon, he built [VibeTunnel](https://vibetunnel.sh/) (a browser-based terminal) with a team of three - including legendary (Python and Rust) programmer Armin Ronacher.

In the blogpost, Peter outlines how they used Claude Code to achieve "something that works" at a much greater speed:

> 20x is not an understatement in terms of how much faster we are with agents

> Claude excels at bootstrapping. Need to integrate a library you’ve never used? Claude will get you 80% there in minutes. Want to understand how Server-Sent Events work? Claude generates a working example faster than you can read the MDN docs.

> The workflow that emerged was fascinating: Claude would generate the initial implementation, Mario would test it, discover the edge cases, then spend significant time refactoring. But here’s the key insight - even with all the fixes needed, we were still moving 5x faster than coding from scratch. It’s not about getting perfect code; it’s about getting something that works, then iterating rapidly.

I won't go on quoting, since I might as well copy the whole thing verbatim. The clear takeaway that emerges from the "agentic coding assistant" and "Vibe coding" trend is clearer by the day: these are amazing tools to get a *minimum hackable product*. It might not be capable to deliver *viable* products, but it's usually design and scaffolding that are bottlenecks. In that regard, tools like Claude Code are formidable.

The first reason why I like Claude Code so much is that - despite being clearly expensive - it's terminal-first and, unlike OpenAI's `codex`, can access the internet. OpenHands announced just today their [CLI](https://docs.all-hands.dev/usage/how-to/cli-mode) for agentic coding assistance, and I'll definitely give it a go in the upcoming days.

Do yourself a favour and do *not* install it with `pip`. Run it with:

```bash
uvx --from=openhands-ai -- openhands
```
