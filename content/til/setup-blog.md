---
title: "How to setup a blog with Hugo, Blowfish, and GitHub Pages"
date: 2025-04-09
draft: true
description: "Yet another unasked tutorial about something people have been doing forever"
tags: ["pytorch"]
---

It took me too long to set this up, so I might as well write down what I did.

## Prerequisites

You will need `hugo`, and `git`. I installed them with Homebrew:

```bash
brew install hugo
```

Choosing a theme can take you longer than you might want to. Pick one. I chose [Blowfish](https://blowfish.page/) since [Dario](https://dwarez.github.io/) was already using it (go check him out!) and it was pleasant to the eye.

## Setup

Navigate to the directory where you want to store your blog, and run:

```bash
hugo new site <mysite> <directory>
```

Then install the theme:

```bash
cd <mysite>
git init
git submodule add -b main https://github.com/nunocoracao/blowfish.git themes/blowfish
```

The important step now is to clone the configuration folder in the appropriate place:

```bash
cp -r themes/blowfish/config .
```

Now you can run `hugo server` and see your site at `localhost:1313`.

