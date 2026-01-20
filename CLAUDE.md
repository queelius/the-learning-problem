# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This is a collection of blog posts, part of a series on metafunctor.com.

## Build Commands

```bash
make docs        # Build mkdocs site
make docs-serve  # Serve locally at localhost:8000
make docs-clean  # Clean build artifacts
```

## Structure

```
repo/
├── post/           # Blog posts (Hugo/mkdocs compatible)
│   └── YYYY-MM-DD-title/
│       └── index.md
├── docs/           # mkdocs source
│   ├── index.md    # Landing page
│   └── about.md
├── mkdocs.yml
├── Makefile
└── README.md
```

## Code Style

Posts use YAML frontmatter with author info:
```yaml
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
```
