# Git Hooks

This directory contains Git hooks for the project.

## Setup

To set up the Git hooks, run:

```bash
./scripts/setup-git-hooks.sh
```

This will create a symlink from `.git/hooks/pre-commit` to `scripts/pre-commit`.

## Available Hooks

- `pre-commit`: Runs the formatter script before each commit and automatically stages any changes made by the formatter.

## Manual Setup

If you prefer to set up the hooks manually, you can create a symlink:

```bash
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
``` 