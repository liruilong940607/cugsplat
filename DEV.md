# Development Guide

## Formatting

This project uses automatic code formatting before each commit. The formatting is enforced through Git hooks.

### Setup

To set up the Git hooks, run:

```bash
./scripts/setup-git-hooks.sh
```

This will create a symlink from `.git/hooks/pre-commit` to `scripts/pre-commit`, which runs the formatter script (`formatter.sh`) before each commit and automatically stages any changes made by the formatter.