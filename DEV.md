# Development Guide

## Build

Simplly `bash build.sh`

## Run Tests

After build, you will find test executables under `build/tests`. You can run any of them in bash, for example:

```bash
./build/tests/core/math
```

## Formatting

This project uses automatic code formatting before each commit. The formatting is enforced through Git hooks.

### Setup

First install `clang-format` which is required by the formatter script(`formatter.sh`):

```bash
sudo apt-get install clang-format
```

To set up the Git hooks, run:

```bash
./scripts/setup-git-hooks.sh
```

This will create a symlink from `.git/hooks/pre-commit` to `scripts/pre-commit`, which runs the formatter script (`formatter.sh`) before each commit and automatically stages any changes made by the formatter.