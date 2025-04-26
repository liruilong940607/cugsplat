#!/bin/bash

# Create a symlink from .git/hooks/pre-commit to scripts/pre-commit
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit

echo "Git hooks have been set up successfully!" 