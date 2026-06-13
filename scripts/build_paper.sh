#!/usr/bin/env bash
# Compile paper/main.tex -> paper/main.pdf using the local TeX Live at ~/texlive/2026.
# TEXINPUTS points at figures/ because main.tex references Figure1.pdf / Figure2.pdf by
# bare filename while they live in paper/figures/. Runs pdflatex twice to resolve refs.
set -euo pipefail

export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/paper"
export TEXINPUTS=".:./figures:"

pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

echo "-> $REPO_ROOT/paper/main.pdf"
