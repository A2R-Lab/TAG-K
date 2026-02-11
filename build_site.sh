#!/usr/bin/env bash
# Build the full website locally.
# Usage: bash build_site.sh
# The result is in _site/ — open _site/index.html in a browser.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Building Sphinx docs ..."
# Resolve Python: prefer venv in this repo, then whatever is on PATH.
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
fi
cd _sphinx_src
"$PYTHON" -m sphinx -b html . ../docs 2>&1
cd ..

echo "==> Assembling site into _site/ ..."
rm -rf _site
mkdir -p _site

cp index.html _site/
cp -r static _site/ 2>/dev/null || true
cp .nojekyll _site/
cp -r docs _site/docs

echo "==> Done!  Open _site/index.html in your browser."
