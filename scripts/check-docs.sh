#!/usr/bin/env bash
# Run every python/bash/sh code example in docs/**/*.md and report which ones fail.
#
# Usage:
#   scripts/check-docs.sh                    # check every file under docs/
#   scripts/check-docs.sh docs/scoring/*.md   # check specific files
#   scripts/check-docs.sh -v                  # also print PASS lines
#
# Exits non-zero if any example failed. See scripts/check_docs.py for the extraction and
# execution logic, and the design decisions (shared state per file, the `no-run` marker).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: $VENV_PYTHON not found or not executable. Set up .venv first." >&2
    exit 2
fi

exec "$VENV_PYTHON" "$REPO_ROOT/scripts/check_docs.py" "$@"
