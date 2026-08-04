#!/usr/bin/env bash
# Type-check against the oldest toolchain CI uses.
#
# `python_version` in pyproject does not change which numpy is installed, so a bare
# `np.ndarray` can type-check locally under a new numpy and fail on the 3.10 runner under an
# older one. This builds an environment matching what CI actually resolves for Python 3.10
# and runs mypy with the project's real config.
#
# Run it before pushing anything that touches numpy types.
set -euo pipefail

ENV_DIR="${TMPDIR:-/tmp}/context-grid-oldest"
NUMPY_VERSION="2.2.6"   # the newest numpy Python 3.10 can install

if [ ! -d "$ENV_DIR" ]; then
  echo "building $ENV_DIR ..."
  python3.12 -m venv "$ENV_DIR"
  "$ENV_DIR/bin/pip" install -q "numpy==$NUMPY_VERSION" "mypy>=2.3" pymupdf pdfplumber
fi

echo "mypy $("$ENV_DIR/bin/mypy" --version | cut -d' ' -f2) / numpy $NUMPY_VERSION / target py3.10"
exec "$ENV_DIR/bin/mypy" --config-file pyproject.toml
