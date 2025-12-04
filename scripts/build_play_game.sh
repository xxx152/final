#!/usr/bin/env bash
set -euo pipefail

# Build a standalone play_game binary with PyInstaller, bundling numpy/pygame/torch.
# Uses system python (or $PYTHON if set). No venv required to run the built binary.

PYTHON=${PYTHON:-python3}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# Ensure PyInstaller is available
if ! "$PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "[build] Installing PyInstaller to user site..."
  "$PYTHON" -m pip install --user pyinstaller
fi

# Optional: ensure runtime deps available at build time
"$PYTHON" -m pip install --user numpy pygame torch || true

# Clean previous outputs
rm -rf build dist play_game.spec || true

# Build single-file binary and collect required packages
"$PYTHON" -m PyInstaller \
  --onefile \
  --name play_game \
  --paths src \
  --hidden-import=numpy \
  --hidden-import=torch \
  --hidden-import=pygame \
  --collect-all numpy \
  --collect-all pygame \
  --collect-all torch \
  play_game.py

echo "[build] Done. Binary at dist/play_game"
