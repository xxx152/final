#!/usr/bin/env bash
set -euo pipefail

# Build a standalone play_game binary (Linux) without requiring a venv.
# It uses the system Python (or $PYTHON if provided) and installs PyInstaller to the user site if missing.

PYTHON=${PYTHON:-python3}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# Ensure PyInstaller is available
if ! "$PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "[build] Installing PyInstaller to user site..."
  "$PYTHON" -m pip install --user pyinstaller
fi

# Clean previous outputs
rm -rf build dist play_game.spec || true

# Build one-file binary
"$PYTHON" -m PyInstaller \
  --onefile \
  --name play_game \
  --paths src \
  --collect-all pygame \
  play_game.py

echo "[build] Done. Binary at dist/play_game"
