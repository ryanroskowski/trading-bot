#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

echo "[+] Using project root: $ROOT"

if command -v py >/dev/null 2>&1; then
  PY="py -3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "[+] Creating venv with: $PY"
$PY -m venv .venv

if [ -f .venv/Scripts/activate ]; then
  # Windows (Git Bash)
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # Linux/WSL/Mac
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[+] Running pytest"
pytest -q



