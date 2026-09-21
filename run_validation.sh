#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q .
python -m ruff check .
python -m pytest -q
python verify_inputs.py
