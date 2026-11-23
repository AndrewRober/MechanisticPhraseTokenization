#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
PY="${PYTHON_BIN:-python3}"

echo "=== Runtime & Unit Verification ==="
"${PY}" scripts/validate_stage_a.py
"${PY}" scripts/validate_breakthroughs.py

echo "=== Stage A + Shared Pilot Launch ==="
"${PY}" -u scripts/prepare_and_run_pilot.py

echo "=== Stage B All 11 Methods ==="
"${PY}" -u scripts/run_stage_b_all.py

echo "=== Stage D Breakthroughs & Iso-FLOPs ==="
"${PY}" -u scripts/run_breakthroughs_stage_cd.py

echo "=== Stage C 300M 3-Seed & Stage E 150M 5x Scale-Up ==="
"${PY}" -u scripts/run_stage_ce_scaleup.py
