#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
PY="${PYTHON_BIN:-python3}"

echo "=== Stage B Orchestrator Launch ==="
"${PY}" -u scripts/run_stage_b_all.py
