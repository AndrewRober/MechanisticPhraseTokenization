#!/usr/bin/env bash
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p "${REPO_DIR}/artifacts"
STATUS_FILE="${REPO_DIR}/artifacts/STAGE_CE_EXIT_STATUS.json"
rm -f "${STATUS_FILE}"

write_exit_status() {
    local code=$?
    printf '{"exit_code": %d, "finished_utc": "%s"}\n' "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
    echo "[StageCE] Wrote ${STATUS_FILE} with exit_code=${code}"
}
trap write_exit_status EXIT

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
PY="${PYTHON_BIN:-python3}"

echo "=== [Stage C & E Launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"${PY}" -u scripts/run_stage_ce_scaleup.py
