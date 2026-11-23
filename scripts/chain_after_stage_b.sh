#!/usr/bin/env bash
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p "${REPO_DIR}/artifacts"
STATUS_FILE="${REPO_DIR}/artifacts/PIPELINE_EXIT_STATUS.json"
rm -f "${STATUS_FILE}"

write_exit_status() {
    local code=$?
    printf '{"exit_code": %d, "finished_utc": "%s"}\n' "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
    echo "[AutoChain] Wrote ${STATUS_FILE} with exit_code=${code}"
}
trap write_exit_status EXIT

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
PY="${PYTHON_BIN:-python3}"

STAGE_B_PID=$(pgrep -f '[s]cripts/run_stage_b_all.py' | head -n 1 || true)
if [ -n "${STAGE_B_PID}" ]; then
    echo "[AutoChain] Monitoring Stage B (PID ${STAGE_B_PID})..."
    while kill -0 "${STAGE_B_PID}" 2>/dev/null; do
        sleep 15
    done
fi

echo "=== [AutoChain] Stage B Finished -> Printing Stage B Summary & Launching Breakthrough Suite ==="
"${PY}" -u scripts/summarize_results.py || true
"${PY}" -u scripts/run_breakthroughs_stage_cd.py
