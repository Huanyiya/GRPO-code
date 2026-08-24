#!/usr/bin/env bash

# Start a standalone 8-PPU SGLang server, then filter Polaris-53K with eight
# non-thinking Qwen3.5-9B samples per problem.  The progress bar belongs to
# the Python filter; SGLang logs are kept in a file so they do not corrupt it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/mnt/cpfs/users/zhy/opd/slime-OPD/.venv/bin/python}

MODEL_DIR=${MODEL_DIR:-/mnt/cpfs/weights/Qwen3.5-9B}
INPUT_PATH=${INPUT_PATH:-/mnt/cpfs/users/zhy/opd/OPD/datasets/Polaris-53K/polaris-data-53K.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/cpfs/users/zhy/opd/OPD/datasets/Polaris-53K/polaris-data-53K-qwen3.5-9b-1to7of8.jsonl}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-8}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-32}
SGLANG_LOG=${SGLANG_LOG:-/tmp/sglang-polaris-qwen3.5-9b-${PORT}.log}

export PYTHONUNBUFFERED=1
export SGLANG_HEALTH_CHECK_TIMEOUT=${SGLANG_HEALTH_CHECK_TIMEOUT:-120}
export SGLANG_WARMUP_TIMEOUT=${SGLANG_WARMUP_TIMEOUT:-1800}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [[ ! -d "${MODEL_DIR}" || ! -f "${INPUT_PATH}" ]]; then
  echo "Missing MODEL_DIR or INPUT_PATH." >&2
  exit 1
fi
if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "Output already exists: ${OUTPUT_PATH}" >&2
  echo "Choose OUTPUT_PATH=/new/path or remove it deliberately before rerunning." >&2
  exit 1
fi

"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name qwen3.5-9b \
  --tensor-parallel-size "${TP_SIZE}" \
  --context-length 18432 \
  --mem-fraction-static "${SGLANG_MEM_FRACTION}" \
  --mamba-scheduler-strategy extra_buffer \
  >"${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!

cleanup() {
  if kill -0 "${SGLANG_PID}" 2>/dev/null; then
    kill "${SGLANG_PID}" 2>/dev/null || true
    wait "${SGLANG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting SGLang (pid=${SGLANG_PID}); log: ${SGLANG_LOG}"
until curl -sf "http://127.0.0.1:${PORT}/health_generate" >/dev/null; do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "SGLang exited. Last log lines:" >&2
    tail -n 80 "${SGLANG_LOG}" >&2 || true
    exit 1
  fi
  printf 'Waiting for SGLang server...\n'
  sleep 5
done

cd "${REPO_ROOT}"
"${PYTHON_BIN}" tools/filter_polaris_by_qwen_rollouts.py \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3.5-9b \
  --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}"
