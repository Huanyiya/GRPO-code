#!/usr/bin/env bash

# Qwen3.5-4B GRPO, single node / GPUs 0-5 (6 x 96G), colocated train + rollout.
#
# Batch mapping:
#   128 prompts/rollout * 8 responses/prompt = 1,024 samples
#   1,024 samples / 1 optimizer step         = global batch size 1,024
#   500 rollouts                             = 500 optimizer steps
#
# Polaris uses the `problem` and `answer` fields. The custom reward accepts
# scalar answers and checks final boxed answers without requiring </think>.

set -euo pipefail

# torch_memory_saver is incompatible with expandable-segments allocation.
unset PYTORCH_CUDA_ALLOC_CONF

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# Paths can be overridden from the environment.
MODEL_DIR=${MODEL_DIR:-/mnt/cpfs/users/zhy/opd/slime-OPD/ckpt_hf/08_18_00_08_student_Qwen3.5-4B_teacher_Qwen3.5-9B-GRPO-nonthinking-polaris_iter_0000109_hf_iter_0000099_hf}
INIT_CKPT=${INIT_CKPT:-/mnt/cpfs/users/zhy/opd/GRPO/checkpoints/08_18_00_08_student_Qwen3.5-4B_opd_iter99_torch_dist}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AIME25/test.parquet}
SAVE_DIR=${SAVE_DIR:-/mnt/cpfs/users/zhy/opd/GRPO/checkpoints/Qwen3.5-4B-opd-step99-GRPO}

# Training-data preset. Set TRAIN_DATASET=polaris53k to select Polaris-53K;
# DATA_PATH can still override the selected preset for an arbitrary compatible
# dataset.
TRAIN_DATASET=${TRAIN_DATASET:-dapo}
DAPO_DATA_PATH=/mnt/cpfs/users/zhy/opd/OPD/datasets/dapo-math-17k-processed.parquet
POLARIS_DATA_PATH=/mnt/cpfs/users/zhy/opd/OPD/datasets/Polaris-53K/polaris-difficulty-1to5-random-20k.jsonl
case "${TRAIN_DATASET}" in
  dapo)
    DEFAULT_DATA_PATH="${DAPO_DATA_PATH}"
    TRAIN_INPUT_KEY=prompt
    TRAIN_LABEL_KEY=reward_model
    ;;
  polaris|polaris53k)
    DEFAULT_DATA_PATH="${POLARIS_DATA_PATH}"
    TRAIN_INPUT_KEY=problem
    TRAIN_LABEL_KEY=answer
    ;;
  *)
    echo "Unknown TRAIN_DATASET=${TRAIN_DATASET}; choose dapo or polaris53k." >&2
    exit 1
    ;;
esac
DATA_PATH=${DATA_PATH:-${DEFAULT_DATA_PATH}}
echo "Training dataset: ${TRAIN_DATASET} (${DATA_PATH}; input=${TRAIN_INPUT_KEY}; label=${TRAIN_LABEL_KEY})"

# Existing PPU environment and Megatron checkout.
PYTHON_BIN=${PYTHON_BIN:-/mnt/cpfs/users/zhy/opd/slime-OPD/.venv/bin/python}
RAY_BIN=${RAY_BIN:-/mnt/cpfs/users/zhy/opd/slime-OPD/.venv/bin/ray}
MEGATRON_PATH=${MEGATRON_PATH:-/mnt/cpfs/users/zhy/opd/slime-OPD/Megatron-LM}

# This launcher owns physical GPUs 0-5 only.  Ray will map these to logical
# GPU IDs 0-5 for both the colocated Megatron actors and SGLang engines.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.75}
# On PPU the first compiled forward can take longer than SGLang's 20 s
# default health-check request.  A timed-out health check is reported as a
# client disconnect even when the engine itself is healthy.
SGLANG_HEALTH_CHECK_TIMEOUT=${SGLANG_HEALTH_CHECK_TIMEOUT:-120}
SGLANG_WARMUP_TIMEOUT=${SGLANG_WARMUP_TIMEOUT:-1800}
CACHE_ROOT=${CACHE_ROOT:-${SAVE_DIR}/runtime-cache}
# On this PPU, rebuilding PCCL groups and then exporting CUDA IPC tensors can
# crash inside HGGC.  Use the OPD-proven path by default: export HF weights
# while Megatron is resident, CPU-offload it, then reload SGLang from disk.
WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT:-disk}
UPDATE_WEIGHT_DISK_DIR=${UPDATE_WEIGHT_DISK_DIR:-${SAVE_DIR}/weight-sync}


WANDB_DIR=${WANDB_DIR:-${SAVE_DIR}/wandb}

for required_path in "${MODEL_DIR}" "${INIT_CKPT}" "${DATA_PATH}" "${EVAL_DATA_PATH}" "${PYTHON_BIN}" "${RAY_BIN}" "${MEGATRON_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [[ ! -f "${INIT_CKPT}/latest_checkpointed_iteration.txt" ]]; then
  echo "INIT_CKPT is not a Megatron torch_dist checkpoint: ${INIT_CKPT}" >&2
  echo "Generate it first, or override INIT_CKPT=/path/to/your_torch_dist_checkpoint." >&2
  exit 1
fi

if [[ "${NUM_GPUS}" -ne 8 ]]; then
  echo "This script is configured for exactly 8 GPUs (physical IDs 0-7); got NUM_GPUS=${NUM_GPUS}." >&2
  exit 1
fi

source "${SCRIPT_DIR}/models/qwen3.5-4B.sh"

export PYTHONUNBUFFERED=1
export MASTER_ADDR MASTER_PORT
export SGLANG_HEALTH_CHECK_TIMEOUT SGLANG_WARMUP_TIMEOUT
# HGGC's NVML does not implement nvmlDeviceGetProcessesUtilizationInfo. Ray
# 2.55's dashboard agent otherwise exits during GPU metric collection, which
# makes raylet exit and causes `ray start` to time out.
export RAY_DISABLE_DASHBOARD_GPU_METRICS=1
# tmux servers keep the environment from the time they were created.  In
# particular, SGLang Model Gateway (Rust) also honors ALL_PROXY, not only the
# usual HTTP(S)_PROXY variables.  A proxy must never sit between the gateway
# and its local 172.18.x.x worker ports.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY NO_PROXY RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

mkdir -p \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}"

NVLINK_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
fi
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} (${NVLINK_COUNT} NVLink references detected)"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  # First run falls back to INIT_CKPT; later runs resume from SAVE_DIR.
  --ref-load "${INIT_CKPT}"
  --load "${SAVE_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 20
)

ROLLOUT_ARGS=(
  --prompt-data "${DATA_PATH}"
  --input-key "${TRAIN_INPUT_KEY}"
  --label-key "${TRAIN_LABEL_KEY}"
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle

  --num-rollout 140
  --rollout-batch-size 128
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1
  --global-batch-size 1024

  --rollout-max-prompt-len 2048
  --rollout-max-response-len 16384
  --rollout-max-context-len 18432
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --balance-data
)

EVAL_ARGS=(
  # One rollout equals one optimizer step; this evaluates every 50 steps.
  --eval-interval 500
  --skip-eval-before-train
  --eval-prompt-data AIME25 "${EVAL_DATA_PATH}"
  --eval-input-key prompt
  --eval-label-key reward_model
  --eval-prompt-suffix $'\n\nGive your final answer on its own last line in the exact form \\boxed{answer}.'
  --n-samples-per-eval-prompt 16
  --eval-max-prompt-len 2048
  --eval-max-response-len 16384
  --eval-max-context-len 18432
  --eval-temperature 1.0
  --eval-top-p 0.95
  --eval-top-k 20
  --eval-presence-penalty 1.5
  --eval-repetition-penalty 1.0
)

REWARD_ARGS=(
  --custom-rm-path slime_plugins.rewards.dapo_math.reward_func
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-coef 0.0
  --kl-loss-coef 0.0
  --entropy-coef 0.0
  --eps-clip 0.2
)

WEIGHT_SYNC_ARGS=()
case "${WEIGHT_SYNC_TRANSPORT}" in
  nccl)
    ;;
  disk)
    mkdir -p "${UPDATE_WEIGHT_DISK_DIR}"
    WEIGHT_SYNC_ARGS=(
      --update-weight-mode full
      --update-weight-transport disk
      --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
    )
    ;;
  *)
    echo "Unknown WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT}; choose nccl or disk." >&2
    exit 1
    ;;
esac

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --override-opt-param-scheduler
)

WANDB_ARGS=(
  --use-wandb
  --wandb-mode online
  --wandb-project Qwen3.5-4B-opd-step-99-GRPO
  --wandb-group dapo-math-20k-nonthinking
  --wandb-dir "${WANDB_DIR}"
  --log-passrate
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --pipeline-model-parallel-size 1
  --context-parallel-size 4
  # CP=3 requires seq-length to be divisible by 2 * CP.  This also matches
  # the 2,048-token prompt + 16,384-token response rollout context.
  --seq-length 18432
  --sequence-parallel
  --use-distributed-ptimizer
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  # Dynamic token batching replaces an infeasible physical micro-batch of
  # 128 sequences. With CP=3, a full 18,432-token sample spans three GPUs.
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
  --calculate-per-token-loss
  --log-probs-chunk-size 1024
  # 加到 PERF_ARGS
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-context-length 18432
  --sglang-server-concurrency 48
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
  --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
  # Colocated mode normally enables both flags; keep them explicit because the
  # split disk-sync lifecycle depends on both sides being offloadable.
  --offload-train
  --offload-rollout
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

cd "${REPO_ROOT}"

# Stop only Ray processes managed by Ray; do not kill unrelated Python jobs.
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
"${RAY_BIN}" start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port "${MASTER_PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON=$(printf \
  '{"env_vars":{"PYTHONPATH":"%s:%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"%s","GLOO_SOCKET_IFNAME":"%s","TP_SOCKET_IFNAME":"%s","MASTER_ADDR":"%s","MASTER_PORT":"%s","RAY_DISABLE_DASHBOARD_GPU_METRICS":"1","SLIME_RELOAD_PROCESS_GROUPS":"0","SGLANG_HEALTH_CHECK_TIMEOUT":"%s","SGLANG_WARMUP_TIMEOUT":"%s","no_proxy":"%s","NO_PROXY":"%s","FLASHINFER_WORKSPACE_BASE":"%s","TRITON_CACHE_DIR":"%s","TORCH_EXTENSIONS_DIR":"%s","XDG_CACHE_HOME":"%s"}}' \
  "${REPO_ROOT}" "${MEGATRON_PATH}" "${HAS_NVLINK}" "${SOCKET_IFNAME}" "${SOCKET_IFNAME}" \
  "${MASTER_ADDR}" "${MASTER_PORT}" "${SGLANG_HEALTH_CHECK_TIMEOUT}" "${SGLANG_WARMUP_TIMEOUT}" "${no_proxy}" "${NO_PROXY}" "${FLASHINFER_WORKSPACE_BASE}" \
  "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}")

"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${REWARD_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WEIGHT_SYNC_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
