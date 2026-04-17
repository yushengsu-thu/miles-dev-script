#!/bin/bash
#
# Quick delta weight update test with Qwen2.5-0.5B (2 RL steps).
# Tests colocate mode with and without --delta-weight-update.
#
# Usage (run inside Docker container):
#   bash /home/ubuntu/yushengsu/run_delta_test_0.5b.sh

set -euo pipefail

MILES_DIR="/home/ubuntu/yushengsu/miles"
cd "${MILES_DIR}"

LOG="/home/ubuntu/yushengsu/delta_test_0.5b.log"
> "$LOG"
exec > >(tee -a "$LOG") 2>&1

# ── cleanup ─────────────────────────────────────────────────────────
cleanup() {
    pkill -9 -f sglang  2>/dev/null || true
    ray stop --force     2>/dev/null || true
    pkill -9 -f "ray::"  2>/dev/null || true
    pkill -9 -f train.py 2>/dev/null || true
    rm -rf /tmp/ray      2>/dev/null || true
    sleep 3
}

cleanup

# ── model config ────────────────────────────────────────────────────
source "${MILES_DIR}/scripts/models/qwen2.5-0.5B.sh"

HF_CKPT="/root/Qwen2.5-0.5B-Instruct"
DATA="/root/gsm8k/train.parquet"

# ── shared args ─────────────────────────────────────────────────────
COMMON_ARGS=(
    ${MODEL_ARGS[@]}
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --megatron-to-hf-mode bridge

    --prompt-data "${DATA}"
    --input-key messages
    --label-key label
    --apply-chat-template
    --rm-type math
    --num-rollout 2
    --rollout-batch-size 4
    --n-samples-per-prompt 4
    --rollout-max-response-len 256
    --rollout-temperature 0.8
    --global-batch-size 16

    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096

    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28

    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.6

    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --colocate
)

ENV_JSON='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}'

# ── run one benchmark ───────────────────────────────────────────────
run_one() {
    local LABEL=$1
    shift
    local EXTRA_ARGS=("$@")

    echo ""
    echo "============================================================"
    echo "  RUN: ${LABEL}   ($(date))"
    echo "============================================================"

    cleanup

    ray start --head --num-gpus 4 --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port=8265 2>&1 | tail -3

    local T_START=${SECONDS}

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${ENV_JSON}" \
        -- python3 "${MILES_DIR}/train.py" \
        ${COMMON_ARGS[@]} \
        ${EXTRA_ARGS[@]} \
        2>&1

    local T_END=${SECONDS}
    local ELAPSED=$(( T_END - T_START ))
    echo ""
    echo ">>> ${LABEL}: wall time = ${ELAPSED}s"
    echo "============================================================"
}

# ── Run 1: baseline (no delta) ──────────────────────────────────────
run_one "colocate_baseline"

# ── Run 2: with delta ───────────────────────────────────────────────
run_one "colocate_delta" --delta-weight-update

echo ""
echo "============================================================"
echo "  ALL RUNS COMPLETE: $(date)"
echo "  Log: ${LOG}"
echo "============================================================"
