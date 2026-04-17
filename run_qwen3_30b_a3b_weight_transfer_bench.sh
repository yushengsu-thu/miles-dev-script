#!/bin/bash
#
# Qwen3-30B-A3B weight transfer mode benchmark
#
# Compare broadcast vs p2p (with and without --delta-weight-update)
# on a single 8-GPU node using colocated actor (4 GPU) + rollout (4 GPU).
#
# Usage:
#   # Broadcast (baseline)
#   bash run_qwen3_30b_a3b_weight_transfer_bench.sh broadcast
#
#   # Broadcast + delta
#   bash run_qwen3_30b_a3b_weight_transfer_bench.sh broadcast --delta
#
#   # P2P RDMA (baseline)
#   bash run_qwen3_30b_a3b_weight_transfer_bench.sh p2p
#
#   # P2P RDMA + delta
#   bash run_qwen3_30b_a3b_weight_transfer_bench.sh p2p --delta
#
#   # Colocated (baseline, no separate transfer mode)
#   bash run_qwen3_30b_a3b_weight_transfer_bench.sh colocate
#
#   # Colocated + delta
#   bash run_qwen3_30b_a3b_weight_transfer_bench.sh colocate --delta

ray stop --force 2>/dev/null
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "ray::" 2>/dev/null
pkill -9 -f "train.py" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "ray" 2>/dev/null
sleep 3

set -euo pipefail

# ── parse args ──────────────────────────────────────────────────────
MODE="${1:-broadcast}"
DELTA_FLAG=""
if [[ "${2:-}" == "--delta" ]]; then
    DELTA_FLAG="--delta-weight-update"
    echo ">>> Delta weight update ENABLED"
fi

echo ">>> Weight transfer mode: ${MODE}"

# ── cleanup stale processes & old Ray sessions ────────────────────
pkill -9 sglang  2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray     2>/dev/null || true
pkill -9 python  2>/dev/null || true
sleep 2
rm -rf /tmp/ray/session_* 2>/dev/null || true
rm -rf /dev/shm/ray_tmp/session_* 2>/dev/null || true

export PYTHONUNBUFFERED=1

# ── detect NVLink ───────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$( (( NVLINK_COUNT > 0 )) && echo 1 || echo 0 )
echo ">>> NVLink detected: ${HAS_NVLINK} (${NVLINK_COUNT} links)"

# ── model config (Qwen3-30B-A3B MoE) ───────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/miles/scripts/models/qwen3-30B-A3B.sh"

# ── paths — adjust these to your local setup ────────────────────────
BASE_DIR="/root"
HF_CKPT="${BASE_DIR}/Qwen3-30B-A3B"              # or Qwen3-30B-A3B-FP8 / -INT4
REF_LOAD="${BASE_DIR}/Qwen3-30B-A3B_torch_dist"
SAVE_DIR="/dev/shm/Qwen3-30B-A3B_miles_bench"
PROMPT_DATA="${BASE_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${REF_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 999
)

# ── rollout — small batch for benchmarking weight transfer ──────────
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 5
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --rollout-max-response-len 512
    --rollout-temperature 1
    --global-batch-size 32
    --balance-data
)

# ── training parallelism (actor on 4 GPUs) ──────────────────────────
PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
)

# ── GRPO ────────────────────────────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

# ── optimizer ───────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

# ── misc ────────────────────────────────────────────────────────────
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --no-check-for-nan-in-loss-and-grad
)

# ── mode-specific config ────────────────────────────────────────────
TRANSFER_ARGS=()
ACTOR_ARGS=()
SGLANG_ARGS=()

case "${MODE}" in
    colocate)
        # Actor + rollout share the same 8 GPUs
        ACTOR_ARGS=(
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate
        )
        SGLANG_ARGS=(
            --rollout-num-gpus-per-engine 8
            --sglang-mem-fraction-static 0.5
            --sglang-ep-size 4
            --sglang-disable-cuda-graph
            --sglang-moe-runner-backend triton
        )
        ;;

    broadcast)
        # Actor 4 GPU, Rollout 4 GPU (non-colocated → uses broadcast)
        ACTOR_ARGS=(
            --actor-num-nodes 1
            --actor-num-gpus-per-node 4
        )
        SGLANG_ARGS=(
            --rollout-num-gpus 4
            --rollout-num-gpus-per-engine 4
            --sglang-mem-fraction-static 0.7
            --sglang-ep-size 4
            --sglang-moe-runner-backend triton
        )
        TRANSFER_ARGS=(
            --update-weight-transfer-mode broadcast
            --update-weight-buffer-size $(( 1 * 1024 * 1024 * 1024 ))
        )
        ;;

    p2p)
        # Actor 4 GPU, Rollout 4 GPU (non-colocated → uses P2P RDMA)
        ACTOR_ARGS=(
            --actor-num-nodes 1
            --actor-num-gpus-per-node 4
        )
        SGLANG_ARGS=(
            --rollout-num-gpus 4
            --rollout-num-gpus-per-engine 4
            --sglang-mem-fraction-static 0.7
            --sglang-ep-size 4
            --sglang-moe-runner-backend triton
        )
        TRANSFER_ARGS=(
            --update-weight-transfer-mode p2p
            --update-weight-buffer-size $(( 1 * 1024 * 1024 * 1024 ))
        )
        ;;

    *)
        echo "ERROR: Unknown mode '${MODE}'. Use: colocate | broadcast | p2p"
        exit 1
        ;;
esac

# ── launch ray (keep all temp on /dev/shm to avoid root disk usage) ─
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_TMPDIR="/dev/shm/ray_tmp"
mkdir -p "${RAY_TMPDIR}"
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/dev/shm/ray_spill"}}'
mkdir -p /dev/shm/ray_spill
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir="${RAY_TMPDIR}" \
    --object-store-memory=$(( 20 * 1024 * 1024 * 1024 ))

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_TIMEOUT_MS\": \"36000000\"
  }
}"

# ── submit ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Qwen3-30B-A3B  weight transfer benchmark"
echo "  Mode:  ${MODE}"
echo "  Delta: ${DELTA_FLAG:-disabled}"
echo "============================================================"
echo ""

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 /home/ubuntu/yushengsu/miles/train.py \
    ${ACTOR_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${TRANSFER_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${DELTA_FLAG}
