#!/bin/bash
set -euo pipefail

LOG="/home/ubuntu/yushengsu/bench_log.txt"
> "$LOG"
exec > >(tee -a "$LOG") 2>&1

MILES_DIR="/home/ubuntu/yushengsu/miles"
MEGATRON_PATH="/root/Megatron-LM"

echo "============================================================"
echo "  Step 1: Download Qwen3-30B-A3B + dapo-math-17k"
echo "  Started: $(date)"
echo "============================================================"

# Download model
if [ ! -f /root/Qwen3-30B-A3B/config.json ]; then
    echo ">>> Downloading Qwen/Qwen3-30B-A3B ..."
    hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B
else
    echo ">>> Qwen3-30B-A3B already exists, skipping download"
fi

# Download dataset
if [ ! -f /root/dapo-math-17k/dapo-math-17k.jsonl ]; then
    echo ">>> Downloading dapo-math-17k ..."
    python3 -c "
from datasets import load_dataset
import json, os
ds = load_dataset('zhuzilin/dapo-math-17k', split='train')
os.makedirs('/root/dapo-math-17k', exist_ok=True)
with open('/root/dapo-math-17k/dapo-math-17k.jsonl', 'w') as f:
    for row in ds:
        f.write(json.dumps(row) + '\n')
print(f'Saved {len(ds)} examples')
"
else
    echo ">>> dapo-math-17k already exists, skipping download"
fi
echo ">>> Downloads complete: $(date)"

echo ""
echo "============================================================"
echo "  Step 2: Convert Megatron checkpoint (8 GPU)"
echo "  Started: $(date)"
echo "============================================================"

if [ ! -f /root/Qwen3-30B-A3B_torch_dist/latest_checkpointed_iteration.txt ]; then
    cd "${MILES_DIR}"
    source "${MILES_DIR}/scripts/models/qwen3-30B-A3B.sh"
    PYTHONPATH="${MILES_DIR}:${MEGATRON_PATH}" torchrun --nproc-per-node 8 \
        "${MILES_DIR}/tools/convert_hf_to_torch_dist.py" \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint /root/Qwen3-30B-A3B \
        --save /root/Qwen3-30B-A3B_torch_dist
    echo ">>> Conversion complete: $(date)"
else
    echo ">>> Megatron checkpoint already exists, skipping conversion"
fi

echo ""
echo "============================================================"
echo "  Step 3: Run 6 benchmarks (5 RL steps each, fixed seed)"
echo "  Started: $(date)"
echo "============================================================"

cd /home/ubuntu/yushengsu

BENCH_RESULTS="/home/ubuntu/yushengsu/bench_results.txt"
> "$BENCH_RESULTS"

run_one() {
    local MODE=$1
    local DELTA_ARG="${2:-}"
    local LABEL="${MODE}"
    if [[ -n "$DELTA_ARG" ]]; then
        LABEL="${MODE}+delta"
    fi

    echo "" | tee -a "$BENCH_RESULTS"
    echo "------------------------------------------------------------" | tee -a "$BENCH_RESULTS"
    echo "  Benchmark: ${LABEL}   ($(date))" | tee -a "$BENCH_RESULTS"
    echo "------------------------------------------------------------" | tee -a "$BENCH_RESULTS"

    # cleanup from previous run
    pkill -9 sglang  2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    sleep 1
    pkill -9 ray     2>/dev/null || true
    pkill -9 python  2>/dev/null || true
    sleep 3

    local START_SEC=$SECONDS

    # run the benchmark and capture output
    bash /home/ubuntu/yushengsu/run_qwen3_30b_a3b_weight_transfer_bench.sh ${MODE} ${DELTA_ARG} 2>&1 || true

    local ELAPSED=$(( SECONDS - START_SEC ))
    echo "  >>> ${LABEL}: total wall time = ${ELAPSED}s" | tee -a "$BENCH_RESULTS"
}

# All 6 configurations
run_one colocate
run_one colocate "--delta"
run_one broadcast
run_one broadcast "--delta"
run_one p2p
run_one p2p "--delta"

echo ""
echo "============================================================" | tee -a "$BENCH_RESULTS"
echo "  ALL BENCHMARKS COMPLETE: $(date)" | tee -a "$BENCH_RESULTS"
echo "  Full log: ${LOG}" | tee -a "$BENCH_RESULTS"
echo "  Results summary: ${BENCH_RESULTS}" | tee -a "$BENCH_RESULTS"
echo "============================================================" | tee -a "$BENCH_RESULTS"
