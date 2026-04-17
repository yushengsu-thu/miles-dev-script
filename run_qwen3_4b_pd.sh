#!/bin/bash
# PD Disaggregation (non-colocate) with Qwen2.5-0.5B-Instruct on 8 GPUs
# Training: 4 GPUs (actor)
# Rollout:  4 GPUs (1 prefill + 3 decode)

ray stop --force 2>/dev/null
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "ray::" 2>/dev/null
pkill -9 -f "train.py" 2>/dev/null
sleep 5
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "ray" 2>/dev/null
sleep 3

set -ex

export PYTHONUNBUFFERED=1

HF_TOKEN="${HF_TOKEN:-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}"

# ---- Download model and data if needed ----
echo "==> Checking model ..."
if [ -f /root/Qwen2.5-0.5B-Instruct/config.json ]; then
    echo "    Model already present, skipping."
else
    echo "    Downloading Qwen/Qwen2.5-0.5B-Instruct ..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
        --local-dir /root/Qwen2.5-0.5B-Instruct --token "${HF_TOKEN}"
fi

echo "==> Checking dataset ..."
if [ -f /root/gsm8k/train.parquet ]; then
    echo "    Dataset already present, skipping."
else
    echo "    Downloading zhuzilin/gsm8k ..."
    huggingface-cli download --repo-type dataset zhuzilin/gsm8k \
        --local-dir /root/gsm8k --token "${HF_TOKEN}"
fi

# ---- Model architecture args (Qwen2.5-0.5B) ----
MILES_DIR="/home/ubuntu/yushengsu/miles"
source "${MILES_DIR}/scripts/models/qwen2.5-0.5B.sh"

cd "${MILES_DIR}"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/
   --ref-load /root/Qwen2.5-0.5B-Instruct/
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 5
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 32
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-dev
   # --wandb-group qwen2.5-0.5B-pd-disaggregated
)

# ---- PD Disaggregation (non-colocate) ----
# 4 rollout GPUs: prefill-num-servers=1 → 1 prefill + 3 decode
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.85
   --prefill-num-servers 1
   --sglang-mooncake-ib-device mlx5_7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

# ---- Launch Ray and submit job ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
