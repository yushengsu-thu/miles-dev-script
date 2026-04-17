#!/bin/bash
ray stop --force
pkill -9 -f ray
pkill -9 -f sglang

set -e

HF_TOKEN="${HF_TOKEN:-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}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"

echo "============================================"
echo " Miles LoRA Training: Qwen2.5-0.5B + GSM8K"
echo "============================================"

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

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set — switching W&B to offline mode."
    echo "         To enable cloud logging, set: export WANDB_API_KEY=<your-key>"
    export WANDB_MODE="offline"
else
    export WANDB_API_KEY
fi

echo "==> Starting LoRA training ..."
MILES_DIR=/home/ubuntu/yushengsu/miles
export HF_TOKEN
export PYTHONUNBUFFERED=1

cd "${MILES_DIR}"
ulimit -n 1048576
bash "${MILES_DIR}/examples/lora/run-qwen2.5-0.5B-megatron-lora.sh"
