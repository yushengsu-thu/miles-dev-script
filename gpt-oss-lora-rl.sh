#!/bin/bash
ray stop --force
pkill -9 -f ray
pkill -9 -f sglang
pkill -9 -f python
sleep 3

set -e

HF_TOKEN="${HF_TOKEN:-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}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"

echo "============================================"
echo " Miles LoRA RL Training: GPT-OSS-20B MoE"
echo "============================================"

echo "==> Checking model ..."
if [ -f /root/models/gpt-oss-20b/config.json ]; then
    echo "    Model already present, skipping."
else
    echo "    Downloading gpt-oss-20b ..."
    hf download lmsys/gpt-oss-20b-bf16 \
        --local-dir /root/models/gpt-oss-20b --token "${HF_TOKEN}"
fi

echo "==> Checking training dataset (dapo-math-17k) ..."
if [ -f /root/dapo-math-17k/dapo-math-17k.jsonl ]; then
    echo "    Dataset already present, skipping."
else
    echo "    Downloading zhuzilin/dapo-math-17k ..."
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
fi

echo "==> Checking eval dataset (gsm8k) ..."
if [ -f /root/gsm8k/test.parquet ]; then
    echo "    Dataset already present, skipping."
else
    echo "    Downloading zhuzilin/gsm8k ..."
    hf download --repo-type dataset zhuzilin/gsm8k \
        --local-dir /root/gsm8k --token "${HF_TOKEN}"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set — switching W&B to offline mode."
    export WANDB_MODE="offline"
else
    export WANDB_API_KEY
fi

echo "==> Starting GPT-OSS-20B MoE LoRA RL training ..."
MILES_DIR=/home/ubuntu/yushengsu/miles
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_TOKEN
export PYTHONUNBUFFERED=1

cd "${MILES_DIR}"
ulimit -n 1048576
bash "${MILES_DIR}/examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh"
