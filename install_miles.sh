#!/bin/bash
set -euo pipefail

MILES_DIR="/home/ubuntu/yushengsu/miles"

cd "$MILES_DIR"
pip install -e . --no-deps

echo "Miles installed successfully (editable mode)."
