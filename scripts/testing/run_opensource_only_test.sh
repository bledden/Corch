#!/bin/bash
# Override config models with OPEN SOURCE ONLY
export OVERRIDE_ARCHITECT_MODEL="qwen/qwen3-coder-plus"
export OVERRIDE_CODER_MODEL="deepseek/deepseek-v3"
export OVERRIDE_REVIEWER_MODEL="meta-llama/llama-3.3-70b-instruct"
export OVERRIDE_DOCUMENTER_MODEL="mistralai/mistral-large-2411"
export SMOKE_TEST_NAME="opensource-only"

python3 run_smoke_test.py
