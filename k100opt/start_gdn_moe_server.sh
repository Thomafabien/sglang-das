#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/model/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30010}"
GPU_DEVICES="${GPU_DEVICES:-4,5,6,7}"

export HIP_VISIBLE_DEVICES="$GPU_DEVICES"
export LD_LIBRARY_PATH="/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/k100opt/patch:$REPO_ROOT/python"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"

# Conv1D + GDN source integrations.
export SGLANG_USE_AITER_LINEAR_ATTN="${SGLANG_USE_AITER_LINEAR_ATTN:-1}"
export SGLANG_CONV1D_CFG="${SGLANG_CONV1D_CFG:-1}"
export SGLANG_CHUNK_H_CFG="${SGLANG_CHUNK_H_CFG:-1}"
export SGLANG_GDN_NOZERO="${SGLANG_GDN_NOZERO:-1}"
export SGLANG_GDN_CFG="${SGLANG_GDN_CFG:-1}"

# MoE runtime integrations.  sitecustomize.py keeps them disabled unless these
# switches are explicitly enabled.
export SGLANG_MOE_CONFIG_DIR="${SGLANG_MOE_CONFIG_DIR:-$REPO_ROOT/k100opt/config/moe_config}"
export SGLANG_LIGHTOP_COMPAT="${SGLANG_LIGHTOP_COMPAT:-1}"
export SGLANG_FAST_MOE_ALIGN="${SGLANG_FAST_MOE_ALIGN:-1}"
export SGLANG_ACT_FLAT="${SGLANG_ACT_FLAT:-1}"
export SGLANG_ACT_FLAT_MIN="${SGLANG_ACT_FLAT_MIN:-20000}"
export SGLANG_MOE_PACK_W13="${SGLANG_MOE_PACK_W13:-1}"

exec python3 -u -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp 4 \
  --disable-radix-cache \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --page-size 64 \
  --numa-node 0 1 2 3 \
  --host "$HOST" \
  --port "$PORT" \
  "$@"

