#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/model/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30010}"
GPU_DEVICES="${GPU_DEVICES:-4,5,6,7}"

export HIP_VISIBLE_DEVICES="$GPU_DEVICES"
export LD_LIBRARY_PATH="/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export SGLANG_HCU_KERNEL_LIB_DIR="${SGLANG_HCU_KERNEL_LIB_DIR:-$REPO_ROOT/python/sglang/srt/hcu_optim/lib}"
export SGLANG_MOE_CONFIG_DIR="${SGLANG_MOE_CONFIG_DIR:-$REPO_ROOT/k100opt/config/moe_config}"

# Direct Conv1D and GDN source paths.
export SGLANG_USE_AITER_LINEAR_ATTN="${SGLANG_USE_AITER_LINEAR_ATTN:-1}"
export SGLANG_CONV1D_CFG="${SGLANG_CONV1D_CFG:-1}"
export SGLANG_CHUNK_H_CFG="${SGLANG_CHUNK_H_CFG:-1}"
export SGLANG_GDN_NOZERO="${SGLANG_GDN_NOZERO:-1}"
export SGLANG_GDN_CFG="${SGLANG_GDN_CFG:-1}"

# MoE operator paths.
export SGLANG_LIGHTOP_COMPAT="${SGLANG_LIGHTOP_COMPAT:-1}"
export SGLANG_FAST_MOE_ALIGN="${SGLANG_FAST_MOE_ALIGN:-1}"
export SGLANG_ACT_FLAT="${SGLANG_ACT_FLAT:-1}"
export SGLANG_ACT_FLAT_MIN="${SGLANG_ACT_FLAT_MIN:-20000}"
export SGLANG_MOE_PACK_W13="${SGLANG_MOE_PACK_W13:-1}"
export SGLANG_DENSE_PACK="${SGLANG_DENSE_PACK:-1}"
export SGLANG_M3_FUSE="${SGLANG_M3_FUSE:-1}"

# KV-cache, speculative decoding, scheduling, communication and infrastructure.
export SGLANG_FP8_KV_HCU="${SGLANG_FP8_KV_HCU:-1}"
export SGLANG_FUSED_DRAFT_TOPK="${SGLANG_FUSED_DRAFT_TOPK:-1}"
export SGLANG_VPA_DRAFT="${SGLANG_VPA_DRAFT:-1}"
export SGLANG_VPA_VERIFY="${SGLANG_VPA_VERIFY:-1}"
export SGLANG_WARM_BATCH="${SGLANG_WARM_BATCH:-1}"
export SGLANG_ADAPTIVE_PMR="${SGLANG_ADAPTIVE_PMR:-1}"
export SGLANG_LONG_ADMIT="${SGLANG_LONG_ADMIT:-1}"
export SGLANG_FIRST_TOKEN_FIRST="${SGLANG_FIRST_TOKEN_FIRST:-3}"
export SGLANG_ADAPTIVE_CHUNK="${SGLANG_ADAPTIVE_CHUNK:-1}"
export SGLANG_TOK_POOL="${SGLANG_TOK_POOL:-1}"
export SGLANG_KV_WRITE_FUSED="${SGLANG_KV_WRITE_FUSED:-1}"
export SGLANG_MTP_ROPE_TRITON="${SGLANG_MTP_ROPE_TRITON:-1}"
export SGLANG_QUANT_AR="${SGLANG_QUANT_AR:-1}"
export SGLANG_SERVER_DEFAULTS="${SGLANG_SERVER_DEFAULTS:-1}"
export SGLANG_FAST_IMPORT="${SGLANG_FAST_IMPORT:-1}"

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
