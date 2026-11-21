"""Explicit source integration points for K100_AI optimizations.

Target SGLang modules call :func:`apply_for_module` after defining their normal
implementation.  This is deliberately different from the legacy delivery
overlay: there is no meta-path finder and no automatic interception of imports.
If a feature flag is absent or false, its optimization module is not imported
and the original SGLang implementation remains untouched.
"""

from __future__ import annotations

import importlib
import logging
import os
from types import ModuleType

logger = logging.getLogger(__name__)


# Target module -> [(implementation module, top-level environment switch)].
# Every switch is opt-in.  Secondary tuning variables are documented in the
# implementation module and in k100opt/SOURCE_INTEGRATION.md.
TARGETS = {
    "sglang.srt.arg_groups.overrides": (
        ("sglang.srt.hcu_optim.kv.fp8_kv_hcu", "SGLANG_FP8_KV_HCU"),
    ),
    "sglang.srt.layers.attention.flashattention_interface": (
        ("sglang.srt.hcu_optim.kv.fp8_kv_hcu", "SGLANG_FP8_KV_HCU"),
    ),
    "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size": (
        ("sglang.srt.hcu_optim.ops.lightop_compat", "SGLANG_LIGHTOP_COMPAT"),
    ),
    "sglang.srt.speculative.eagle_worker_v2": (
        ("sglang.srt.hcu_optim.ops.vocab_par_verify", "SGLANG_VPA_VERIFY"),
    ),
    "sglang.srt.layers.logits_processor": (
        ("sglang.srt.hcu_optim.ops.vocab_par_draft", "SGLANG_VPA_DRAFT"),
        ("sglang.srt.hcu_optim.ops.vocab_par_verify", "SGLANG_VPA_VERIFY"),
    ),
    "sglang.srt.models.qwen3_5_mtp": (
        ("sglang.srt.hcu_optim.ops.vocab_par_draft", "SGLANG_VPA_DRAFT"),
    ),
    "sglang.srt.entrypoints.warmup": (
        ("sglang.srt.hcu_optim.infra.warm_batch", "SGLANG_WARM_BATCH"),
    ),
    "sglang.srt.managers.schedule_policy": (
        ("sglang.srt.hcu_optim.sched.adaptive_pmr", "SGLANG_ADAPTIVE_PMR"),
        ("sglang.srt.hcu_optim.sched.long_admit", "SGLANG_LONG_ADMIT"),
    ),
    "sglang.srt.server_args": (
        ("sglang.srt.hcu_optim.infra.server_defaults", "SGLANG_SERVER_DEFAULTS"),
        ("sglang.srt.hcu_optim.kv.fp8_kv_hcu", "SGLANG_FP8_KV_HCU"),
    ),
    "sglang.srt.managers.scheduler": (
        ("sglang.srt.hcu_optim.sched.first_token_first", "SGLANG_FIRST_TOKEN_FIRST"),
        ("sglang.srt.hcu_optim.sched.adaptive_chunk", "SGLANG_ADAPTIVE_CHUNK"),
    ),
    "sglang.srt.managers.tokenizer_manager": (
        ("sglang.srt.hcu_optim.infra.tok_pool", "SGLANG_TOK_POOL"),
    ),
    "sglang.srt.distributed.parallel_state": (
        ("sglang.srt.hcu_optim.comm.quant_ar", "SGLANG_QUANT_AR"),
    ),
}


def is_enabled(name: str) -> bool:
    """Return true only when the user explicitly enables a feature."""

    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
        "none",
        "null",
    }


def apply_for_module(module: ModuleType) -> None:
    """Apply explicitly enabled optimizations owned by ``module``."""

    for implementation, env_name in TARGETS.get(module.__name__, ()):
        if not is_enabled(env_name):
            continue
        try:
            importlib.import_module(implementation).install(module)
        except Exception:
            logger.exception(
                "Failed to enable %s from %s; original path remains active",
                env_name,
                implementation,
            )


def apply_fast_import() -> None:
    """Integrate the optional vllm_hcu import fast path.

    ``vllm_hcu`` is an external dependency, so its target module cannot be
    edited in this repository.  SGLang invokes the integration explicitly from
    its package initialization instead of intercepting imports globally.
    """

    if not is_enabled("SGLANG_FAST_IMPORT"):
        return
    try:
        target = importlib.import_module("vllm_hcu.patch_utils")
        importlib.import_module("sglang.srt.hcu_optim.infra.fast_import").install(
            target
        )
    except Exception:
        logger.exception(
            "Failed to enable SGLANG_FAST_IMPORT; original import hook remains active"
        )
