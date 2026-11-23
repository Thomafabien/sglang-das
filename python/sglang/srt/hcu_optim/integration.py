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
    "sglang.srt.layers.logits_processor": (
        ("sglang.srt.hcu_optim.ops.vocab_par_draft", "SGLANG_VPA_DRAFT"),
        ("sglang.srt.hcu_optim.ops.vocab_par_verify", "SGLANG_VPA_VERIFY"),
    ),
    "sglang.srt.models.qwen3_5_mtp": (
        ("sglang.srt.hcu_optim.ops.vocab_par_draft", "SGLANG_VPA_DRAFT"),
    ),
    "sglang.srt.managers.scheduler": (
        ("sglang.srt.hcu_optim.sched.first_token_first", "SGLANG_FIRST_TOKEN_FIRST"),
        ("sglang.srt.hcu_optim.sched.adaptive_chunk", "SGLANG_ADAPTIVE_CHUNK"),
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
