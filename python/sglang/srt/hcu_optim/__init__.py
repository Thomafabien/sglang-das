"""Source-integrated HCU optimizations for the Qwen3.6 K100_AI profile.

Every optimization is connected explicitly from its owning SGLang module.  No
meta-path loader or ``sitecustomize`` hook is used.  All top-level feature flags
default to disabled so an ordinary SGLang launch keeps the upstream code path.
"""
import os

from sglang.srt.environ import envs

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = envs.SGLANG_HCU_KERNEL_LIB_DIR.get()
if PATCH_DIR is None:
    PATCH_DIR = os.path.join(PACKAGE_DIR, "lib")
