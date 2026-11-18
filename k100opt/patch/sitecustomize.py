"""Install only the explicitly enabled K100_AI MoE patches.

Add this directory before ``python/`` on ``PYTHONPATH``.  Every patch defaults
to disabled so importing this branch without the K100 profile preserves the
upstream behavior.
"""

import importlib
import importlib.abc
import os
import sys


TARGETS = {
    "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size": [
        ("kpatch.ops.lightop_compat", "SGLANG_LIGHTOP_COMPAT"),
    ],
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe": [
        ("kpatch.ops.moe_pack_w13", "SGLANG_MOE_PACK_W13"),
        ("kpatch.ops.fast_moe_align", "SGLANG_FAST_MOE_ALIGN"),
        ("kpatch.ops.act_flat", "SGLANG_ACT_FLAT"),
    ],
    "sglang.srt.layers.quantization.unquant": [
        ("kpatch.ops.moe_pack_w13", "SGLANG_MOE_PACK_W13"),
    ],
}


def _enabled(name):
    return os.environ.get(name, "0").strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, real_loader, patches):
        self.real_loader = real_loader
        self.patches = patches

    def create_module(self, spec):
        return self.real_loader.create_module(spec)

    def exec_module(self, module):
        self.real_loader.exec_module(module)
        for patch_module, env_name in self.patches:
            if not _enabled(env_name):
                continue
            try:
                importlib.import_module(patch_module).install(module)
            except Exception as exc:
                print(
                    f"[{patch_module.rsplit('.', 1)[-1]}] install failed; "
                    f"using original implementation: {type(exc).__name__}: {exc}",
                    flush=True,
                )


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        patches = TARGETS.get(fullname)
        if not patches:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader, patches)
                return spec
        return None


if not any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _PatchFinder())

