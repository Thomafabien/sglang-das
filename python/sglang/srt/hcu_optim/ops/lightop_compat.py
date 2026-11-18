"""lightop 版本兼容：moe_align_block_size_out 的位置变了。

lightop >= 2608xx 把 `moe_align_block_size_out` 从 `lightop.op` 挪到了 `lightop.moe`，
而海光 sglang 的 moe_align_block_size.py 仍按 `op.moe_align_block_size_out(...)` 调用
→ 运行时 AttributeError，MoE 直接跑不起来。

该兼容逻辑现在作为 SGLang 源码内的显式可选扩展交付，无需外置 patch vendor 树。

开关：SGLANG_LIGHTOP_COMPAT=0 可关闭（若将来上游自己修了）。
"""


def install(module):
    op = getattr(module, "op", None)
    if op is None:
        return                                  # 非 HCU 分支，无需处理
    if hasattr(op, "moe_align_block_size_out"):
        print("[lightop_compat] 无需修补：op.moe_align_block_size_out 已存在", flush=True)
        return
    try:
        from lightop import moe as _lightop_moe
    except ImportError:
        print("[lightop_compat] 跳过：lightop.moe 不可导入", flush=True)
        return
    fn = getattr(_lightop_moe, "moe_align_block_size_out", None)
    if fn is None:
        print("[lightop_compat] 跳过：lightop.moe 里也没有该符号", flush=True)
        return
    op.moe_align_block_size_out = fn
    print("[lightop_compat] 已安装：op.moe_align_block_size_out ← lightop.moe", flush=True)
