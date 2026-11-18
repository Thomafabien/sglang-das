"""lightop 版本兼容：moe_align_block_size_out 的位置变了。

lightop >= 2608xx 把 `moe_align_block_size_out` 从 `lightop.op` 挪到了 `lightop.moe`，
而海光 sglang 的 moe_align_block_size.py 仍按 `op.moe_align_block_size_out(...)` 调用
→ 运行时 AttributeError，MoE 直接跑不起来。

原先是直接改海光源码解决的（11 行）。这里改成运行时绑定，
保持「不改海光一个字」的性质：clone 本仓库即可，无需 patch vendor 树。

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
