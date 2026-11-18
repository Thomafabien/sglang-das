"""跳过 moe_align_block_size 里为 EP(专家并行) 准备的哨兵映射。

原 HCU 路径（moe_align_block_size.py 的 `if _is_hcu` 分支）为了让 LightOp 收到非负专家 id，
把 -1 经哨兵专家绕一圈再映射回来，额外引入 5 个 kernel：
  ② topk_ids < 0      ③ full_like(topk_ids, N)   ④ where(...)
  ⑥ expert_ids == N   ⑦ masked_fill_(..., -1)
我们 ep_size=1 / moe_a2a_backend=none / enable_eplb=False，topk_ids 永不含负值，
这 5 个算子是纯空转。

实测(hipprof, 不带图, 生产形状)：这 5 个的纯 GPU 时间合计 38.1 µs/层，
每轮 decode 43 层 → 1.64 ms ≈ 一轮 54.5 ms 的 3.0%。

等价性已验证：7 个 M 值 × 3 个 BLOCK_M × 3 次随机，共 21 组，
npad / expert_ids / 每个 block 的元素集合全部一致。
（注意：lightop 的 align kernel **本身非确定性** —— 原路径自己跑两次
 block 内顺序也会变，只有集合不变；而 block 内换序不影响 MoE 结果，
 因为输出按 token id 写回。所以判据是"集合一致"而非"逐位一致"。）
"""
import os, torch, triton

_enabled = os.environ.get("SGLANG_FAST_MOE_ALIGN", "1") != "0"
_guard = None      # None=未判定, True=可用快路径, False=回退

def _check_guard():
    """静态判断 EP 是否关闭。只判一次并缓存；判不了就回退。"""
    try:
        from sglang.srt.server_args import get_global_server_args
        sa = get_global_server_args()
    except Exception:
        try:
            from sglang.srt.server_args import global_server_args_dict as _d
            sa = None
        except Exception:
            return False
    try:
        if sa is not None:
            if getattr(sa, "ep_size", 1) != 1: return False
            if getattr(sa, "enable_eplb", False): return False
            if getattr(sa, "ep_num_redundant_experts", 0) != 0: return False
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        if get_moe_a2a_backend().is_deepep(): return False
        return True
    except Exception:
        return False

def install(mod):
    if getattr(mod, "_fast_moe_align_installed", False) or not _enabled:
        return
    orig = mod.moe_align_block_size
    try:
        from lightop import op
    except Exception as e:
        print(f"[fast_moe_align] lightop 不可用, 保持原实现: {e}", flush=True)
        return

    def fast(topk_ids, block_size, num_experts, ignore_invalid_expert=False):
        global _guard
        if _guard is None:
            _guard = _check_guard()
            print(f"[fast_moe_align] 守卫判定 EP 已关闭 = {_guard}", flush=True)
        if not _guard:
            return orig(topk_ids, block_size, num_experts, ignore_invalid_expert)
        try:
            n = topk_ids.numel()
            mx = n * block_size if n < num_experts + 1 else n + (num_experts + 1) * (block_size - 1)
            dev = topk_ids.device
            sorted_ids = torch.empty((mx,), dtype=torch.int32, device=dev)
            expert_ids = torch.empty((triton.cdiv(mx, block_size),), dtype=torch.int32, device=dev)
            npad = torch.empty((1,), dtype=torch.int32, device=dev)
            sorted_ids.fill_(n)
            ti = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
            op.moe_align_block_size(ti, num_experts + 1, block_size, sorted_ids,
                                    expert_ids, npad, None, None, None, False, True)
            return sorted_ids, expert_ids, npad
        except Exception as e:
            print(f"[fast_moe_align] 快路径异常, 本次回退: {type(e).__name__}: {e}", flush=True)
            return orig(topk_ids, block_size, num_experts, ignore_invalid_expert)

    mod.moe_align_block_size = fast
    mod._fast_moe_align_installed = True
    print(f"[fast_moe_align] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)
