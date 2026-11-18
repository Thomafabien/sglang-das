"""M3：把 MoE 的 `_moe_sum_reduce` 与 `_fused_gate_sigmoid_mul_add` 合成一个核。

现状（qwen2_moe.py:540-595 + fused_moe.py:1101-1116）：
  ① experts() 内部 moe_sum_reduce：读 [T,topk,H] → 乘 routed_scaling → **按 bf16 写出 [T,H]**
  ② 模型层 fused_gate_sigmoid_mul_add：**再把 [T,H] 读回**，加 sigmoid(hidden·gate_w)×shared，写回
合并后一遍完成：省掉"写出+读回"两趟 [T,H]，且中间不再经 bf16 落盘（精度反而更好）。

离线实测（单卡，T=8192/4288/1795/563，`m3/m3_bench.py`）：
  1.27× / 1.28× / 1.41× / 2.01×；每 8192 块省 5.53 ms（40 层）；decode 每轮 0.12–0.17 ms。
  对 fp32 真值相对误差 **2.5e-3（合并） vs 5.0e-3（原路径）** —— 原路径的 5e-3 正是那次 bf16 落盘舍入。

实现方式：不改 vendor 源码。
  - 把 fused_moe 模块的 moe_sum_reduce_triton / moe_sum_reduce_torch_compile 换成"只记账不计算"；
  - 把 qwen2_moe 模块属性 fused_gate_sigmoid_mul_add 换成"用合并核一次算完"（必须改消费者模块属性，
    因为它是 `from ... import` 绑定的 —— 见测试准则陷阱清单）；
  - 只在"确定随后会调门控"的条件下记账；万一没被消费，**立刻补做并告警**，不静默留坑。
开关：SGLANG_M3_FUSE=0(默认关) / 1 / compare（两条路都算并比对，输出仍用合并结果）。
"""
import os, torch, triton, triton.language as tl

_MODE = os.environ.get("SGLANG_M3_FUSE", "0")
_EVERY = int(os.environ.get("SGLANG_M3_FUSE_EVERY", "200"))
_st = {"pending": None, "armed": False, "n": 0, "maxdiff": 0.0, "off": False}


@triton.jit
def _sum_gate_kernel(exp_ptr, es0, es1, hid_ptr, w_ptr, sh_ptr, out_ptr, os0,
                     hidden_dim: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr,
                     scale):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < hidden_dim
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    base = exp_ptr + t * tl.cast(es0, tl.int64) + offs
    for k in tl.static_range(TOPK):
        acc += tl.load(base + k * tl.cast(es1, tl.int64), mask=m, other=0.0).to(tl.float32)
    acc *= scale
    h = tl.load(hid_ptr + t * hidden_dim + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(h * w, axis=0))
    s = tl.load(sh_ptr + t * hidden_dim + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * tl.cast(os0, tl.int64) + offs,
             (acc + gate * s).to(out_ptr.dtype.element_ty), mask=m)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def install(mod):
    if _MODE not in ("1", "compare") or getattr(mod, "_m3_installed", False):
        return
    # 前置条件：非 deepep / 非 amx / 非 npu（否则 qwen2_moe 走别的分支，记账可能不被消费）
    try:
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        if get_moe_a2a_backend().is_deepep():
            print("[m3_fuse] deepep 后端，不安装", flush=True)
            return
    except Exception:
        pass
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as FM
    o_tri = getattr(FM, "moe_sum_reduce_triton", None)
    o_tc = getattr(FM, "moe_sum_reduce_torch_compile", None)
    o_gate = getattr(mod, "fused_gate_sigmoid_mul_add", None)
    if o_tri is None or o_gate is None:
        print("[m3_fuse] 未找到目标函数，不安装", flush=True)
        return

    def _defer(orig):
        def f(inp, out, scale):
            if _st["off"] or not _st["armed"]:
                return orig(inp, out, scale)
            _st["pending"] = (inp, out, float(scale), orig)
            return None
        return f

    FM.moe_sum_reduce_triton = _defer(o_tri)
    if o_tc is not None:
        FM.moe_sum_reduce_torch_compile = _defer(o_tc)

    def _fused(inp, hidden, w, shared, out, scale):
        T, H = out.shape
        TOPK = inp.shape[1]
        _sum_gate_kernel[(T,)](inp, inp.stride(0), inp.stride(1), hidden, w, shared,
                               out, out.stride(0), hidden_dim=H, TOPK=TOPK,
                               BLOCK=_next_pow2(H), scale=scale, num_warps=16)

    def gate(hidden_states, gate_weight, shared_output, final_hidden_states):
        p = _st["pending"]
        _st["pending"] = None
        why = None
        if _st["off"]:
            why = "补丁已自关"
        elif p is None:
            why = "没有记账（reduce 未走被包装的那条路）"
        else:
            # 身份用 data_ptr 比，experts() 可能返回 view，不是同一个 Python 对象
            if p[1].data_ptr() != final_hidden_states.data_ptr():
                why = f"输出不是同一块内存 pend={hex(p[1].data_ptr())} gate={hex(final_hidden_states.data_ptr())} 形状 {tuple(p[1].shape)} vs {tuple(final_hidden_states.shape)}"
            elif not (p[0].is_contiguous() and hidden_states.is_contiguous()
                      and shared_output.is_contiguous() and final_hidden_states.is_contiguous()
                      and gate_weight.is_contiguous()):
                why = ("张量非连续 inp=%s hid=%s sh=%s out=%s w=%s" % (
                    p[0].is_contiguous(), hidden_states.is_contiguous(), shared_output.is_contiguous(),
                    final_hidden_states.is_contiguous(), gate_weight.is_contiguous()))
            elif p[0].dim() != 3 or p[0].shape[0] != final_hidden_states.shape[0] or p[0].shape[2] != final_hidden_states.shape[1]:
                why = f"形状不符 inp={tuple(p[0].shape)} out={tuple(final_hidden_states.shape)}"
        if why is not None:
            if not _st.get("why_logged"):
                _st["why_logged"] = True
                print(f"[m3_fuse] ⚠ 首次回退原因：{why}（后续同因不再打印）", flush=True)
            if p is not None:
                p[3](p[0], p[1], p[2])          # 补做 reduce，保证语义
            return o_gate(hidden_states, gate_weight, shared_output, final_hidden_states)
        inp, out, scale, orig_reduce = p
        # ⚠ CUDA graph 捕获期间禁止任何主机同步（.item()/float()），否则
        #   HIP error: operation not permitted when stream is capturing。
        #   所以 compare 只在非捕获时比对，捕获时只走融合。
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = False
        if _MODE == "compare" and not capturing:
            if not _st.get("hit_logged"):
                _st["hit_logged"] = True
                print(f"[m3_fuse] ✅ 首次走融合路径(compare) T={out.shape[0]} topk={inp.shape[1]} H={out.shape[1]}", flush=True)
            ref = torch.empty_like(out)
            orig_reduce(inp, ref, scale)
            o_gate(hidden_states, gate_weight, shared_output, ref)
            _fused(inp, hidden_states, gate_weight, shared_output, out, scale)
            d = float((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9))
            _st["n"] += 1
            _st["maxdiff"] = max(_st["maxdiff"], d)
            if _st["n"] % _EVERY == 0:
                print(f"[m3_fuse] compare n={_st['n']} T={out.shape[0]} 相对差最大 {_st['maxdiff']:.3e}", flush=True)
            return None
        if not _st.get("hit_logged"):
            _st["hit_logged"] = True
            print(f"[m3_fuse] ✅ 首次走融合路径 T={out.shape[0]} topk={inp.shape[1]} H={out.shape[1]}", flush=True)
        _fused(inp, hidden_states, gate_weight, shared_output, out, scale)
        return None

    mod.fused_gate_sigmoid_mul_add = gate

    # 只在会走到门控的分支里开启记账；未被消费则补做并告警（不静默留坑）
    Blk = getattr(mod, "Qwen2MoeSparseMoeBlock", None)
    if Blk is None:
        print("[m3_fuse] 未找到 Qwen2MoeSparseMoeBlock，不安装", flush=True)
        FM.moe_sum_reduce_triton = o_tri
        if o_tc is not None:
            FM.moe_sum_reduce_torch_compile = o_tc
        mod.fused_gate_sigmoid_mul_add = o_gate
        return
    o_fwd = Blk.forward

    def forward(self, hidden_states, forward_batch=None):
        arm = (getattr(self, "shared_expert_gate", None) is not None
               and hidden_states.shape[0] > 0 and not _st["off"])
        prev = _st["armed"]
        _st["armed"] = arm
        try:
            return o_fwd(self, hidden_states, forward_batch)
        finally:
            _st["armed"] = prev
            p = _st["pending"]
            if p is not None:                   # 记了账却没被门控消费 → 补做 + 关闭融合
                _st["pending"] = None
                p[3](p[0], p[1], p[2])
                if not _st["off"]:
                    _st["off"] = True
                    print("[m3_fuse] ⚠ 记账未被门控消费，已补做 reduce 并关闭本补丁"
                          "（该层输出可能已受影响，请查此日志）", flush=True)

    Blk.forward = forward
    mod._m3_installed = True
    print(f"[m3_fuse] 已安装 mode={_MODE} (pid={os.getpid()})", flush=True)
