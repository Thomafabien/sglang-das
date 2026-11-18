"""act_flat: 替换 MoE 路由专家的激活核 silu_and_mul 的窄行宽退化路径。

## 机制（2026-09-13 实测钉死）
厂商 `sgl_hip::activation::act_and_mul_kernel`（include/hip/hip_act_and_mul.cuh:30，
启动在 csrc/elementwise/activation.cu:87）用 `grid=(num_tokens,)`、
`block=min(d/vec_size, 1024)`，bf16 下 vec_size=8 ⇒ **一行只发 D/8 个线程**。
可证伪预测已验证：固定 M=65536 扫 D，耗时与 D 无关
（D=64/128/256 → 242.2/244.0/244.4 µs，差<1%），达成带宽严格正比于 D；
D=512 时 64 线程拿到满 wave64，病理消失（327.9 µs / 614 GB/s / 87%）。
即「块数×单块延迟」受限，不是带宽受限；**加大 blockDim 无效**（`idx += stride` 循环只跑一次）。
⇒ 病理条件是 **D/8 < 64**，即 D < 512。TP4 下 D = moe_intermediate_size/TP = 128，只用 16/64 条车道。
D=128 时厂商核的尾部标量循环不执行（`d % (stride*vec_size) = 128 % 128 = 0`），没有第二条数值路径。

## 代价模型（eager 口径）
厂商 ≈ 9 µs + 3.67 ns×M；本核 ≈ 11 µs + 1.10 ns×M（达同批 copy 对照 93%）。
GPU 侧交叉点 M≈780。但 Triton 每次发射 55 µs CPU（厂商 C++ 11 µs），每层多 44 µs、40 层 1.76 ms
⇒ **壁钟交叉点 M > 17121**。门限 16384 与它只差 4%：落在 16384–17121 这段余量里的生产形状
只有图桶 2048（M=16384）与 2112（M=16896），而它们在 CUDA graph 内重放、重放期不执行 Python 包装器，
44 µs/层一次都不付 ⇒ 这两个形状净赚。**不要把门限往下调**：1024 输入（M=8704）GPU 只省 0.9 ms
而 eager CPU 要多付 1.76 ms，按此账净亏或打平。

## 门限 20000：构造性地把补丁限制在 eager 大块上（2026-09-13 定案）
早期用 16384，它把预填充图桶 2048（M=16384）与 2112（M=16896）也纳入生效范围，带来两处代价：
① 短行高并发落在「GPU 省 1.71 ms/批 vs Triton 发射多付 1.76 ms/批」的同量级反号模糊区，
   两臂实测行 4/8/9 开臂反而慢 0.8–2.4%（跨实例，未定论，但方向不利）；
② 整个 CUDA graph 风险面（核是否真被录进图、捕获期 JIT、图内计数不可见）都要论证。
抬到 20000 后：实测形状分布是 eager 块 M=65536/65040/53584/41376/37336（全 >20000）、
图桶 16896/16384（全 <20000），**中间无形状**，所以是干净切割。
代价仅 0.05 O（+0.50 → +0.45），换掉模糊区与整个图风险面。
eager 大块上的收益是三臂 A-B-A 干净测过的：16384 行 −0.92%、32768 行 −0.85%，
与代价模型预测 0.94%/0.87% 吻合 ≤1.2%。

## CUDA graph：门限 20000 下不再进任何图（以下为 16384 时的论证，留档）
- **会**：`--cuda-graph-bs-prefill` 的桶是 **token 数**（prefill_cuda_graph_runner.py:169-177），
  含 2048/2112 ⇒ M = 16384/16896 均 ≥ 门限；捕获按 `reversed(buckets)` 降序，首个形状即 2112。
  日志实测 `首次真正生效: M=16896` 出现在 `Capture ... begin` 与 `end` 之间。
- **捕获期 JIT 由图后端的两遍 eager warmup 兜住**（breakable_cuda_graph_backend.py:113-119
  `for _ in range(2): synchronize(); barrier(); forward_fn()`，注释写明「Two warmups so kernels
  are loaded and one-time setup is paid before capture」）⇒ 编译发生在 warmup，不在捕获。
- **不会进 decode 图**：M = bs × num_draft_tokens × topk = bs×4×8；`max_running_requests=48`
  （上限由 mamba 池而非 KV cache 决定）⇒ M ≤ 1536，离门限 10.7 倍。
  ⚠ decode 桶表排到 bs=512，若有人把 max-running-requests 提到 ≥512，M 正好 = 16384 撞上门限 ⇒
  install 里有 assert 护栏。
- **本函数会在捕获期被执行，因此禁止任何同步/读值操作**（capture_error_mode="global" 是最严格模式）。

## 正确性（真分布 + 6 档尺度 + 8 类边界，全部离线实测）
机制：厂商是 `round(round(silu(x)) · y)` 两次 bf16 舍入（模板返回值即 T=bf16）；
本核全程 fp32、只在 store 舍一次。
- **真实 w13 GEMM 输出**（生产分布，|x|max≈5.4）：M=16896 与 M=65536 两个形状下，
  厂商更准的元素 **0 个**、本核更准 27.45–27.46%、其余逐位相同；均值|Δ| 4.48e-4 对 6.28e-4。
- 尺度扫描：×0.01/×0.1/×1/×8 厂商更准全为 0；**×64 出现 103 个、×512 出现 19 个**且均值|Δ|相等
  ⇒ **「严格占优」只在生产量程及其 8 倍内成立**，超出 60 倍量程优势消失。
- 边界：全零 / ±1e4 / denormal 1e-40 / bf16 最大值 → 100% 逐位相同；
  含 inf / 含 nan → 两边意外非有限元素都是 0。
- **有 27.5% 元素不逐位相同 ⇒ 生成序列必然分叉，任何「开/关输出逐字一致」的判据都不可用。**

## 收益（已修正）
按块结构逐行算：1024 档 c1 为 0（低于门限）、2048 档 0.64%、4096 0.6%、7168 0.74%、
16384 0.94%、32768 0.87%、65536 0.68%、131072 0.51% ⇒ 45 行加权 TTFT −0.62%
⇒ **ΔO ≈ +0.52~0.6**（不是 +0.67，那是把 16384 行的 0.92% 套到全部行）。
实测 A-B-A：16384 行 −0.92%、32768 行 −0.85%，与模型预测 0.94%/0.87% 吻合 ≤1.2%。
注意 `prefill_max_requests=2`：两条 1024 请求会合批成 2048 token ⇒ 1024 档 **c≥2 的行也会触发**，
只有 c1 那一行是干净对照。

## 作用域
只覆盖**路由专家**。共享专家走另一个符号（`silu_and_mul_kernel_opt1<c10::BFloat16,2>`，
125 次/22.9 µs），同样病理仍在，但其 M 低于门限，本补丁替换不到也不会生效。

env: SGLANG_ACT_FLAT=0/false/off/no 关闭（默认开）/ _MIN 门限 / _BLOCK / _WARPS / _AUDIT 收尾统计
"""
import atexit
import hashlib
import os

import torch
import triton
import triton.language as tl


def _env_int(name, default):
    v = os.environ.get(name, "")
    if v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        print(f"[act_flat] 警告: {name}={v!r} 无法解析为整数, 回落默认 {default}", flush=True)
        return default


_ON = os.environ.get("SGLANG_ACT_FLAT", "1").strip().lower() not in ("0", "false", "off", "no", "")
_MIN_M = _env_int("SGLANG_ACT_FLAT_MIN", 20000)
_BLOCK = _env_int("SGLANG_ACT_FLAT_BLOCK", 1024)
_WARPS = _env_int("SGLANG_ACT_FLAT_WARPS", 4)
_MAX_D = _env_int("SGLANG_ACT_FLAT_MAX_D", 256)   # 病理条件是 D/8 < 64 ⇒ D < 512
_AUDIT = os.environ.get("SGLANG_ACT_FLAT_AUDIT", "1") != "0"

# import 期断言：让非法旋钮在 sitecustomize 的「安装失败, 回退原实现」分支被接住并打出原因
assert _BLOCK >= 16 and (_BLOCK & (_BLOCK - 1)) == 0, f"BLOCK 必须是 2 的幂且 >=16, 实为 {_BLOCK}"
assert _WARPS in (1, 2, 4, 8, 16), f"num_warps 必须属于 (1,2,4,8,16), 实为 {_WARPS}"

_state = {"fired": 0, "vendor_calls": 0, "shapes": {}, "warned": False, "pid": os.getpid()}


@triton.jit
def _silu_mul_flat_kernel(x_ptr, o_ptr, NEL, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < NEL
    m = idx // D
    d = idx % D
    base = m * (2 * D)
    a = tl.load(x_ptr + base + d, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + base + D + d, mask=mask, other=0.0).to(tl.float32)
    o = (a * tl.sigmoid(a)) * b
    tl.store(o_ptr + idx, o.to(o_ptr.dtype.element_ty), mask=mask)


def _capturing():
    """只在打日志时调用（很稀疏）。查询不是同步操作，捕获期安全。"""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return None


def install(module):
    # 幂等守卫：标记必须打在**函数对象**上。
    # 若打在模块上会有致命静默回退：sitecustomize 的 loader 在 install 前会先
    # `self.real.exec_module(module)` 重跑模块体，把 module.silu_and_mul 重置成干净厂商函数，
    # 而模块级标记仍为 True ⇒ 打印「已装过」却实际跑厂商核，性能掉回基线并被误判成「优化无效」。
    if getattr(module.silu_and_mul, "_act_flat", False):
        print(f"[act_flat] 已装过, 跳过二次安装 pid={os.getpid()}", flush=True)
        return
    if not _ON:
        print(f"[act_flat] 已跳过（SGLANG_ACT_FLAT 关）pid={os.getpid()}", flush=True)
        return

    vendor = module.silu_and_mul

    def dispatched(x, out=None, *args, **kwargs):
        # ⚠ 本函数会在 CUDA graph 捕获期被执行（预填充桶 2048/2112）。
        # 禁止任何 .item()/float()/cpu()/synchronize()/打印张量值 —— capture_error_mode="global"。
        if (
            out is not None
            and not args
            and not kwargs
            and out.dim() == 2
            and x.dim() == 2
            and out.shape[0] >= _MIN_M
            and out.shape[1] <= _MAX_D                  # 病理条件：D/8 < wave64
            and out.is_contiguous()
            and x.is_contiguous()
            and x.dtype == out.dtype
            and x.shape[0] == out.shape[0]
            and x.shape[1] == 2 * out.shape[1]
            and (out.shape[1] & (out.shape[1] - 1)) == 0
        ):
            M, D = out.shape
            k = (M, D)
            new_shape = k not in _state["shapes"]
            _state["shapes"][k] = _state["shapes"].get(k, 0) + 1
            _state["fired"] += 1
            _silu_mul_flat_kernel[(triton.cdiv(M * D, _BLOCK),)](
                x, out, M * D, D=D, BLOCK=_BLOCK, num_warps=_WARPS
            )
            if new_shape:   # 日志放在 launch 之后：打了这行就代表 enqueue 成功
                print(
                    f"[act_flat] 生效·新形状 M={M} D={D} grid={triton.cdiv(M*D,_BLOCK)} "
                    f"capturing={_capturing()} pid={os.getpid()}",
                    flush=True,
                )
            # 周期性累计：killsrv.sh 用 kill -9，atexit 永不执行，不能依赖收尾统计
            if _AUDIT and _state["fired"] % 2000 == 0:
                print(
                    f"[act_flat] 累计 生效={_state['fired']} 回退={_state['vendor_calls']} "
                    f"形状数={len(_state['shapes'])} pid={os.getpid()}（图重放不计入）",
                    flush=True,
                )
            return out
        _state["vendor_calls"] += 1
        # 阴性自报警：只喊一次。最可能的失效形态是「谓词永远不满足」，靠人记得去 grep 缺席不可靠。
        if not _state["warned"] and _state["vendor_calls"] >= 200 and _state["fired"] == 0:
            _state["warned"] = True
            sh = (tuple(out.shape) if (out is not None and hasattr(out, "shape")) else None)
            print(
                f"[act_flat] 警告: 已回退 {_state['vendor_calls']} 次仍未生效一次; "
                f"门限={_MIN_M} 末次 out.shape={sh} args={len(args)} kwargs={list(kwargs)} "
                f"pid={os.getpid()}",
                flush=True,
            )
        return vendor(x, out, *args, **kwargs)

    dispatched._act_flat = True
    module.silu_and_mul = dispatched
    module._act_flat_state = _state

    try:
        h = hashlib.sha1(open(__file__.replace(".pyc", ".py"), "rb").read()).hexdigest()[:8]
    except Exception:
        h = "?"
    print(
        f"[act_flat] 已安装: 门限 M>={_MIN_M} maxD={_MAX_D} BLOCK={_BLOCK} warps={_WARPS} "
        f"pid={os.getpid()} sha1={h} file={__file__}",
        flush=True,
    )

    if _AUDIT:
        def _dump():
            try:
                sh = ", ".join(f"{m}x{d}:{n}" for (m, d), n in sorted(_state["shapes"].items()))
                print(
                    f"[act_flat] 收尾统计 pid={os.getpid()} 生效={_state['fired']} "
                    f"厂商回退={_state['vendor_calls']} 形状={{{sh}}}  "
                    f"（注意：图重放不计入；且 kill -9 下本行不会出现，以「累计」行为准）",
                    flush=True,
                )
            except Exception:
                pass
        atexit.register(_dump)
