"""decode 稠密投影走「按 tile 连续存放」的 Triton 内核（风冷 K100_AI 用；与 moe_pack_w13 同一思路）。

为什么：风冷板卡 DF 地址散列未开，跳行读吃亏。rocBLAS（已用风冷选解表）在小 M 下随 M 增大带宽下降
（2560×2048：M=8 294 GB/s、M=32 156），decode 账里风冷 GEMM c1→c2 变慢而液冷不变。实测（离线探针，
图内计时，只用不切 K 的配置）打包 Triton / rocBLAS：
    3072×2048  M=8 1.44  16 1.53  32 2.14  64 1.32      2560×2048  M=8 1.48  16 1.40  32 2.01  64 1.34
    2048×1024  M=8 1.14  16 1.20  32 1.24  64 1.21      2048×4096  M=8 1.14  16 1.62  32 1.41  64 0.87（不用）
    256×2048（路由）与 62080×2048（LM head）打包更慢，不纳入；M=4（c1 验证步）打包不占优，不纳入。
做法：加载后给白名单形状的层另存一份打包副本（[N/32][K/128][32][128]，原权重不动，预填充与小 M 仍走 rocBLAS）；
      apply 时 M 落在该形状的区间内、无偏置、bf16 → 走打包内核（直接写 bf16），否则原路径。
代价：每卡多占白名单层权重大小的显存（约 0.5 GB）；数值与 rocBLAS 不逐位相同（fp32 累加、顺序不同），须看 accept 与精度抽查。
开关：SGLANG_DENSE_PACK=1 开（默认关）。"""
import os
import torch, triton, triton.language as tl

_enabled = os.environ.get("SGLANG_DENSE_PACK", "0") != "0"
PBN, PBK = 32, 128
# (N, K) → (M 下限, M 上限)
_WHITE = {(3072, 2048): (8, 64), (2560, 2048): (8, 64), (2048, 1024): (8, 64), (2048, 4096): (8, 32)}
_seen = {"pack": 0, "run": False, "bytes": 0}


@triton.jit
def _dense_packed_kernel(x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_ym,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PBN_: tl.constexpr, PBK_: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BM); offs_n = pid_n * BN + tl.arange(0, BN); offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wn = ((offs_n // PBN_) * (PBN_ * K) + (offs_n % PBN_) * PBK_)[None, :]
    for kb in range(0, K, BK):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (kb + offs_k)[None, :], mask=offs_m[:, None] < M, other=0.0)
        kg = kb + offs_k   # 元素 (n,k) 在 (n//PBN)·PBN·K + (k//PBK)·PBN·PBK + (n%PBN)·PBK + k%PBK
        w = tl.load(w_ptr + wn + ((kg // PBK_) * (PBN_ * PBK_) + kg % PBK_)[:, None])
        acc += tl.dot(x, w)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=offs_m[:, None] < M)


def _pack(w):
    N, K = w.shape
    return w.view(N // PBN, PBN, K // PBK, PBK).permute(0, 2, 1, 3).contiguous().view(N, K)


def _run(x2, wp, N, K):
    M = x2.shape[0]
    if M <= 16: BN, BK, nw = 32, 256, 2
    else: BN, BK, nw = 64, 256, 8
    y = torch.empty(M, N, device=x2.device, dtype=torch.bfloat16)
    _dense_packed_kernel[(N // BN,)](x2, wp, y, M, N, K, x2.stride(0), y.stride(0),
                                     BM=max(16, triton.next_power_of_2(M)), BN=BN, BK=BK, PBN_=PBN, PBK_=PBK, num_warps=nw)
    return y


def install(mod):
    if not _enabled or getattr(mod, "_dense_pack_installed", False) or not hasattr(mod, "UnquantizedLinearMethod"):
        return
    mod._dense_pack_installed = True
    cls = mod.UnquantizedLinearMethod
    orig_pw, orig_apply = cls.process_weights_after_loading, cls.apply

    def process_weights_after_loading(self, layer):
        orig_pw(self, layer)
        w = getattr(layer, "weight", None)
        if w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or tuple(w.shape) not in _WHITE or not w.is_contiguous():
            return
        layer._dp_packed = _pack(w.data); layer._dp_range = _WHITE[tuple(w.shape)]
        _seen["pack"] += 1; _seen["bytes"] += w.numel() * 2
        if _seen["pack"] in (1, 40, 80):
            print(f"[dense_pack] 已打包 {_seen['pack']} 层（本层 {tuple(w.shape)}，累计额外显存 {_seen['bytes']/2**20:.0f} MiB） pid={os.getpid()}", flush=True)

    def apply(self, layer, x, bias=None, *a, **kw):
        wp = getattr(layer, "_dp_packed", None)
        if wp is not None and bias is None and not a and not kw and x.dtype == torch.bfloat16:
            K = x.shape[-1]; M = x.numel() // K; lo, hi = layer._dp_range
            if lo <= M <= hi:
                x2 = x.reshape(M, K)
                if x2.stride(-1) == 1:
                    N = wp.shape[0]
                    y = _run(x2, wp, N, K)
                    if not _seen["run"]:
                        _seen["run"] = True
                        print(f"[dense_pack] ✅ 首次真正生效：M={M} N={N} K={K} capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
                    return y.view(*x.shape[:-1], N)
        return orig_apply(self, layer, x, bias, *a, **kw)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    print(f"[dense_pack] 已安装：白名单 {_WHITE}（(N,K) → M 区间） pid={os.getpid()}", flush=True)
