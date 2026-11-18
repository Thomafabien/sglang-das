"""MoE gate/up 权重（w13）预打包：为 DF 地址散列关闭的卡（风冷 K100_AI）重排权重，让 fused_moe 每轮读连续的一块。

为什么：Triton fused_moe 每个 block 每轮读 BN 行 × BK 列，w13 行主序 [E, N, K] 下这 BN 段彼此相隔一整行（4 KiB），
是"跳行读"。风冷卡（hy-smi --showhash = Not support）对这种读法极不友好：同样的专家与 tile，
跳行读 70–148 GB/s，同一 tile 连续存放后 273–458 GB/s（离线探针实测）。

做法：
  ① 加载后把每层 w13 按 [N/BN][K/BK][BN][BK] 就地重排（形状、显存不变），登记其 data_ptr；
  ② 替换 triton_utils.fused_moe 里的 invoke_fused_moe_kernel：B 是登记过的打包权重时，走本文件的内核
     （生产 fused_moe_kernel 的 bf16 路径逐行照抄，只改权重地址）；表里的 BN/BK 须为打包块 32/128 的整数倍
     （每轮读几个连续块），否则退到 32/128，其余配置沿用调优表。
  w2（down）的 K=128，一个 tile 本来就是连续的，不动。

正确性：本内核（不打包）与生产内核逐位一致；打包后与打包前逐位一致（离线原型，M=4/8/16/64）。
离线提速（w13 这一段，风冷 4 号卡）：M=4 3.01×、M=8 1.75×、M=16 1.49×、M=64 1.15×。

开关：SGLANG_MOE_PACK_W13=1 开（默认关，只在风冷上开）。不满足条件的层（带偏置、非 bf16、形状不整除）不打包并打印原因；
量化、TMA、融合写回等本内核不支持的调用一律报错而不是静默回退——权重已打包，回退原内核会算错。
"""
import os
import torch, triton, triton.language as tl

_enabled = os.environ.get("SGLANG_MOE_PACK_W13", "0") != "0"
BN, BK = 32, 128
_REG = {}                 # data_ptr -> (BN, BK)
_seen = {"pack": 0, "run": False}


@triton.jit
def _w13_packed_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
                       N, K, EM, num_valid_tokens,
                       stride_am, stride_ak, stride_be, stride_cm, stride_cn,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr, top_k: tl.constexpr, even_Ks: tl.constexpr,
                       PBN: tl.constexpr, PBK: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, compute_type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if off_experts == -1:
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    # 打包布局 [N/PBN][K/PBK][PBN][PBK]：元素 (n, k) 在 (n//PBN)·PBN·K + (k//PBK)·PBN·PBK + (n%PBN)·PBK + k%PBK。
    # BN、BK 取 PBN、PBK 的整数倍时，每轮读 (BN/PBN)×(BK/PBK) 个各自连续的打包块。
    offs_j = tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = (b_ptr + off_experts * stride_be + pid_n * BLOCK_SIZE_N * K
              + ((offs_j // PBN) * (PBN * K) + (offs_j % PBN) * PBK)[None, :]
              + ((offs_k // PBK) * (PBN * PBK) + offs_k % PBK)[:, None])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k_start), other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * PBN
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _pack(w):
    E, N, K = w.shape
    return w.view(E, N // BN, BN, K // BK, BK).permute(0, 1, 3, 2, 4).contiguous().view(E, N, K)


def _install_pack(mod):
    cls = mod.UnquantizedFusedMoEMethod
    orig = cls.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig(self, layer)
        w = getattr(layer, "w13_weight", None)
        why = None
        if w is None or w.dim() != 3: why = "无 3 维 w13_weight"
        elif w.dtype != torch.bfloat16: why = f"dtype {w.dtype}"
        elif w.shape[1] % BN or w.shape[2] % BK: why = f"形状 {tuple(w.shape)} 不整除 {BN}/{BK}"
        elif getattr(layer, "w13_weight_bias", None) is not None: why = "带偏置"
        elif not w.is_contiguous(): why = "非连续"
        if why:
            print(f"[moe_pack_w13] 跳过一层：{why}", flush=True)
            return
        p = _pack(w.data)
        w.data.copy_(p); del p
        _REG[w.data_ptr()] = (BN, BK)
        _seen["pack"] += 1
        if _seen["pack"] == 1:
            print(f"[moe_pack_w13] ✅ 首次打包 w13 {tuple(w.shape)} → tile {BN}×{BK} pid={os.getpid()}", flush=True)

    cls.process_weights_after_loading = process_weights_after_loading
    print(f"[moe_pack_w13] 已安装（权重打包）pid={os.getpid()}", flush=True)


def _install_kernel(mod):
    orig = mod.invoke_fused_moe_kernel

    def invoke(A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids, sorted_token_ids, expert_ids,
               num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type=None, **kw):
        if B.data_ptr() not in _REG:
            return orig(A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type=compute_type, **kw)
        bad = [k for k in ("use_fp8_w8a8", "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16", "per_channel_quant",
                           "a_use_tma", "b_use_tma", "c_sorted", "no_combine", "fuse_sum_all_reduce",
                           "fuse_add_to_output", "mask_output", "lora_preserve_base") if kw.get(k)]
        if bias is not None: bad.append("bias")
        if kw.get("block_shape") is not None: bad.append("block_shape")
        if bad:
            raise RuntimeError(f"[moe_pack_w13] 打包权重遇到不支持的调用 {bad}（不可回退原内核，会算错）")
        pbn, pbk = _REG[B.data_ptr()]
        N, K = B.shape[1], B.shape[2]
        cfg = dict(config); bn, bk = cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
        if bn % pbn or N % bn: bn = pbn          # 表里的 BN/BK 不是打包块整数倍时退到打包块本身
        if bk % pbk or K % bk: bk = pbk
        cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"] = bn, bk
        grid = (triton.cdiv(sorted_token_ids.shape[0], cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, bn),)
        _w13_packed_kernel[grid](
            A, B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, sorted_token_ids.shape[0], topk_ids.numel(),
            A.stride(0), A.stride(1), B.stride(0), C.stride(-2), C.stride(-1),
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=cfg.get("GROUP_SIZE_M", 1),
            top_k=top_k, even_Ks=(K % bk == 0), PBN=pbn, PBK=pbk, MUL_ROUTED_WEIGHT=bool(mul_routed_weight),
            compute_type=compute_type if compute_type is not None else tl.bfloat16,
            num_warps=cfg.get("num_warps", 4), num_stages=cfg.get("num_stages", 2))
        if not _seen["run"]:
            _seen["run"] = True
            print(f"[moe_pack_w13] ✅ 首次真正生效：A {tuple(A.shape)} B {tuple(B.shape)} 配置 BM{cfg['BLOCK_SIZE_M']} "
                  f"BN{bn} BK{bk} w{cfg.get('num_warps')} capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)

    mod.invoke_fused_moe_kernel = invoke
    print(f"[moe_pack_w13] 已安装（内核替换）pid={os.getpid()}", flush=True)


def install(mod):
    if not _enabled or getattr(mod, "_moe_pack_w13_installed", False):
        return
    mod._moe_pack_w13_installed = True
    if hasattr(mod, "UnquantizedFusedMoEMethod"):
        _install_pack(mod)
    if hasattr(mod, "invoke_fused_moe_kernel"):
        _install_kernel(mod)
