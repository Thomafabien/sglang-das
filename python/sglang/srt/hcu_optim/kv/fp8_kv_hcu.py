"""fp8_kv_hcu：在 gfx928 上让 --kv-cache-dtype fp8_e5m2 走通（实验项，默认关，SGLANG_FP8_KV_HCU=1 开）。

## 依据（2026-09-21 离线探针，/workspace/pa_probe4.py、draft_probe.py、prefill_probe.py）
- 厂商 flash_attn_2_cuda.paged_attention 在 gfx928 上有 fp8 KV 内核
  `paged_attention_kernel<u16, u8, false, 256, 64, 256, N>`，按 K/V 张量的标量类型选中，
  **只有 float8_e5m2 算对**（对同一量化 KV 的 fp32 参考 maxabs 3.7e-3，与 bf16 内核同级）；e4m3fn 输出错。
  decode 两条路径都走它：草稿步 q=1（vllm_flash_attn_with_kvcache）、验证步 q=4（varlen 的 is_mtp 分支）。
  实测 131072×4：q=1 1.497→0.771 ms、q=4 1.426→0.830 ms；8192×1 反而 0.071→0.081（延迟受限）。
- **预填充的 fp8 路径是空操作**：varlen 非 mtp 分支对 fp8 池调 vllm_mha_varlen_fwd_kv_fp8，
  打印了参数却不发射任何内核、输出全零、0.26 ms 恒定（= 旧记忆里的「快 380 倍」）。
- sglang 的 `_attention_backend_fa3_fp8_fallback`（arg_groups/overrides.py:1787）把 fa3+fp8_e5m2 一律改成 triton，
  这条规则是给 NVIDIA FA3 写的；海光的 fa3 是厂商 flash-attn，且 triton 后端不认海光 KV 布局（启动即断言失败）。

## 本优化做两件事
1. overrides：海光上保留 fa3，不回退 triton。
2. flashattention_interface.vllm_flash_attn_varlen_func_interface：当 K/V 是 fp8 且这次调用**不会**进入厂商的
   paged_attention 分支（复刻厂商判据：max_seqlen_q==1 或 is_mtp，且 window 左界 -1）时，
   按块表把引用到的页 gather 出来反量化成 bf16 临时分页缓冲，块表重排为 0..bs*P-1，再调原函数（走 bf16 splitkv）。
   decode 路径不经过这里的改写，仍直接读 fp8 池。
   纯张量操作、无 .item()/同步，可被 CUDA graph 捕获。代价：每层读 fp8 页 + 写 2 倍字节的 bf16。
"""
import os, torch
from .. import PATCH_DIR as _PATCH_DIR

_ON = os.environ.get("SGLANG_FP8_KV_HCU", "0") != "0"
# 自写分页注意力内核（verify 步 q=4）。SGLANG_PA_OWN=1 开启；默认关，开启时才加载 .so。
_OWN = os.environ.get("SGLANG_PA_OWN", "0") != "0"
_OWN_FMT = os.environ.get("SGLANG_PA_FMT", "e4m3")      # e4m3（精度 2 倍）或 e5m2（与厂商一致）
_CHECK = int(os.environ.get("SGLANG_PA_CHECK", "0"))    # >0：前 N 次调用同时跑厂商内核对拍（仅 e5m2 有意义）
_chk_n = [0]
_pa = {}            # sq -> PagedAttn 实例，懒加载
_pa_fail = False
_FP8 = (torch.float8_e5m2, torch.float8_e4m3fn)
_TRACE = os.environ.get("SGLANG_PA_TRACE", "0") != "0"
# 页表结构诊断：默认关。开启会在首个符合条件的 decode 步做一次设备同步（见 _try_own 内的说明）。
_DIAG_BT = os.environ.get("SGLANG_PA_DIAG_BT", "0") != "0"


class _SkipDiag(Exception):
    """诊断条件不满足时的内部跳过信号，不代表错误。"""
_route_seen = set()
_seen = {"dequant": False, "direct": False, "own": False, "own_err": False,
         "own_kv": False, "own_kv_err": False}

# 自写分页 prefill 注意力（读 fp8 KV，装载时解包成 fp16 走 mmac；离线 131072×8192 为厂商 bf16 的 65%）。
# SGLANG_PP_OWN=1 开启；形状不吻合或调用异常一律回退「反量化→bf16 厂商」路径并打日志。
_PP_OWN = os.environ.get("SGLANG_PP_OWN", "0") != "0"
_pp = {"lib": None, "fail": False}
_seen.update({"pp": False, "pp_err": False, "pp_fallback": False})


def _load_pp():
    if _pp["fail"] or _pp["lib"] is not None:
        return _pp["lib"]
    try:
        import ctypes
        d = _PATCH_DIR          # .so 位于 hcu_optim/lib
        fp = os.path.join(d, f"libpp_v5_q{4 if _OWN_FMT == 'e4m3' else 5}.so")
        lib = ctypes.CDLL(fp)
        lib.pp_prefill.restype = None
        lib.pp_prefill.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        lib.pp_qfmt.restype = ctypes.c_int
        _pp["lib"] = lib
        print(f"[fp8_kv_hcu] 自写 prefill 内核已加载: {os.path.basename(fp)} QFMT={lib.pp_qfmt()} pid={os.getpid()}", flush=True)
        return lib
    except Exception as e:
        _pp["fail"] = True
        print(f"[fp8_kv_hcu] ⚠ 自写 prefill 内核加载失败，永久回退反量化路径: {type(e).__name__}: {e}", flush=True)
        return None


# 草稿扩展（每序列 1..4 个新 token、变长）专用：split-KV 的 pa_ext 内核。此前走 prefill 核，grid 只有 bs 个 block，
# 128k 一次 17 ms（离线实测）；pa_ext 同形状 0.3 ms。SGLANG_PA_EXT=0 可关（回退 prefill 核路径以便 A/B）。
_EXT_ON = os.environ.get("SGLANG_PA_EXT", "0") != "0"   # 默认关：路由取证证明草稿扩展不经此入口，留作备用
_ext = {"lib": None, "fail": False, "ws": {}}
_seen.update({"ext": False, "ext_err": False})


def _load_ext():
    if _ext["fail"] or _ext["lib"] is not None:
        return _ext["lib"]
    try:
        import ctypes
        d = _PATCH_DIR          # .so 位于 hcu_optim/lib
        fp = os.path.join(d, f"libpa_ext_{_OWN_FMT}.so")
        lib = ctypes.CDLL(fp)
        lib.pa_decode_ext.restype = None
        lib.pa_decode_ext.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        lib.pa_pick_nsplit.restype = ctypes.c_int
        lib.pa_pick_nsplit.argtypes = [ctypes.c_int, ctypes.c_int]
        _ext["lib"] = lib
        print(f"[fp8_kv_hcu] 草稿扩展内核已加载: {os.path.basename(fp)} pid={os.getpid()}", flush=True)
        return lib
    except Exception as e:
        _ext["fail"] = True
        print(f"[fp8_kv_hcu] ⚠ 草稿扩展内核加载失败，永久回退: {type(e).__name__}: {e}", flush=True)
        return None


def _try_ext(q, k, v, bt, cu_q, kw, mq, bs):
    """草稿扩展形状（max_seqlen_q ≤ 4、因果、有 cu_seqlens_q）用 pa_ext；不吻合返回 None。"""
    if not _EXT_ON or mq > 4 or cu_q is None or not kw.get("causal", False) or kw.get("seqused_k") is None:
        return None
    if k.dim() != 4 or k.shape[1] != 1 or k.shape[2] != 64 or k.shape[3] != 256 or q.dim() != 3 or q.shape[1] != 4 or q.shape[2] != 256:
        return None
    if kw.get("alibi_slopes") is not None or (kw.get("softcap") or 0.0) != 0.0 or kw.get("return_softmax_lse", False):
        return None
    lib = _load_ext()
    if lib is None:
        return None
    try:
        qc = q.contiguous(); out = torch.empty_like(qc)
        btc = bt.to(torch.int32).contiguous(); cuc = cu_q.to(torch.int32).contiguous(); skc = kw["seqused_k"].to(torch.int32).contiguous()
        n_per = int(btc.shape[1]); ns = int(lib.pa_pick_nsplit(n_per, int(bs)))
        key = (int(bs), ns)
        if key not in _ext["ws"]:
            _ext["ws"][key] = (torch.empty(bs * ns * 16 * 256, dtype=torch.float32, device=q.device),
                               torch.empty(bs * ns * 16 * 2, dtype=torch.float32, device=q.device))
        o_part, ml = _ext["ws"][key]
        lib.pa_decode_ext(out.data_ptr(), qc.data_ptr(), k.data_ptr(), v.data_ptr(), btc.data_ptr(), skc.data_ptr(), cuc.data_ptr(),
                          o_part.data_ptr(), ml.data_ptr(), int(bs), n_per, n_per, ns, float(kw.get("softmax_scale")),
                          torch.cuda.current_stream().cuda_stream)
        if not _seen["ext"]:
            _seen["ext"] = True
            print(f"[fp8_kv_hcu] ✅ 首次真正生效(草稿扩展 pa_ext {_OWN_FMT}): q={tuple(q.shape)} max_seqlen_q={mq} bs={bs} 页表宽 {n_per} nsplit={ns} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return out
    except Exception as e:
        if not _seen["ext_err"]:
            _seen["ext_err"] = True
            print(f"[fp8_kv_hcu] ⚠ 草稿扩展内核调用异常，本次回退: {type(e).__name__}: {e}", flush=True)
        return None


def _try_pp(q, k, v, bt, cu_q, kw, mq, bs):
    """prefill 形状吻合就用自写内核；否则返回 None（调用方回退反量化路径）。"""
    if not _PP_OWN:
        return None
    why = None
    if k.dim() != 4 or k.shape[1] != 1 or k.shape[2] != 64 or k.shape[3] != 256:
        why = f"K 形状 {tuple(k.shape)}"
    elif q.dim() != 3 or q.shape[1] != 4 or q.shape[2] != 256:
        why = f"Q 形状 {tuple(q.shape)}"
    elif not kw.get("causal", False):
        why = "causal=False"
    elif kw.get("seqused_k") is None:
        why = "无 seqused_k"
    elif kw.get("alibi_slopes") is not None or (kw.get("softcap") or 0.0) != 0.0:
        why = "alibi/softcap"
    elif kw.get("return_softmax_lse", False):
        why = "要 lse"
    elif k.dtype != (torch.float8_e4m3fn if _OWN_FMT == "e4m3" else torch.float8_e5m2):
        why = f"K dtype {k.dtype} 与 SGLANG_PA_FMT={_OWN_FMT} 不符"
    if why is not None:
        if not _seen["pp_fallback"]:
            _seen["pp_fallback"] = True
            print(f"[fp8_kv_hcu] prefill 自写内核回退（{why}），走反量化路径 pid={os.getpid()}", flush=True)
        return None
    lib = _load_pp()
    if lib is None:
        return None
    try:
        qc = q.contiguous()
        out = torch.empty_like(qc)
        btc = bt.to(torch.int32).contiguous()
        cuc = cu_q.to(torch.int32).contiguous()
        skc = kw["seqused_k"].to(torch.int32).contiguous()
        lib.pp_prefill(out.data_ptr(), qc.data_ptr(), k.data_ptr(), v.data_ptr(), btc.data_ptr(),
                       cuc.data_ptr(), skc.data_ptr(), int(bs), int(btc.shape[1]), int(mq),
                       float(kw.get("softmax_scale")), torch.cuda.current_stream().cuda_stream)
        if not _seen["pp"]:
            _seen["pp"] = True
            print(f"[fp8_kv_hcu] ✅ 首次真正生效(prefill 自写内核 {_OWN_FMT}): q={tuple(q.shape)} max_seqlen_q={mq} bs={bs} "
                  f"页表宽 {btc.shape[1]} seqused[:2]={(skc[:2].tolist() if not torch.cuda.is_current_stream_capturing() else '(capturing)')} K={tuple(k.shape)} {k.dtype} "
                  f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return out
    except Exception as e:
        if not _seen["pp_err"]:
            _seen["pp_err"] = True
            print(f"[fp8_kv_hcu] ⚠ prefill 自写内核调用异常，本次回退反量化路径: {type(e).__name__}: {e}", flush=True)
        return None


def _is_hcu():
    try:
        from sglang.srt.utils import is_hcu
        return bool(is_hcu())
    except Exception:
        return False





def _load_pa(sq):
    """懒加载自写内核（sq=4 验证步 / sq=1 草稿步）。失败只警告一次并永久回退厂商。"""
    global _pa_fail
    if _pa_fail:
        return None
    if sq in _pa:
        return _pa[sq]
    try:
        import sys as _sys
        d = _PATCH_DIR          # .so 位于 hcu_optim/lib
        if d not in _sys.path:
            _sys.path.insert(0, d)
        from .pa_py import PagedAttn
        _pa[sq] = PagedAttn(_OWN_FMT, libdir=d, sq=sq)
        print(f"[fp8_kv_hcu] 自写内核已加载: libpa_{_OWN_FMT}_q{sq}.so QFMT={_pa[sq].qfmt} pid={os.getpid()}", flush=True)
        return _pa[sq]
    except Exception as e:
        _pa_fail = True
        print(f"[fp8_kv_hcu] ⚠ 自写内核加载失败，永久回退厂商: {type(e).__name__}: {e}", flush=True)
        return None


def _try_own(q, k, v, bt, kw, mq, bs):
    """形状吻合就用自写内核，返回 [total_q, HQ, HD]；不吻合返回 None（调用方回退厂商）。"""
    if not _OWN or mq not in (1, 4):
        return None
    pa = _load_pa(mq)
    if pa is None:
        return None
    # 池形状：K [pages, HK, PAGE, HD]，V [pages, HK, HD, PAGE]
    if k.dim() != 4 or k.shape[1] != 1 or k.shape[2] != 64 or k.shape[3] != 256:
        return None
    if q.dim() != 3 or q.shape[1] != 4 or q.shape[2] != 256:
        return None
    if q.shape[0] != bs * mq:
        return None
    seqused = kw.get("seqused_k")
    if seqused is None:
        return None
    try:
        import torch as _t
        out = _t.empty_like(q)
        npages = int(bt.shape[1])
        pa(out.view(bs, mq, 4, 256), q.view(bs, mq, 4, 256), k, v,
           bt.to(_t.int32), seqused.to(_t.int32), npages, float(kw.get("softmax_scale")))
        # 页表结构诊断（连续段数），用于核对 decode 内核「顺序页表」的设计前提。
        # 默认关闭：它含 int(seqused.max())，那是一次设备同步。此前该同步写在条件链尾部，
        # 而标记只在全部条件成立时才置位，于是 bs≥2 且上下文 <32768 的每一步都要同步一次
        # （实测每步多约 2 ms，即"首遍效应"的根因）。现在改为先置位、再判断，且整块由开关控制。
        if _DIAG_BT and (not _seen.get("bt_shape")) and bs >= 2 and (not _t.cuda.is_current_stream_capturing()):
            _seen["bt_shape"] = True          # 无论后续条件是否成立，本进程只进这里一次
            try:
                if int(seqused.max()) < 32768:
                    raise _SkipDiag
                parts = []
                for bi in range(min(bs, 4)):
                    n = (int(seqused[bi]) + 63) // 64; row = bt[bi, :n].to(_t.int64)
                    runs = int((row[1:] - row[:-1] != 1).sum().item()) + 1
                    parts.append(f"seq{bi}: 页数 {n} 首页 {int(row[0])} 连续段 {runs} 前6 {row[:6].tolist()}")
                print("[fp8_kv_hcu] 真实页表结构: " + " | ".join(parts) + f" pid={os.getpid()}", flush=True)
            except _SkipDiag:
                pass
            except Exception as e:
                print(f"[fp8_kv_hcu] 页表结构统计失败: {e}", flush=True)
        if not _seen["own"]:
            _seen["own"] = True
            print(f"[fp8_kv_hcu] ✅ 首次真正生效(decode 自写内核 {_OWN_FMT} q={mq}): q={tuple(q.shape)} bs={bs} "
                  f"页表宽 {npages} K={tuple(k.shape)} {k.dtype} "
                  f"capturing={_t.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
        return out
    except Exception as e:
        if not _seen["own_err"]:
            _seen["own_err"] = True
            print(f"[fp8_kv_hcu] ⚠ 自写内核调用异常，本次及后续回退厂商: {type(e).__name__}: {e}", flush=True)
        return None


def dispatch_varlen(orig, *args, **kw):
    k = kw.get("k", args[1] if len(args) > 1 else None)
    bt = kw.get("block_table")
    if k is None or k.dtype not in _FP8 or bt is None:
        return orig(*args, **kw)
    q = kw.get("q", args[0] if args else None)
    v = kw.get("v", args[2] if len(args) > 2 else None)
    cu_q = kw.get("cu_seqlens_q")
    mq = kw.get("max_seqlen_q")
    ws = kw.get("window_size")
    w0 = -1 if ws is None else ws[0]
    bs = cu_q.shape[0] - 1
    is_mtp = (mq * bs == q.shape[0] and 1 < mq < 5)
    # 路由取证（SGLANG_PA_TRACE=1）：每种 (mq, bs, total_q, causal) 组合打一次，最多 12 种
    if _TRACE and int(mq) <= 64 and len(_route_seen) < 12 and not torch.cuda.is_current_stream_capturing() and kw.get("seqused_k") is not None and int(kw["seqused_k"][0]) > 10000:
        key = (int(mq), int(bs), int(q.shape[0]), bool(kw.get("causal", False)))
        if key not in _route_seen:
            _route_seen.add(key)
            print(f"[fp8_kv_hcu][route] varlen: mq={mq} bs={bs} total_q={q.shape[0]} q={tuple(q.shape)} K={tuple(k.shape)} {k.dtype} "
                  f"causal={kw.get('causal')} is_mtp={is_mtp} w0={w0} seqused[:2]={kw.get('seqused_k')[:2].tolist() if kw.get('seqused_k') is not None else None} pid={os.getpid()}", flush=True)
    if (mq == 1 or is_mtp) and w0 == -1:
        # decode 路径。形状吻合时用自写内核（verify 步 q=4 / 每卡 1 个 kv 头 / hd 256 / page 64），
        # 否则回退厂商 paged_attention：它只对 e5m2 正确，e4m3 会算错，故下面对 e4m3 直接报错。
        own = _try_own(q, k, v, bt, kw, mq, bs)
        if own is not None:
            if _CHECK and _chk_n[0] < _CHECK:
                _chk_n[0] += 1
                try:
                    import torch as _t
                    ref = orig(*args, **kw)
                    r = ref[0] if isinstance(ref, (tuple, list)) else ref
                    d = (own.float() - r.float())
                    rel = (d.pow(2).mean().sqrt() / r.float().pow(2).mean().sqrt()).item()
                    sq = kw.get("seqused_k")
                    print(f"[pa_check#{_chk_n[0]}] bs={bs} q={tuple(q.shape)} 页表宽={bt.shape[1]} "
                          f"seqused={sq[:4].tolist() if sq is not None else None} "
                          f"最大差={d.abs().max().item():.4e} 相对RMS={rel:.4e} "
                          f"本核absmax={own.float().abs().max().item():.4e} 厂商absmax={r.float().abs().max().item():.4e}",
                          flush=True)
                except Exception as e:
                    print(f"[pa_check] 对拍失败: {type(e).__name__}: {e}", flush=True)
            return own
        ext_out = _try_ext(q, k, v, bt, cu_q, kw, mq, bs)
        if ext_out is not None:
            return ext_out
        if k.dtype is torch.float8_e4m3fn:
            # 厂商 paged_attention 的 fp8 内核只对 e5m2 算对，e4m3 会输出错误结果（不是变慢）。
            # 走到这里说明自写内核未接这个形状，宁可失败也不能返回错误数值。
            raise RuntimeError(
                f"[fp8_kv_hcu] decode 形状未被自写内核接管而 KV 为 e4m3，厂商内核对该格式不正确："
                f"q={tuple(q.shape)} max_seqlen_q={mq} bs={bs} K={tuple(k.shape)}。"
                f" 请检查 .so 是否编译（bash kernels/build.sh），或改用 SGLANG_PA_FMT=e5m2 / KV_DTYPE=auto。")
        if not _seen["direct"]:
            _seen["direct"] = True
            print(f"[fp8_kv_hcu] 首次真正生效(decode 走厂商 paged_attention): q={tuple(q.shape)} "
                  f"max_seqlen_q={mq} bs={bs} K={tuple(k.shape)} {k.dtype} pid={os.getpid()}", flush=True)
        return orig(*args, **kw)
    # 草稿扩展（max_seqlen_q ≤ 4）：split-KV 专用核；否则预填充走自写 prefill 内核；都不吻合则反量化后走 bf16 厂商
    ext_out = _try_ext(q, k, v, bt, cu_q, kw, mq, bs)
    if ext_out is not None:
        return ext_out
    pp_out = _try_pp(q, k, v, bt, cu_q, kw, mq, bs)
    if pp_out is not None:
        return pp_out
    P = bt.shape[1]
    idx = bt.reshape(-1).long()
    kd = k[idx].to(torch.bfloat16)
    vd = v[idx].to(torch.bfloat16)
    nbt = torch.arange(bs * P, device=bt.device, dtype=torch.int32).view(bs, P)
    if not _seen["dequant"]:
        _seen["dequant"] = True
        print(f"[fp8_kv_hcu] 首次真正生效(prefill 反量化): q={tuple(q.shape)} max_seqlen_q={mq} bs={bs} "
              f"页数={P} K池={tuple(k.shape)} {k.dtype} → 临时 bf16 {tuple(kd.shape)} "
              f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
    kw2 = dict(kw)
    kw2["q"], kw2["k"], kw2["v"], kw2["block_table"] = q, kd, vd, nbt
    return orig(*args[3:], **kw2) if len(args) > 3 else orig(**kw2)


def dispatch_with_kvcache(orig_kv, *a, **kw):
    def fallback(*args, **kwargs):
        cache = kwargs.get("k_cache")
        if cache is not None and cache.dtype == torch.float8_e4m3fn and kwargs.get("block_table") is not None:
            raise RuntimeError("Paged e4m3 draft attention requires a compatible PA kernel")
        return orig_kv(*args, **kwargs)
    k = kw.get("k_cache")
    bt = kw.get("block_table")
    q = kw.get("q")
    if _TRACE and q is not None and len(_route_seen) < 12 and not torch.cuda.is_current_stream_capturing() and kw.get("cache_seqlens") is not None and int(kw["cache_seqlens"][0]) > 10000:
        key = ("kv", tuple(q.shape), None if k is None else str(k.dtype))
        if key not in _route_seen:
            _route_seen.add(key)
            print(f"[fp8_kv_hcu][route] with_kvcache: q={tuple(q.shape)} K={None if k is None else tuple(k.shape)} {None if k is None else k.dtype} "
                  f"cache_seqlens[:2]={kw.get('cache_seqlens')[:2].tolist() if kw.get('cache_seqlens') is not None else None} pid={os.getpid()}", flush=True)
    if (not _OWN) or k is None or bt is None or q is None or k.dtype not in _FP8:
        return fallback(*a, **kw)
    # q 形状 [bs, 1, HQ, HD]；只接管 1 个 kv 头 / hd 256 / page 64 的生产形状
    if q.dim() != 4 or q.shape[1] != 1 or q.shape[2] != 4 or q.shape[3] != 256:
        return fallback(*a, **kw)
    if k.dim() != 4 or k.shape[1] != 1 or k.shape[2] != 64 or k.shape[3] != 256:
        return fallback(*a, **kw)
    cs = kw.get("cache_seqlens")
    if cs is None or kw.get("return_softmax_lse"):
        return fallback(*a, **kw)
    pa = _load_pa(1)
    if pa is None:
        return fallback(*a, **kw)
    try:
        import torch as _t
        bs = q.shape[0]
        out = _t.empty_like(q)
        pa(out.view(bs, 1, 4, 256), q.view(bs, 1, 4, 256), k, kw.get("v_cache"),
           bt.to(_t.int32), cs.to(_t.int32), int(bt.shape[1]), float(kw.get("softmax_scale")))
        if not _seen["own_kv"]:
            _seen["own_kv"] = True
            print(f"[fp8_kv_hcu] ✅ 首次真正生效(草稿步 q=1 自写内核 {_OWN_FMT}): q={tuple(q.shape)} "
                  f"bs={bs} 页表宽 {bt.shape[1]} capturing={_t.cuda.is_current_stream_capturing()} "
                  f"pid={os.getpid()}", flush=True)
        return out
    except Exception as e:
        if not _seen["own_kv_err"]:
            _seen["own_kv_err"] = True
            print(f"[fp8_kv_hcu] ⚠ 草稿步自写内核异常，回退厂商: {type(e).__name__}: {e}", flush=True)
        return fallback(*a, **kw)


def _preflight_kernels():
    """启动预检：确认所需的 .so 存在。

    厂商的 decode fp8 内核只对 e5m2 算对（e4m3 输出幅值约 3.6 倍，2026-09-21 探针实测），
    所以「KV 为 e4m3 而自写内核不可用」这一组合会产生错误结果而非仅仅变慢。
    此处直接让服务启动失败，而不是留到运行时静默回退。
    """
    need = []
    if _OWN:
        need += [f"libpa_{_OWN_FMT}_q{sq}.so" for sq in (4, 1)]
    if _PP_OWN:
        need.append(f"libpp_v5_q{4 if _OWN_FMT == 'e4m3' else 5}.so")
    miss = [f for f in need if not os.path.exists(os.path.join(_PATCH_DIR, f))]
    if not miss:
        return
    msg = (f"[fp8_kv_hcu] 缺少自写内核: {' '.join(miss)}（查找目录 {_PATCH_DIR}）。\n"
           f"           先编译：bash kernels/build.sh")
    if _OWN_FMT == "e4m3":
        raise RuntimeError(msg + "\n           KV 为 e4m3 时厂商内核会算错，不能在缺少自写内核的情况下运行。"
                                 "\n           若只想验证框架路径，请改用 KV_DTYPE=auto（BF16）启动。")
    print(msg + "\n           KV 为 e5m2，将回退厂商内核（结果正确，但长上下文更慢）。", flush=True)


def validate_e4m3_pa():
    if not _OWN or _OWN_FMT != "e4m3":
        raise ValueError("Paged e4m3 KV requires SGLANG_PA_OWN=1 and SGLANG_PA_FMT=e4m3")
    _preflight_kernels()
