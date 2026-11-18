"""全注意力层的 KV 写入融合：bf16 K/V → fp8 e4m3fn → 分页写入，6 个小核（K/V 各一次 .to(fp8)、loc//64、loc%64、两次 index_put）合成 1 个。

依据（2026-09-23 decode 逐位置模板，并发 1 关图 trace）：每个全注意力层 6 核约 62 µs（验证 M=4），每轮 13 层（验证 10 + 草稿 2 + 扩展 1）
  共 78 核、约 0.78 ms；实现在 memory_pool.py MHATokenToKVPool.set_kv_buffer 的 _kv_layout_hcu_fa 分支
  （K 按 [页,头,64,D] 行写、V 按 [页,头,D,64] 转置写，捕获期 V 走备用流）。
做法：每个 (token, 头) 一个程序，读 K/V（V 可为 qkv 的跨步视图）→ 逐元素转 fp8 → 算页号/页内偏移直接写池。
fp8 转换逐位复刻 c10 fp8e4m3fn_from_fp32_value（RNE，|x| ≥ 480 → NaN 0x7f），全部 65536 个 bf16 位型与 torch .to() 逐位对拍。
**服务实测（2026-09-24，默认关闭）**：离线逐位一致、图内 M=4 由 34 µs 降到 3 µs、eager 发射 147→77 µs，但开启后并发 1 每轮时长
  17.03/17.10 → 17.79/17.67 ms（两个实例都复现，+4%），并发 16 的 TPOT −2%。并发 1 变慢的原因未查明（离线图内与 eager 都更快，
  服务日志无回退）；下一步可做关图 A/B 定位是否与图结构（原写法捕获期 V 走备用流）或 eager 草稿扩展里的 Triton 发射有关。
只在：布局为 hcu_fa、池 dtype 为 float8_e4m3fn、无 k/v scale、无 dcp 掩码、输入为 bf16 [M,H,D] 且末维连续 时接管，其余走原实现。
"""
import os
import torch, triton, triton.language as tl

_ST = {"first": False, "fb": set()}


@triton.jit
def _to_e4m3fn(x):
    """x: fp32 张量 → uint8，照抄 c10::detail::fp8e4m3fn_from_fp32_value（RNE；|x| 的位型 ≥ 1087<<20（即 480）→ NaN 0x7f）。
    常量都写成 int32 范围内的形式，避免 Triton 把字面量提升成 int64 破坏 uint32 回绕语义。"""
    b = x.to(tl.uint32, bitcast=True)
    sign = b >> 31
    a = b & 0x7FFFFFFF
    # 次规格分支（a < 121<<23，即 |x| < 2^-6）：f + 2^14（= 位型 141<<23），再减去 141<<23
    sub = (a.to(tl.float32, bitcast=True) + 16384.0).to(tl.uint32, bitcast=True)
    r_sub = (sub - 1182793728) & 0xFF                       # 1182793728 = 141 << 23
    # 规格分支：指数减 120（(7-127)<<23），加舍入偏置 0x7FFFF 与尾数奇偶位
    mant_odd = (a >> 20) & 1
    r_norm = ((a - 1006632960 + 0x7FFFF + mant_odd) >> 20) & 0xFF   # 1006632960 = 120 << 23
    r = tl.where(a >= 1139802112, 0x7F, tl.where(a < 1015021568, r_sub, r_norm))   # 1087<<20 / 121<<23
    r = r | (sign << 7)
    return r.to(tl.uint8)

@triton.jit
def _kv_write(k_ptr, v_ptr, loc_ptr, kb_ptr, vb_ptr, sk_t, sk_h, sv_t, sv_h, kb_p, kb_h, vb_p, vb_h,
              D: tl.constexpr, PAGE: tl.constexpr):
    t = tl.program_id(0); h = tl.program_id(1)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    p = loc // PAGE; o = loc % PAGE
    d = tl.arange(0, D)
    k = tl.load(k_ptr + t.to(tl.int64) * sk_t + h * sk_h + d).to(tl.float32)
    v = tl.load(v_ptr + t.to(tl.int64) * sv_t + h * sv_h + d).to(tl.float32)
    tl.store(kb_ptr + p * kb_p + h * kb_h + o * D + d, _to_e4m3fn(k))
    tl.store(vb_ptr + p * vb_p + h * vb_h + d * PAGE + o, _to_e4m3fn(v))

def _fused(k, v, loc, kbuf, vbuf):
    M, H, D = k.shape
    PAGE = kbuf.shape[2]
    _kv_write[(M, H)](k, v, loc, kbuf, vbuf, k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                      kbuf.stride(0), kbuf.stride(1), vbuf.stride(0), vbuf.stride(1), D=D, PAGE=PAGE)


def install(mod):
    C = mod.MHATokenToKVPool
    if getattr(C, "_kp_kvw_installed", False):
        return
    orig = C.set_kv_buffer

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                      layer_id_override=None, dcp_kv_mask=None):
        why = None
        if not getattr(mod, "_kv_layout_hcu_fa", False): why = "非 hcu_fa 布局"
        elif self.dtype != torch.float8_e4m3fn or self.store_dtype != torch.uint8: why = f"池 dtype {self.dtype}"
        elif k_scale is not None or v_scale is not None: why = "带 k/v scale"
        elif dcp_kv_mask is not None: why = "dcp 掩码"
        elif (cache_k.dtype != torch.bfloat16 or cache_v.dtype != torch.bfloat16 or cache_k.dim() != 3 or cache_v.dim() != 3
              or cache_k.stride(2) != 1 or cache_v.stride(2) != 1): why = f"输入 {cache_k.dtype}/{tuple(cache_k.shape)}"
        if why is None:
            loc, _, _ = mod.unwrap_write_loc(loc_info)
            mod.maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
            lid = (layer_id_override if layer_id_override is not None else layer.layer_id) - self.start_layer
            kb, vb = self.k_buffer[lid], self.v_buffer[lid]
            M, H, D = cache_k.shape
            if (kb.dim() == 4 and vb.dim() == 4 and kb.is_contiguous() and vb.is_contiguous() and kb.shape[1] == H and kb.shape[3] == D
                    and vb.shape[2] == D and vb.shape[3] == kb.shape[2] == self.page_size):
                if M > 0:
                    _fused(cache_k, cache_v, loc, kb, vb)
                if not _ST["first"]:
                    _ST["first"] = True
                    print(f"[kv_write_fused] 首次生效: M={M} H={H} D={D} page={self.page_size} "
                          f"capturing={torch.cuda.is_current_stream_capturing()} pid={os.getpid()}", flush=True)
                return
            why = f"池形状 K{tuple(kb.shape)} V{tuple(vb.shape)}"
        if why not in _ST["fb"]:
            _ST["fb"].add(why)
            print(f"[kv_write_fused] 回退原实现: {why} pid={os.getpid()}", flush=True)
        return orig(self, layer, loc_info, cache_k, cache_v, k_scale, v_scale, layer_id_override, dcp_kv_mask)

    C.set_kv_buffer = set_kv_buffer
    C._kp_kvw_installed = True
    print(f"[kv_write_fused] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)
