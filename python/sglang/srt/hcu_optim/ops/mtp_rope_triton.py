"""MTP 草稿层的 RoPE 改走 Triton 融合核（与验证步同一个核），不再落到原生实现的 17 个小核。

依据（2026-09-23 decode 逐位置模板，并发 1 关图 trace）：
  - MRotaryEmbedding.forward_cuda 只在 positions 为 2 维（mrope）时走 triton_mrope_fused；MTP 草稿层拿到的是 1 维 positions，
    落到 forward_native：index 取 cos/sin + Mul×8 + add×4 + Cat×4 共 17 核约 98 µs（草稿扩展 M=4 时 111 µs），每轮 3 次
  - 1 维位置等价于三个分量位置都相同的 mrope：每个频率取的都是 cos_sin_cache[pos]，与 axis_map 无关。
    这里把 positions 扩成 [3, T] 的零步长视图（不拷贝）喂给同一个 Triton 核
数值：原生路径在 bf16 里舍入 3 次（x·cos、y·sin、相减），Triton 用 fp32 中间量只舍入 1 次——只影响草稿，看接受长度哨兵；
改后草稿与验证用同一个 RoPE 核。
"""
import os

_ST = {"first": False, "fb": set()}


def install(mod):
    C = mod.MRotaryEmbedding
    if getattr(C, "_kp_mtp_rope_installed", False):
        return
    orig = C.forward_cuda

    def forward_cuda(self, positions, query, key, fused_set_kv_buffer_arg=None):
        if positions.ndim == 1:
            why = None
            if not self.mrope_section: why = "无 mrope_section"
            elif fused_set_kv_buffer_arg is not None: why = "带 fused_set_kv_buffer_arg"
            elif query.dim() != 2 or key.dim() != 2: why = f"q/k 维数 {query.dim()}/{key.dim()}"
            if why is None:
                if not _ST["first"]:
                    _ST["first"] = True
                    import torch
                    print(f"[mtp_rope_triton] 首次生效: T={positions.shape[0]} capturing={torch.cuda.is_current_stream_capturing()} "
                          f"pid={os.getpid()}", flush=True)
                return self.forward_triton(positions.unsqueeze(0).expand(3, -1), query, key)
            if why not in _ST["fb"]:
                _ST["fb"].add(why)
                print(f"[mtp_rope_triton] 回退原生实现: {why} pid={os.getpid()}", flush=True)
        return orig(self, positions, query, key, fused_set_kv_buffer_arg)

    C.forward_cuda = forward_cuda
    C._kp_mtp_rope_installed = True
    print(f"[mtp_rope_triton] 已安装到 {mod.__name__} (pid={os.getpid()})", flush=True)
