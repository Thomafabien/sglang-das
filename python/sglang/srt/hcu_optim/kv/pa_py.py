#!/usr/bin/env python3
"""自写分页 decode 注意力的 Python 包装（ctypes 调 HIP 共享库），并与参考数据对拍。
用法: python3 pa_py.py <参考目录> [e4m3|e5m2]
"""
import sys, os, json, ctypes, math, numpy as np, torch

try:                                   # 作为源码包导入时，.so 位于 hcu_optim/lib
    from .. import PATCH_DIR as LIBDIR
except ImportError:                     # 直接以脚本方式运行（离线对拍）时退回同级目录
    LIBDIR = os.path.dirname(os.path.abspath(__file__))

class PagedAttn:
    """一次构造、多次调用；工作区显存预分配，便于在 CUDA graph 内复用。"""
    def __init__(self, fmt="e4m3", libdir=LIBDIR, sq=4):
        if fmt not in ("e4m3", "e5m2") or sq not in (1, 4):
            raise ValueError(f"Unsupported PA configuration: fmt={fmt!r}, sq={sq}")
        cand = [f"libpa_{fmt}_q{sq}.so"]
        if sq == 4:
            # Legacy libraries use the four-query ABI.
            cand.append(f"libpa_{fmt}.so")
        for c in cand:
            fp = os.path.join(libdir, c)
            if os.path.exists(fp):
                self.lib = ctypes.CDLL(fp); break
        else:
            raise FileNotFoundError(f"未找到 {cand} @ {libdir}")
        self.sq = sq
        self.nq = 4 * sq
        self.lib.pa_decode.restype = None
        self.lib.pa_decode.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_void_p] * 4 + \
                                      [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        self.lib.pa_pick_nsplit.restype = ctypes.c_int
        self.lib.pa_pick_nsplit.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.pa_qfmt.restype = ctypes.c_int
        self.fmt = fmt
        self.qfmt = self.lib.pa_qfmt()
        self._ws = {}          # (bs, nsplit) -> (o_part, ml)

    def pick_nsplit(self, npages, bs):
        return int(self.lib.pa_pick_nsplit(int(npages), int(bs)))

    def workspace(self, bs, nsplit, device):
        key = (bs, nsplit)
        if key not in self._ws:
            NQ, HD = self.nq, 256
            self._ws[key] = (torch.empty(bs * nsplit * NQ * HD, dtype=torch.float32, device=device),
                             torch.empty(bs * nsplit * NQ * 2, dtype=torch.float32, device=device))
        return self._ws[key]

    def __call__(self, out, q, kc, vc, bt, slen, npages, scale, nsplit=None, stream=None):
        """out/q: [bs, 4, 4, 256] bf16；kc: [pages,1,64,256] fp8；vc: [pages,1,256,64] fp8
           bt: [bs, n_per] int32；slen: [bs] int32"""
        bs, n_per = bt.shape
        if nsplit is None: nsplit = self.pick_nsplit(npages, bs)
        o_part, ml = self.workspace(bs, nsplit, q.device)
        st = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        self.lib.pa_decode(
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(kc.data_ptr()), ctypes.c_void_p(vc.data_ptr()),
            ctypes.c_void_p(bt.data_ptr()), ctypes.c_void_p(slen.data_ptr()),
            ctypes.c_void_p(o_part.data_ptr()), ctypes.c_void_p(ml.data_ptr()),
            ctypes.c_int(bs), ctypes.c_int(n_per), ctypes.c_int(npages), ctypes.c_int(nsplit),
            ctypes.c_float(scale), ctypes.c_void_p(st))
        return out


if __name__ == "__main__":
    REF = sys.argv[1]
    FMT = sys.argv[2] if len(sys.argv) > 2 else "e4m3"
    man = json.load(open(os.path.join(REF, "manifest.json")))
    HQ, HK, HD, PAGE, SQ = man["HQ"], man["HK"], man["HD"], man["PAGE"], man["SQ"]
    CTX, BS, scale = man["ctx"], man["bs"], man["scale"]
    dev = "cuda"
    DT = torch.float8_e4m3fn if FMT == "e4m3" else torch.float8_e5m2

    def load(name, dt, shape):
        a = np.fromfile(os.path.join(REF, name + ".bin"), dtype=dt)
        return torch.from_numpy(a).to(dev).view(*shape)

    q  = load("q", np.int16, man["items"]["q"]["shape"]).view(torch.bfloat16)
    kc = load("kcache", np.uint8, man["items"]["kcache"]["shape"]).view(DT)
    vc = load("vcache", np.uint8, man["items"]["vcache"]["shape"]).view(DT)
    bt = load("block_table", np.int32, man["items"]["block_table"]["shape"])
    sl = load("seqlens", np.int32, man["items"]["seqlens"]["shape"])
    ref = load("out", np.int16, man["items"]["out"]["shape"]).view(torch.bfloat16)
    out = torch.zeros_like(ref)
    npages = math.ceil(CTX / PAGE)

    pa = PagedAttn(FMT)
    ns = pa.pick_nsplit(npages, BS)
    print(f"# 库 libpa_{FMT}.so  QFMT={pa.qfmt}  ctx={CTX} bs={BS} 页数/序列={npages}  选中 nsplit={ns}")

    pa(out, q, kc, vc, bt, sl, npages, scale)
    torch.cuda.synchronize()
    d = (out.float() - ref.float())
    rms = (d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    print(f"# 对参考输出：最大差 {d.abs().max():.3e}  相对 RMS {rms:.3e}  "
          f"逐位不等 {(out.view(torch.int16) != ref.view(torch.int16)).sum().item()}/{ref.numel()}")
    print(f"# 前 4 个（本库 / 参考）: " + "  ".join(
        f"{out.flatten()[i].item():.6f}/{ref.flatten()[i].item():.6f}" for i in range(4)))

    # 计时（与离线 opkit graph_time 同口径：CUDA graph 内重放 R 次取中位）
    import statistics
    for _ in range(6): pa(out, q, kc, vc, bt, sl, npages, scale)
    torch.cuda.synchronize()
    R = 20
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): pa(out, q, kc, vc, bt, sl, npages, scale)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(R): pa(out, q, kc, vc, bt, sl, npages, scale)
    torch.cuda.synchronize()
    ts = []
    for _ in range(9):
        g.replay(); torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / R)
    med = statistics.median(ts)
    print(f"# 性能 {med*1000:.1f} us  极差 {(min(ts)/med-1)*100:+.1f}%/{(max(ts)/med-1)*100:+.1f}%  "
          f"(图内 R={R} × 9 轮) —— **CUDA graph 可捕获**")
