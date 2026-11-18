"""长文本分词并行化：独立进程池（spawn）+ 安全切分点，结果与整条编码逐位一致。

背景：TTFT 里「入口→入队」的残差 86%–91% 是服务端分词（65536 档约 426 ms，131072 档约 915 ms），
      且单条长文本的分词是单线程的。HF tokenizers 的批编码并行依赖 Rust 侧 Rayon 线程池，
      该线程池不随 fork 存活，而本服务的分词进程是 fork 出来的 —— 所以这里不用线程池，
      改用 spawn 方式另起的工作进程，各自持有一份分词器。

切分点取「字母或数字」之后、紧跟「空白或十进制数字」之前的位置。依据是逐条核对 Qwen2 预分词正则的 7 个分支：
      能把空格含在匹配中部或尾部的只有三个空白分支（\\s*[\\r\\n]+、\\s+(?!\\S)、\\s+），它们只可能从空白字符起始；
      空格前一字符为字母/数字时，前一个匹配必在空格前结束，故此处必为预分词边界。
      BPE 合并不跨预分词边界；NFC 归一化不跨空格组合；前一字符限定为字母数字，也避开了特殊 token（以 '>' 结尾）的 rstrip 语义。
      找不到足够切分点时退回原路径。
自证：首次命中时额外做一次整条编码逐位对拍并打印耗时；不一致则永久停用并告警。
"""
import asyncio
import os
import re
import time

_MIN_CHARS = int(os.environ.get("SGLANG_TOK_POOL_MIN_CHARS", "16000"))
_NSEG = int(os.environ.get("SGLANG_TOK_POOL_SEGS", "12"))
_SPLIT_RE = re.compile(r"(?<=[^\W_])(?=[\s\d])")
_st = {"pool": None, "verified": False, "dead": False, "n": 0, "warming": False}
_W = {}


def _winit(path, trust):
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))      # 父进程可能被绑在单个 NUMA 节点上
    except Exception:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _W["tk"] = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)


def _wenc(piece):
    return _W["tk"]([piece])["input_ids"][0]


def _split(text, nseg):
    """返回切分后的片段列表；安全切分点不足时返回 None。
    安全点 = 「字母或数字」之后、紧跟「空白或十进制数字」之前的位置（见文件头的正则分支分析；
    数字一条的依据：\\p{N} 每次只匹配单个字符，且 \\p{L}+ 分支的可选前缀不含数字）。
    压测文本由连续 token 编号解码而来，会有大段无空白区域，故先收集全部安全点，再取最接近等分目标的那些。"""
    n = len(text)
    cand = [m.start() for m in _SPLIT_RE.finditer(text)]
    if len(cand) < nseg - 1:
        return None
    import bisect
    cuts = []
    for k in range(1, nseg):
        tgt = n * k // nseg
        i = bisect.bisect_left(cand, tgt)
        best = min((c for c in cand[max(0, i - 1):i + 1]), key=lambda c: abs(c - tgt))
        if (not cuts or best > cuts[-1]) and 0 < best < n:
            cuts.append(best)
    if not cuts:
        return None
    parts, a = [], 0
    for c in cuts:
        parts.append(text[a:c]); a = c
    parts.append(text[a:])
    return parts


def _get_pool(tm):
    if _st["pool"] is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        path = tm.server_args.tokenizer_path or tm.server_args.model_path
        _st["pool"] = ProcessPoolExecutor(max_workers=_NSEG, mp_context=mp.get_context("spawn"),
                                          initializer=_winit, initargs=(path, bool(tm.server_args.trust_remote_code)))
    return _st["pool"]


def install(mod):
    TM = getattr(mod, "TokenizerManager", None)
    if TM is None or getattr(TM, "_tok_pool_installed", False):
        return
    orig = TM._tokenize_texts

    async def _tokenize_texts(self, texts, is_cross_encoder=False):
        if _st["dead"] or is_cross_encoder or self.tokenizer is None:
            return await orig(self, texts, is_cross_encoder)
        if not _st["warming"]:                       # 任意首个请求（含服务自带预热）触发后台起池，不等待
            _st["warming"] = True
            try:
                # spawn 的子进程默认会把父进程的 __main__（sglang.launch_server）重新导入一遍，既慢又占内存。
                # 起池期间临时摘掉 __main__ 的 __spec__/__file__，子进程就只导入本模块与 transformers。
                import sys
                main = sys.modules.get("__main__")
                saved = {k: getattr(main, k) for k in ("__spec__", "__file__") if hasattr(main, k)}
                try:
                    for k in saved:
                        if k == "__spec__": main.__spec__ = None
                        else: delattr(main, k)
                    pool = _get_pool(self)
                    _st["warm_futs"] = [pool.submit(_wenc, "warm up") for _ in range(_NSEG)]
                finally:
                    for k, v in saved.items(): setattr(main, k, v)
            except Exception as e:
                _st["dead"] = True
                print(f"[tok_pool] ⚠ 进程池创建失败，停用: {type(e).__name__}: {e}", flush=True)
        if not isinstance(texts, str) or len(texts) < _MIN_CHARS or _st["dead"]:
            n = len(texts) if hasattr(texts, "__len__") else -1
            if (not isinstance(texts, str) and n > 0) or _st["dead"]:
                _st["fb"] = _st.get("fb", 0) + 1
                if _st["fb"] <= 5:
                    inner = type(texts[0]).__name__ if n > 0 and not isinstance(texts, str) else "-"
                    print(f"[tok_pool] 回退原路径：输入类型 {type(texts).__name__}[{inner}] 长度 {n} dead={_st['dead']}", flush=True)
            return await orig(self, texts, is_cross_encoder)
        parts = _split(texts, _NSEG)
        if parts is None:
            _st["fb2"] = _st.get("fb2", 0) + 1
            if _st["fb2"] <= 5:
                print(f"[tok_pool] 回退原路径：{len(texts)} 字符内找不到 {_NSEG - 1} 个安全切分点", flush=True)
            return await orig(self, texts, is_cross_encoder)
        t0 = time.perf_counter()
        pool = _get_pool(self)
        futs = [asyncio.wrap_future(pool.submit(_wenc, p)) for p in parts]
        segs = await asyncio.gather(*futs)
        ids = [t for s in segs for t in s]
        t1 = time.perf_counter()
        if not _st["verified"]:
            ref, _ = await orig(self, texts, is_cross_encoder)
            t2 = time.perf_counter()
            if list(ref) != ids:
                _st["dead"] = True
                print(f"[tok_pool] ⚠ 逐位对拍不一致（{len(ids)} 对 {len(ref)}），永久停用，退回原路径", flush=True)
                return ref, None
            _st["verified"] = True
            print(f"[tok_pool] ✅ 首次真正生效并通过逐位对拍: {len(texts)} 字符 / {len(ids)} token, "
                  f"切 {len(parts)} 段, 进程池 {1e3*(t1-t0):.1f} ms 对 整条 {1e3*(t2-t1):.1f} ms "
                  f"({(t2-t1)/(t1-t0):.2f}x)", flush=True)
        _st["n"] += 1
        if _st["n"] <= int(os.environ.get("SGLANG_TOK_POOL_LOG_N", "20")):
            print(f"[tok_pool] 第 {_st['n']} 次: {len(texts)} 字符 / {len(ids)} token, 进程池 {1e3*(t1-t0):.1f} ms", flush=True)
        return ids, None

    TM._tokenize_texts = _tokenize_texts
    TM._tok_pool_installed = True
    print(f"[tok_pool] 已安装 (阈值 {_MIN_CHARS} 字符, 切 {_NSEG} 段, spawn 进程池)", flush=True)
