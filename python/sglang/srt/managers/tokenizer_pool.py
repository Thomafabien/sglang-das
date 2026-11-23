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
import logging
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)
_SPLIT_RE = re.compile(r"(?<=[^\W_])(?=[\s\d])")
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


class TokenizerPool:
    def __init__(self, *, path, trust_remote_code, min_chars, segments, log_count):
        self.path = path
        self.trust_remote_code = trust_remote_code
        self.min_chars = min_chars
        self.segments = segments
        self.log_count = log_count
        self.pool = None
        self.disabled = False
        self.verified = False
        self.calls = 0
        self.verify_lock = asyncio.Lock()

    def _start(self):
        main = sys.modules.get("__main__")
        saved = {
            key: getattr(main, key)
            for key in ("__spec__", "__file__")
            if hasattr(main, key)
        }
        try:
            # Spawn workers without re-importing the server launcher.
            for key in saved:
                if key == "__spec__":
                    main.__spec__ = None
                else:
                    delattr(main, key)
            self.pool = ProcessPoolExecutor(
                max_workers=self.segments,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_winit,
                initargs=(self.path, self.trust_remote_code),
            )
            for _ in range(self.segments):
                self.pool.submit(_wenc, "warm up")
        finally:
            for key, value in saved.items():
                setattr(main, key, value)

    def _disable(self, reason):
        self.disabled = True
        logger.warning("Tokenizer pool disabled: %s", reason)
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    async def tokenize(self, texts, is_cross_encoder, fallback):
        if self.disabled or is_cross_encoder:
            return await fallback(texts, is_cross_encoder)

        try:
            if self.pool is None:
                self._start()
            parts = (
                _split(texts, self.segments)
                if isinstance(texts, str) and len(texts) >= self.min_chars
                else None
            )
            if parts is not None:
                start = time.perf_counter()
                segments = await asyncio.gather(
                    *(asyncio.wrap_future(self.pool.submit(_wenc, part)) for part in parts)
                )
                ids = [token for segment in segments for token in segment]
        except Exception as exc:
            self._disable(exc)
            return await fallback(texts, is_cross_encoder)

        if parts is None or self.disabled:
            return await fallback(texts, is_cross_encoder)

        async with self.verify_lock:
            if self.disabled:
                return await fallback(texts, is_cross_encoder)
            if not self.verified:
                reference = await fallback(texts, is_cross_encoder)
                if list(reference[0]) != ids:
                    self._disable("parallel token IDs differ from the reference")
                    return reference
                self.verified = True
                logger.info("Tokenizer pool passed its first token-by-token comparison")

        self.calls += 1
        if self.calls <= self.log_count:
            logger.info(
                "Tokenizer pool encoded %d characters in %.1f ms",
                len(texts),
                (time.perf_counter() - start) * 1000,
            )
        return ids, None
