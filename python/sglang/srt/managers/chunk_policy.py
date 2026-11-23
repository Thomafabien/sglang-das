"""Select a temporary prefill chunk size for the next admission batch."""

import logging
from pathlib import Path

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_OFF = Path("/tmp/achunk_off")
_NO_EVEN = Path("/tmp/achunk_noeven")


def parse_chunk_tiers(value):
    tiers = []
    for item in value.split(","):
        if item.strip():
            threshold, size = map(int, item.split(":"))
            if threshold <= 0 or size <= 0:
                raise ValueError("chunk thresholds and sizes must be positive")
            tiers.append((threshold, size))
    return tuple(sorted(tiers))


class AdaptiveChunkPolicy:
    def __init__(self, *, pair_prefill):
        self.pair_prefill = pair_prefill
        self.even = envs.SGLANG_ADAPTIVE_CHUNK_EVEN.get()
        self.tiers = parse_chunk_tiers(envs.SGLANG_ADAPTIVE_CHUNK_TIERS.get())

    @classmethod
    def from_env(cls, *, pair_prefill):
        if not envs.SGLANG_ADAPTIVE_CHUNK.get():
            return None
        try:
            return cls(pair_prefill=pair_prefill)
        except ValueError as exc:
            logger.warning("Adaptive chunk disabled: %s", exc)
            return None

    def choose(self, *, base, chunked_req, waiting_queue, page_size, max_tokens):
        if not base or base < 0:
            return base
        req = chunked_req if chunked_req is not None else (
            waiting_queue[0] if waiting_queue else None
        )
        if req is None or _OFF.exists():
            return base
        length = len(req.origin_input_ids)
        size = max((size for threshold, size in self.tiers if length >= threshold),
                   default=base)
        size = max(base, size)
        if self.even and length > size and not _NO_EVEN.exists():
            if length > 1.15 * size:
                step = 2 if self.pair_prefill else 1
                chunks = step
                while -(-length // chunks) > 1.15 * size:
                    chunks += step
                size = -(-length // chunks)
            else:
                size = length
            size = -(-size // page_size) * page_size
        if max_tokens is not None:
            size = min(size, max(base, max_tokens))
        return size
