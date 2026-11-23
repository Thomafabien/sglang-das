"""Optional request limits for one prefill admission batch."""

import time
from pathlib import Path

from sglang.srt.environ import envs

_OVERRIDE_FILES = (Path("/tmp/long_admit_k"), Path("/tmp/long_admit_thresh"))
_next_poll = 0.0
_overrides = (None, None)


def _read_override(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _admission_limits():
    global _next_poll, _overrides
    now = time.monotonic()
    if now >= _next_poll:
        _next_poll = now + envs.SGLANG_LONG_ADMIT_POLL.get()
        _overrides = tuple(map(_read_override, _OVERRIDE_FILES))
    limit, threshold = _overrides
    return (
        envs.SGLANG_LONG_ADMIT_K.get() if limit is None else limit,
        envs.SGLANG_LONG_ADMIT_THRESH.get() if threshold is None else threshold,
    )


class AdaptivePrefillPolicy:
    def __init__(self, base_limit):
        self.adaptive_pmr = envs.SGLANG_ADAPTIVE_PMR.get()
        self.long_admit = envs.SGLANG_LONG_ADMIT.get()
        self.base_limit = base_limit
        self.is_long_batch = None

    def should_hold(self, req, running_batch, admitted):
        if not self.long_admit:
            return False
        limit, threshold = _admission_limits()
        if limit >= 1000 or len(req.origin_input_ids) < threshold:
            return False
        running = running_batch.reqs if running_batch is not None else ()
        active = sum(len(r.origin_input_ids) >= threshold for r in running)
        active += sum(len(r.origin_input_ids) >= threshold for r in admitted)
        return active >= limit

    def prefill_limit(self, req, admitted):
        if not self.adaptive_pmr:
            return self.base_limit
        if self.is_long_batch is None:
            if admitted:
                if envs.SGLANG_ADAPTIVE_PMR_FIXCHUNK.get():
                    self.is_long_batch = True
            else:
                remaining = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
                self.is_long_batch = remaining >= envs.SGLANG_ADAPTIVE_PMR_THRESH.get()
        if self.is_long_batch:
            return envs.SGLANG_ADAPTIVE_PMR_LONG.get()
        if self.base_limit is None or envs.SGLANG_ADAPTIVE_PMR_SHORT.is_set():
            return envs.SGLANG_ADAPTIVE_PMR_SHORT.get()
        return self.base_limit
