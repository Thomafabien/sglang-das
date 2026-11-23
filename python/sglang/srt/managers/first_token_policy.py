"""Ordering decisions for first-token prioritization in the overlap loop."""


class FirstTokenPolicy:
    def __init__(self, mode):
        self.mode = mode
        self.need_decode = False
        self.flush_next = False
        self.token_incomplete = {}

    def should_disable_overlap(self, batch, last_batch, has_results):
        return (
            self.mode == 1 and has_results and batch is not None
            and last_batch is not None and batch.forward_mode.is_extend()
            and last_batch.forward_mode.is_extend()
        )

    def should_flush(self, last_batch, has_results):
        if (
            self.mode >= 2 and has_results and last_batch is not None
            and (last_batch.forward_mode.is_extend() or self.flush_next)
        ):
            self.flush_next = False
            return True
        return False

    def after_flush(self, last_batch, tokenizer):
        if self.mode < 3 or tokenizer is None or not last_batch.forward_mode.is_extend():
            return
        for req in last_batch.reqs:
            if len(req.output_ids) == 1 and not req.finished():
                token = req.output_ids[0]
                if token not in self.token_incomplete:
                    try:
                        text = tokenizer.decode([token])
                        self.token_incomplete[token] = not text or "�" in text
                    except Exception:
                        self.token_incomplete[token] = False
                if self.token_incomplete[token]:
                    self.need_decode = True
                    break

    def take_decode_turn(self, running_batch, has_chunked_req):
        pending = self.need_decode
        self.need_decode = False
        if (
            pending and not has_chunked_req and not running_batch.is_empty()
            and not running_batch.is_prefill_only
        ):
            self.flush_next = True
            return True
        return False
