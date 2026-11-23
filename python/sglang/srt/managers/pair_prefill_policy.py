"""Plan adjacent prefill chunks and retain a lookahead plan when unpaired."""

from pathlib import Path

from sglang.srt.environ import envs


class PairPrefillPolicy:
    def __init__(self, runner):
        self.runner = runner
        self.carry = None
        self.min_tokens = envs.SGLANG_PAIR_PREFILL_MIN_TOKENS.get()
        self.allow_logprob = envs.SGLANG_PAIR_PREFILL_ALLOW_LOGPROB.get()
        self.off_file = Path(envs.SGLANG_PAIR_PREFILL_OFF_FILE.get())

    @classmethod
    def from_env(cls):
        if not envs.SGLANG_PAIR_PREFILL.get():
            return None
        from sglang.srt.hcu_optim.sched import pair_prefill

        return cls(pair_prefill.run_pair) if pair_prefill.ENABLED else None

    def take_plan(self):
        plan, self.carry = self.carry, None
        return plan

    def _matches(self, batch, req):
        return (
            batch is not None and req is not None and batch.forward_mode.is_extend()
            and len(batch.reqs) == 1 and batch.reqs[0] is req
            and (self.allow_logprob or not batch.return_logprob)
            and batch.extend_num_tokens > self.min_tokens
        )

    def plan_second(self, first, chunked_req, running_batch, get_next_plan):
        if self.off_file.exists() or not self._matches(first, chunked_req):
            return None, None
        plan = get_next_plan(running_batch=running_batch, last_batch=first.copy())
        if self._matches(plan.batch_to_run, chunked_req):
            return plan, plan.batch_to_run
        self.carry = plan
        return plan, None

    def run(self, scheduler, first, second):
        return self.runner(scheduler, first, second)
