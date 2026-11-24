import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hcu_optim.sched import pair_prefill
from sglang.srt.managers.pair_prefill_policy import PairPrefillPolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FlashAttentionBackend:
    pass


class FakeGreenlet:
    def __init__(self, execute):
        self.execute = execute
        self.dead = False

    def switch(self):
        try:
            self.execute()
        finally:
            self.dead = True

    @staticmethod
    def getcurrent():
        return object()


class TestPairPrefillPolicy(CustomTestCase):
    def test_disabled_and_idle_hooks(self):
        with (
            patch.dict("os.environ", {"SGLANG_PAIR_PREFILL": "0"}),
            patch.object(pair_prefill, "S", pair_prefill._State()),
            patch.object(pair_prefill.torch, "cuda") as cuda,
            patch.object(pair_prefill, "disable_handoff_in_current") as disable,
        ):
            self.assertIsNone(PairPrefillPolicy.from_env())
            self.assertFalse(pair_prefill.uses_private_eager_input())
            self.assertFalse(pair_prefill.requires_safe_alignment())
            self.assertIsNone(pair_prefill.wait_for_exclusive_attention(FlashAttentionBackend()))
            self.assertIsNone(pair_prefill.prepare_draft_extend())
            cuda.current_stream.assert_not_called()
            disable.assert_not_called()

    def test_active_hooks_wait_for_the_correct_lane(self):
        stream_a, event_b = object(), object()
        state = pair_prefill._State()
        state.active, state.role = True, "A"
        state.ev_comm, state.stream_A = {"B": event_b}, stream_a
        state.handoff = {"A": True, "B": True}
        with (
            patch.object(pair_prefill, "S", state),
            patch.object(pair_prefill, "_NSTREAMS", 2),
            patch.object(pair_prefill, "_EXCL_BACKENDS", {"FlashAttentionBackend"}),
            patch.object(pair_prefill.torch, "cuda") as cuda,
        ):
            self.assertTrue(pair_prefill.uses_private_eager_input())
            stream = cuda.current_stream.return_value
            pair_prefill.wait_for_exclusive_attention(object())
            stream.wait_event.assert_not_called()
            pair_prefill.wait_for_exclusive_attention(FlashAttentionBackend())
            stream.wait_event.assert_called_once_with(event_b)
            stream.wait_event.reset_mock()
            pair_prefill.prepare_draft_extend()
            self.assertFalse(state.handoff["A"])
            stream.wait_event.assert_called_once_with(event_b)
            state.role = "B"
            pair_prefill.prepare_draft_extend()
            self.assertFalse(state.handoff["B"])
            stream.wait_stream.assert_called_once_with(stream_a)

    def test_safe_alignment_is_independent_of_moe_overlap(self):
        state = pair_prefill._State()
        state.active = True
        for overlap in (False, True):
            for safe, override in ((False, False), (True, True), (True, False)):
                with (
                    self.subTest(overlap=overlap, safe=safe, override=override),
                    patch.object(pair_prefill, "S", state),
                    patch.object(pair_prefill, "_MOE_OVERLAP", overlap),
                    patch.object(pair_prefill, "_SAFE_ALIGN", safe),
                    patch.object(pair_prefill.os.path, "exists", return_value=override),
                ):
                    self.assertEqual(
                        pair_prefill.requires_safe_alignment(), safe and not override
                    )

    def test_unpaired_lookahead_is_returned_exactly_once(self):
        req = object()
        first = Mock(
            forward_mode=Mock(is_extend=Mock(return_value=True)),
            reqs=[req], return_logprob=False, extend_num_tokens=4096,
        )
        lookahead = SimpleNamespace(batch_to_run=None)
        get_plan = Mock(return_value=lookahead)
        policy = PairPrefillPolicy(Mock())
        policy.min_tokens = 2432
        policy.off_file = Mock(exists=Mock(return_value=False))
        running = object()
        self.assertEqual(policy.plan_second(first, req, running, get_plan), (lookahead, None))
        get_plan.assert_called_once_with(running_batch=running, last_batch=first.copy.return_value)
        self.assertIs(policy.take_plan(), lookahead)
        self.assertIsNone(policy.take_plan())

    def test_forward_context_restored_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                original = object()
                context = SimpleNamespace(value=original)
                state = pair_prefill._State()
                state.stream_B, state.logged = Mock(), True
                stream, stream_context = Mock(), object()
                sched = SimpleNamespace(forward_stream=stream, forward_stream_ctx=stream_context)
                first, second = object(), object()

                def run_batch(batch):
                    self.assertIsNone(context.value)
                    context.value = batch
                    if fail and batch is second:
                        raise RuntimeError("forward failed")
                    return batch

                sched.run_batch = Mock(side_effect=run_batch)
                with (
                    patch.dict(sys.modules, {"greenlet": SimpleNamespace(greenlet=FakeGreenlet)}),
                    patch.object(pair_prefill, "S", state),
                    patch.object(pair_prefill, "_globals", return_value=[(context, "value")]),
                    patch.object(pair_prefill, "_DEBUG_MODE", ""),
                    patch.object(pair_prefill, "_NSTREAMS", 1),
                    patch.object(pair_prefill, "_TIMING", 0),
                    patch.object(pair_prefill.torch, "cuda") as cuda,
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "forward failed"):
                            pair_prefill.run_pair(sched, first, second)
                    else:
                        self.assertEqual(pair_prefill.run_pair(sched, first, second), (first, second))
                    cuda.set_stream.assert_called_with(cuda.current_stream.return_value)
                self.assertIs(context.value, original)
                self.assertIs(sched.forward_stream, stream)
                self.assertIs(sched.forward_stream_ctx, stream_context)
                self.assertFalse(state.active)
                self.assertIsNone(state.role)


if __name__ == "__main__":
    unittest.main()
