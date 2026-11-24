import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers import first_token_policy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFirstTokenPolicy(CustomTestCase):
    def test_modes_and_one_shot_decode(self):
        batch = SimpleNamespace(
            forward_mode=Mock(is_extend=Mock(return_value=True)),
            reqs=[Mock(output_ids=[7], finished=Mock(return_value=False))],
            is_empty=Mock(return_value=False),
            is_prefill_only=False,
        )
        decode = SimpleNamespace(forward_mode=Mock(is_extend=Mock(return_value=False)))
        for mode in range(4):
            with self.subTest(mode=mode):
                policy = first_token_policy.FirstTokenPolicy(mode)
                tokenizer = Mock(decode=Mock(return_value="\ufffd"))
                self.assertEqual(policy.should_disable_overlap(batch, batch, True), mode == 1)
                self.assertEqual(policy.should_flush(batch, True), mode >= 2)
                policy.after_flush(batch, tokenizer)
                self.assertEqual(policy.take_decode_turn(batch, False), mode == 3)
                self.assertFalse(policy.take_decode_turn(batch, False))
                self.assertEqual(policy.should_flush(decode, True), mode == 3)
                self.assertFalse(policy.should_flush(decode, True))

    def test_decode_requires_runnable_batch_without_continuation(self):
        for empty, prefill_only, chunked in ((True, False, False),
                                              (False, True, False),
                                              (False, False, True)):
            with self.subTest(empty=empty, prefill_only=prefill_only, chunked=chunked):
                policy = first_token_policy.FirstTokenPolicy(3)
                policy.need_decode = True
                batch = Mock(is_empty=Mock(return_value=empty), is_prefill_only=prefill_only)
                self.assertFalse(policy.take_decode_turn(batch, chunked))
                self.assertFalse(policy.need_decode)
                self.assertFalse(policy.flush_next)

    def test_no_source_fingerprint_gate(self):
        source = inspect.getsource(first_token_policy)
        for fingerprint_api in ("getsource", "hashlib", "md5"):
            self.assertNotIn(fingerprint_api, source.lower())


if __name__ == "__main__":
    unittest.main()
