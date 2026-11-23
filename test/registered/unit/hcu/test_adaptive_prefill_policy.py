import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.adaptive_prefill_policy import AdaptivePrefillPolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAdaptivePrefillPolicy(unittest.TestCase):
    def test_continuation_sets_long_batch_limit(self):
        with patch.dict(
            "os.environ",
            {"SGLANG_ADAPTIVE_PMR": "1", "SGLANG_ADAPTIVE_PMR_FIXCHUNK": "1",
             "SGLANG_ADAPTIVE_PMR_LONG": "1"},
        ):
            self.assertEqual(AdaptivePrefillPolicy(4).prefill_limit(None, [object()]), 1)

    def test_long_admission_counts_running_and_prefill_requests(self):
        req = SimpleNamespace(origin_input_ids=range(65536))
        with (
            patch.dict("os.environ", {"SGLANG_LONG_ADMIT": "1"}),
            patch(
                "sglang.srt.managers.adaptive_prefill_policy._admission_limits",
                return_value=(2, 65536),
            ),
        ):
            policy = AdaptivePrefillPolicy(4)
            running = SimpleNamespace(reqs=[req])
            self.assertFalse(policy.should_hold(req, running, []))
            self.assertTrue(policy.should_hold(req, running, [req]))
