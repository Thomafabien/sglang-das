import unittest

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _can_fuse_m3_sum_gate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestM3FusedSumGatePolicy(unittest.TestCase):
    def test_supported_qwen_moe_path_is_enabled(self):
        self.assertTrue(
            _can_fuse_m3_sum_gate(
                enabled=True,
                is_hcu_backend=True,
                has_shared_gate=True,
                is_deepep=False,
                num_tokens=128,
            )
        )

    def test_unsupported_paths_fall_back(self):
        defaults = dict(
            enabled=True,
            is_hcu_backend=True,
            has_shared_gate=True,
            is_deepep=False,
            num_tokens=128,
        )
        for override in (
            dict(enabled=False),
            dict(is_hcu_backend=False),
            dict(has_shared_gate=False),
            dict(is_deepep=True),
            dict(num_tokens=0),
        ):
            args = defaults | override
            with self.subTest(args=args):
                self.assertFalse(_can_fuse_m3_sum_gate(**args))


if __name__ == "__main__":
    unittest.main()
