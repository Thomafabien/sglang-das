import unittest

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    _can_use_fast_hcu_moe_align,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFastHCUMoeAlignPolicy(unittest.TestCase):
    def test_single_rank_non_ep_path_is_enabled(self):
        self.assertTrue(
            _can_use_fast_hcu_moe_align(
                enabled=True,
                ep_size=1,
                enable_eplb=False,
                redundant_experts=0,
                is_deepep=False,
            )
        )

    def test_unsafe_modes_fall_back(self):
        cases = (
            dict(enabled=False),
            dict(ep_size=2),
            dict(enable_eplb=True),
            dict(redundant_experts=1),
            dict(is_deepep=True),
        )
        defaults = dict(
            enabled=True,
            ep_size=1,
            enable_eplb=False,
            redundant_experts=0,
            is_deepep=False,
        )
        for override in cases:
            args = defaults | override
            with self.subTest(args=args):
                self.assertFalse(_can_use_fast_hcu_moe_align(**args))


if __name__ == "__main__":
    unittest.main()
