import unittest

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    _select_causal_conv1d_fwd_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCausalConv1dLongPrefillConfig(unittest.TestCase):
    def test_disabled_uses_upstream_default(self):
        self.assertEqual(
            _select_causal_conv1d_fwd_config(24576, 2048, enabled=False),
            (8, 256, 4),
        )

    def test_selects_validated_long_prefill_configs(self):
        cases = (
            (2047, (8, 256, 4)),
            (2048, (16, 512, 4)),
            (6143, (16, 512, 4)),
            (6144, (64, 1024, 2)),
            (24575, (64, 1024, 2)),
            (24576, (128, 1024, 2)),
        )
        for total_tokens, expected in cases:
            with self.subTest(total_tokens=total_tokens):
                self.assertEqual(
                    _select_causal_conv1d_fwd_config(
                        total_tokens, 2048, enabled=True
                    ),
                    expected,
                )

    def test_skips_config_wider_than_hidden_dimension(self):
        self.assertEqual(
            _select_causal_conv1d_fwd_config(24576, 512, enabled=True),
            (16, 512, 4),
        )
        self.assertEqual(
            _select_causal_conv1d_fwd_config(24576, 256, enabled=True),
            (8, 256, 4),
        )


if __name__ == "__main__":
    unittest.main()
