import unittest

from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
    _select_aiter_gdn_recurrent_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterGDNRecurrentConfig(unittest.TestCase):
    def test_disabled_preserves_aiter_config_lookup(self):
        self.assertIsNone(
            _select_aiter_gdn_recurrent_config(
                4, 8, is_target_verify=True, enabled=False
            )
        )

    def test_selects_validated_target_verify_shape(self):
        self.assertEqual(
            _select_aiter_gdn_recurrent_config(
                4, 8, is_target_verify=True, enabled=True
            ),
            {"BV": 8, "num_warps": 4},
        )

    def test_non_target_verify_preserves_aiter_config_lookup(self):
        self.assertIsNone(
            _select_aiter_gdn_recurrent_config(
                4, 8, is_target_verify=False, enabled=True
            )
        )

    def test_unvalidated_shapes_preserve_aiter_config_lookup(self):
        for shape in ((2, 8), (4, 4), (8, 8)):
            with self.subTest(shape=shape):
                self.assertIsNone(
                    _select_aiter_gdn_recurrent_config(
                        *shape, is_target_verify=True, enabled=True
                    )
                )

    def test_accepts_explicit_validated_tuning_values(self):
        self.assertEqual(
            _select_aiter_gdn_recurrent_config(
                4,
                8,
                is_target_verify=True,
                enabled=True,
                bv=16,
                num_warps=2,
            ),
            {"BV": 16, "num_warps": 2},
        )


if __name__ == "__main__":
    unittest.main()
