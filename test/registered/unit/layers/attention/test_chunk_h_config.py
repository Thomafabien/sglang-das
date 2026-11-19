import unittest

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    _select_aiter_chunk_h_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterChunkHConfig(unittest.TestCase):
    def test_disabled_preserves_aiter_config_lookup(self):
        self.assertIsNone(
            _select_aiter_chunk_h_config(128, 128, 64, 8, enabled=False)
        )

    def test_selects_validated_qwen36_shape(self):
        self.assertEqual(
            _select_aiter_chunk_h_config(128, 128, 64, 8, enabled=True),
            {"BV": 16, "num_warps": 4, "num_stages": 2},
        )

    def test_unvalidated_shapes_preserve_aiter_config_lookup(self):
        for shape in (
            (64, 128, 64, 8),
            (128, 64, 64, 8),
            (128, 128, 32, 8),
            (128, 128, 64, 4),
        ):
            with self.subTest(shape=shape):
                self.assertIsNone(
                    _select_aiter_chunk_h_config(*shape, enabled=True)
                )

    def test_accepts_explicit_validated_tuning_values(self):
        self.assertEqual(
            _select_aiter_chunk_h_config(
                128,
                128,
                64,
                8,
                enabled=True,
                bv=8,
                num_warps=2,
                num_stages=3,
            ),
            {"BV": 8, "num_warps": 2, "num_stages": 3},
        )


if __name__ == "__main__":
    unittest.main()
