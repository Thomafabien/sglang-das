import unittest

from sglang.srt.layers.attention.fla.chunk_o import _should_skip_chunk_o_zero
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGDNNoZeroPolicy(unittest.TestCase):
    def test_enabled_aiter_varlen_path_skips_zero(self):
        self.assertTrue(
            _should_skip_chunk_o_zero(
                enabled=True, use_aiter=True, is_varlen=True
            )
        )

    def test_other_paths_preserve_zero_initialization(self):
        cases = (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        )
        for enabled, use_aiter, is_varlen in cases:
            with self.subTest(
                enabled=enabled, use_aiter=use_aiter, is_varlen=is_varlen
            ):
                self.assertFalse(
                    _should_skip_chunk_o_zero(
                        enabled=enabled,
                        use_aiter=use_aiter,
                        is_varlen=is_varlen,
                    )
                )


if __name__ == "__main__":
    unittest.main()
