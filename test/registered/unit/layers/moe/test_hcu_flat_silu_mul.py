import unittest

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _should_use_flat_hcu_silu_mul,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHCUFlatSiluMulPolicy(unittest.TestCase):
    def test_validated_large_narrow_shape_uses_flat_kernel(self):
        self.assertTrue(
            _should_use_flat_hcu_silu_mul(
                65536,
                128,
                enabled=True,
                is_hcu_backend=True,
                tensors_compatible=True,
            )
        )

    def test_other_paths_use_vendor_kernel(self):
        cases = (
            (19999, 128, True, True, True),
            (65536, 512, True, True, True),
            (65536, 192, True, True, True),
            (65536, 128, False, True, True),
            (65536, 128, True, False, True),
            (65536, 128, True, True, False),
        )
        for m, d, enabled, is_hcu_backend, compatible in cases:
            with self.subTest(m=m, d=d):
                self.assertFalse(
                    _should_use_flat_hcu_silu_mul(
                        m,
                        d,
                        enabled=enabled,
                        is_hcu_backend=is_hcu_backend,
                        tensors_compatible=compatible,
                    )
                )


if __name__ == "__main__":
    unittest.main()
