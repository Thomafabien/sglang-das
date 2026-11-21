import unittest

from sglang.srt.hcu_optim.ops.dense_pack import select_dense_pack_range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHCUDensePackPolicy(unittest.TestCase):
    def test_validated_shapes_and_decode_ranges(self):
        self.assertEqual(select_dense_pack_range((3072, 2048), 8), (8, 64))
        self.assertEqual(select_dense_pack_range((2560, 2048), 64), (8, 64))
        self.assertEqual(select_dense_pack_range((2048, 1024), 32), (8, 64))
        self.assertEqual(select_dense_pack_range((2048, 4096), 32), (8, 32))

    def test_unvalidated_shapes_or_m_values_fall_back(self):
        for shape, m in (
            ((3072, 2048), 4),
            ((2048, 4096), 64),
            ((256, 2048), 16),
            ((62080, 2048), 16),
        ):
            with self.subTest(shape=shape, m=m):
                self.assertIsNone(select_dense_pack_range(shape, m))


if __name__ == "__main__":
    unittest.main()
