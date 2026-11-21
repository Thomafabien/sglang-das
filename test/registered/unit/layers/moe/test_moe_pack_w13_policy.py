import unittest

import torch

from sglang.srt.hcu_optim.ops.moe_pack_w13 import BK, BN, _pack
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoePackW13Layout(unittest.TestCase):
    def test_pack_preserves_shape_and_elements(self):
        weight = torch.arange(BN * BK, dtype=torch.float32).view(1, BN, BK)
        packed = _pack(weight)
        self.assertEqual(packed.shape, weight.shape)
        self.assertEqual(
            torch.sort(packed.flatten()).values.tolist(),
            torch.sort(weight.flatten()).values.tolist(),
        )

    def test_first_tile_is_contiguous_in_packed_storage(self):
        weight = torch.arange(BN * BK * 2, dtype=torch.float32).view(
            1, BN, BK * 2
        )
        packed = _pack(weight).flatten()
        expected = weight[0, :, :BK].contiguous().flatten()
        self.assertTrue(torch.equal(packed[: BN * BK], expected))


if __name__ == "__main__":
    unittest.main()
