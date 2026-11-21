import unittest
from unittest.mock import patch

import torch

from sglang.srt.hcu_optim.ops import kv_write_fused
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFusedKVWrite(unittest.TestCase):
    def test_cpu_input_uses_original_writer(self):
        k = torch.empty((2, 1, 128), dtype=torch.bfloat16)
        loc = torch.tensor([0, 64])
        kb = torch.empty((2, 1, 64, 128), dtype=torch.uint8)
        vb = torch.empty((2, 1, 128, 64), dtype=torch.uint8)
        with patch.object(kv_write_fused, "_fused") as launch:
            self.assertFalse(kv_write_fused.try_fused_write(k, k, loc, kb, vb, 64))
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
