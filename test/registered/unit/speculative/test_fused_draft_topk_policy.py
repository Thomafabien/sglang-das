import unittest
from unittest.mock import Mock

import torch

from sglang.srt.hcu_optim.ops.fused_draft_topk import can_use_fused_draft_topk
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFusedDraftTopkPolicy(unittest.TestCase):
    def test_target_shape_is_selected(self):
        logits = Mock(dim=lambda: 2, is_cuda=True, dtype=torch.float32)
        logits.shape = (8, 248320)
        self.assertTrue(can_use_fused_draft_topk(logits, False, 1))

    def test_rejection_sampling_and_topk_gt_one_fall_back(self):
        logits = Mock(dim=lambda: 2, is_cuda=True, dtype=torch.float32)
        logits.shape = (8, 248320)
        self.assertFalse(can_use_fused_draft_topk(logits, True, 1))
        self.assertFalse(can_use_fused_draft_topk(logits, False, 4))


if __name__ == "__main__":
    unittest.main()
