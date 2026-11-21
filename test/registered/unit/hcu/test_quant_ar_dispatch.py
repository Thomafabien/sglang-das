import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hcu_optim.comm import quant_ar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestQuantARDispatch(unittest.TestCase):
    def test_cpu_fallback_does_not_query_gpu_capture(self):
        group = SimpleNamespace(world_size=4)
        tensor = torch.zeros((8, 2048), dtype=torch.bfloat16)
        original = Mock(return_value=tensor)
        with (
            patch.object(quant_ar, "_PP", None),
            patch.object(quant_ar, "_refresh_cfg"),
            patch.object(torch.cuda, "is_current_stream_capturing") as capturing,
        ):
            self.assertIs(quant_ar.dispatch_all_reduce(group, tensor, original), tensor)
        original.assert_called_once_with(group, tensor)
        capturing.assert_not_called()
