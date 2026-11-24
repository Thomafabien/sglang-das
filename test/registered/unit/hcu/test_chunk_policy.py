import __future__
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers import chunk_policy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestChunkPolicy(CustomTestCase):
    def test_disabled_factory(self):
        for value in (None, "0"):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop("SGLANG_ADAPTIVE_CHUNK", None)
                if value is not None:
                    os.environ["SGLANG_ADAPTIVE_CHUNK"] = value
                self.assertIsNone(chunk_policy.AdaptiveChunkPolicy.from_env(pair_prefill=False))

    def test_thresholds_and_budget(self):
        with patch.dict(os.environ, {
            "SGLANG_ADAPTIVE_CHUNK": "1",
            "SGLANG_ADAPTIVE_CHUNK_EVEN": "0",
            "SGLANG_ADAPTIVE_CHUNK_TIERS": "32768:16384,65536:32768",
        }), patch.object(chunk_policy, "_OFF", Mock(exists=Mock(return_value=False))):
            policy = chunk_policy.AdaptiveChunkPolicy.from_env(pair_prefill=False)
            for length, limit, expected in (
                (32767, None, 8192), (32768, None, 16384),
                (65535, None, 16384), (65536, None, 32768),
                (65536, 20000, 20000),
            ):
                with self.subTest(length=length, limit=limit):
                    self.assertEqual(policy.choose(
                        base=8192, chunked_req=None, page_size=64, max_tokens=limit,
                        waiting_queue=[SimpleNamespace(origin_input_ids=range(length))],
                    ), expected)

    def test_scheduler_restores_chunk_size(self):
        # Compile the actual orchestration method without importing GPU backends.
        path = Path(chunk_policy.__file__).with_name("scheduler.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                         and node.name == "Scheduler")
        method = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef)
                      and node.name == "get_new_batch_prefill")
        namespace = {}
        code = compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        )
        exec(code, namespace)
        call = namespace[method.name]
        for policy_size, fail in ((None, False), (16384, False), (16384, True)):
            with self.subTest(policy_size=policy_size, fail=fail):
                state = SimpleNamespace(
                    adaptive_chunk_policy=Mock(choose=Mock(return_value=policy_size))
                    if policy_size else None,
                    chunked_prefill_size=8192, chunked_req=None, waiting_queue=[],
                    page_size=64, max_prefill_tokens=40960,
                )
                batch = object()

                def plan(actual_batch):
                    self.assertIs(actual_batch, batch)
                    self.assertEqual(state.chunked_prefill_size, policy_size or 8192)
                    if fail:
                        raise RuntimeError("planning failed")
                    return batch

                state._plan_or_yield_prefill = Mock(side_effect=plan)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "planning failed"):
                        call(state, batch)
                else:
                    self.assertIs(call(state, batch), batch)
                state._plan_or_yield_prefill.assert_called_once_with(batch)
                self.assertEqual(state.chunked_prefill_size, 8192)


if __name__ == "__main__":
    unittest.main()
