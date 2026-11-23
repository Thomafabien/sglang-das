import unittest
from concurrent.futures import Future
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.managers.tokenizer_pool import TokenizerPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTokenizerPool(unittest.IsolatedAsyncioTestCase):
    async def test_first_mismatch_disables_pool_and_preserves_reference(self):
        pool = TokenizerPool(
            path="unused", trust_remote_code=False, min_chars=1, segments=2, log_count=0
        )
        executor = Mock()
        result = Future()
        result.set_result([7])
        executor.submit.return_value = result
        pool.pool = executor
        reference = ([1, 2], [0, 0])
        fallback = AsyncMock(return_value=reference)
        with patch("sglang.srt.managers.tokenizer_pool._split", return_value=["a", " b"]):
            self.assertEqual(await pool.tokenize("a b", False, fallback), reference)
            self.assertTrue(pool.disabled)
            self.assertEqual(await pool.tokenize("c d", False, fallback), reference)
        self.assertEqual(executor.submit.call_count, 2)
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
