import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAirTunableOpProfiles(unittest.TestCase):
    def test_all_tp4_profiles_are_packaged(self):
        profile_dir = (
            Path(__file__).parents[4]
            / "python"
            / "sglang"
            / "srt"
            / "hcu_optim"
            / "config"
            / "tunableop"
            / "air"
        )
        for rank in range(4):
            path = profile_dir / f"union{rank}.csv"
            with self.subTest(rank=rank):
                text = path.read_text(encoding="utf-8")
                self.assertIn("Validator,GCN_ARCH_NAME,gfx928", text)
                self.assertIn("GemmTunableOp_BFloat16_TN", text)


if __name__ == "__main__":
    unittest.main()
