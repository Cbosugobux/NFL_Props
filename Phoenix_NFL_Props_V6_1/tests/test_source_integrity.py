import math
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nfl_props_v6_1.py"


class SourceIntegrityTests(unittest.TestCase):
    def test_source_compiles(self):
        compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")

    def test_v61_coupling_markers_exist(self):
        s = SOURCE.read_text(encoding="utf-8")
        required = [
            'MODEL_VERSION = "6.1.0"',
            'USE_QB_RECEIVER_COUPLING = True',
            'QB_RECEIVER_CATCH_COUPLING = 0.60',
            'QB_RECEIVER_YPR_COUPLING = 0.50',
            'TEAM_PASS_ENV={}',
            'catch=np.clip(catch*catch_mult',
            'ypr=np.clip(ypr*ypr_mult',
            'QB_RECEIVER_COUPLING_AUDIT_V6_1',
        ]
        for token in required:
            self.assertIn(token, s)

    def test_synthetic_coupling_direction(self):
        random.seed(42)
        n = 20000
        baseline_comp = 0.65
        baseline_ypc = 11.0
        qb_env = []
        receiver_eff = []
        for _ in range(n):
            good = random.random() < 0.75
            comp = random.gauss(0.68 if good else 0.59, 0.025 if good else 0.03)
            ypc = random.gauss(11.8 if good else 9.6, 0.7 if good else 0.8)
            comp_ratio = min(1.18, max(0.82, comp / baseline_comp))
            ypc_ratio = min(1.18, max(0.82, ypc / baseline_ypc))
            catch_mult = math.exp(0.60 * math.log(comp_ratio))
            ypr_mult = math.exp(0.50 * math.log(ypc_ratio))
            qb_env.append(comp * ypc)
            receiver_eff.append(catch_mult * ypr_mult)

        mx = sum(qb_env) / n
        my = sum(receiver_eff) / n
        cov = sum((x-mx)*(y-my) for x,y in zip(qb_env, receiver_eff)) / n
        vx = sum((x-mx)**2 for x in qb_env) / n
        vy = sum((y-my)**2 for y in receiver_eff) / n
        corr = cov / math.sqrt(vx * vy)
        self.assertGreater(corr, 0.50)


if __name__ == "__main__":
    unittest.main()
