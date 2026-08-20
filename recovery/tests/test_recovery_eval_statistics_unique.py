from __future__ import annotations

import unittest

from recovery.eval.statistics import bootstrap_mean_ci, paired_comparison


class RecoveryEvalStatisticsTests(unittest.TestCase):
    def test_recovery_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_mean_ci(
            [1.0, 2.0, 3.0, 8.0], samples=250, confidence=0.95, seed=19
        )
        second = bootstrap_mean_ci(
            [1.0, 2.0, 3.0, 8.0], samples=250, confidence=0.95, seed=19
        )
        self.assertEqual(first, second)
        self.assertLess(first["low"], first["high"])

    def test_recovery_paired_comparison_orients_lower_as_better(self) -> None:
        comparison = paired_comparison(
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            higher_is_better=False,
            tie_tolerance=1e-12,
            bootstrap_samples=100,
            confidence=0.95,
            seed=3,
        )
        self.assertEqual(comparison["wins"], 3)
        self.assertEqual(comparison["losses"], 0)
        self.assertGreater(comparison["mean_improvement"], 0.0)
        self.assertLess(comparison["mean_candidate_minus_baseline"], 0.0)


if __name__ == "__main__":
    unittest.main()
