import os
import sys
import tempfile
import unittest

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from harness import DATA_DIR, OOS_DIR, artifact_path  # noqa: E402
from diagnose import buy_and_hold  # noqa: E402
from factor_eval import period_mask  # noqa: E402
from factors import build_stock  # noqa: E402
from portfolio import run  # noqa: E402
from prefix_replay import select_prefixes  # noqa: E402


class ResearchArtifactTests(unittest.TestCase):
    def test_artifacts_include_pool_identity(self):
        dev = artifact_path("candidate", "metrics.json", DATA_DIR)
        oos = artifact_path("candidate", "metrics.json", OOS_DIR)

        self.assertNotEqual(dev, oos)
        self.assertIn("__dev__", dev)
        self.assertIn("__oos_revealed__", oos)

    def test_factor_period_mask_purges_forward_labels_across_boundary(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-01", "2023-12-20", "2024-01-02"]),
                "fwd20_date": pd.to_datetime(["2023-12-29", "2024-01-22", "2024-02-01"]),
            }
        )

        mask = period_mask(panel, "fwd20", "2020-01-01", "2023-12-31")

        self.assertEqual(mask.tolist(), [True, False, False])

    def test_factor_library_names_downside_risk_accurately(self):
        rows = 320
        close = pd.Series([100.0 + index * 0.1 for index in range(rows)])
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": [1_000_000.0] * rows,
            }
        )

        factors = build_stock(frame)

        self.assertIn("downside_vol", factors)
        self.assertNotIn("beta_dn", factors)
        self.assertNotIn("amihud", factors)

    def test_formal_prefix_selection_is_exhaustive_from_first_row(self):
        frame = pd.DataFrame(
            {"entry_signal": [0, 1, 0, 0], "exit_signal": [0, 0, 0, 1]}
        )

        self.assertEqual(select_prefixes(frame), [0, 1, 2, 3])
        sampled = select_prefixes(frame, points=3)
        self.assertEqual(len(sampled), 3)
        self.assertIn(1, sampled)
        self.assertIn(3, sampled)


class PortfolioExecutionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2026-01-01", periods=3, freq="D")
        self.markets = {"000001": "CN-A", "000002": "CN-A"}

    def test_missed_entry_does_not_become_a_late_purchase(self):
        close = pd.DataFrame(
            {"000001": [10.0, 20.0, 40.0], "000002": [10.0, 10.0, 10.0]},
            index=self.dates,
        )
        entry = pd.DataFrame(
            {"000001": [1, 0, 0], "000002": [1, 0, 0]}, index=self.dates
        )
        exit_ = pd.DataFrame(
            {"000001": [0, 0, 0], "000002": [0, 1, 0]}, index=self.dates
        )
        ranking = pd.DataFrame(
            {"000001": [0.0, 0.0, 0.0], "000002": [1.0, 1.0, 1.0]},
            index=self.dates,
        )

        result = run(close, entry, exit_, self.markets, ranking, slots=1, initial=10_000.0)

        self.assertLess(result["final"], 10_000.0)
        self.assertGreater(result["final"], 9_900.0)

    def test_none_ranking_uses_stable_code_order(self):
        close = pd.DataFrame(
            {"000001": [10.0, 15.0, 20.0], "000002": [10.0, 7.5, 5.0]},
            index=self.dates,
        )
        entry = pd.DataFrame(
            {"000001": [1, 0, 0], "000002": [1, 0, 0]}, index=self.dates
        )
        exit_ = pd.DataFrame(0, index=self.dates, columns=close.columns)

        first = run(
            close, entry, exit_, self.markets, None, slots=1, seed=1, initial=10_000.0
        )
        second = run(
            close, entry, exit_, self.markets, None, slots=1, seed=999, initial=10_000.0
        )

        self.assertAlmostEqual(first["final"], second["final"])
        self.assertGreater(first["final"], 19_000.0)

    def test_ranked_portfolio_skips_candidates_without_a_score(self):
        close = pd.DataFrame(
            {"000001": [10.0, 15.0, 20.0], "000002": [10.0, 7.5, 5.0]},
            index=self.dates,
        )
        entry = pd.DataFrame(
            {"000001": [1, 0, 0], "000002": [1, 0, 0]}, index=self.dates
        )
        exit_ = pd.DataFrame(0, index=self.dates, columns=close.columns)
        ranking = pd.DataFrame(float("nan"), index=self.dates, columns=close.columns)

        result = run(close, entry, exit_, self.markets, ranking, slots=1, initial=10_000.0)

        self.assertAlmostEqual(result["final"], 10_000.0)


class BuyAndHoldTests(unittest.TestCase):
    def test_buy_and_hold_uses_round_trip_fees(self):
        with tempfile.TemporaryDirectory() as data_dir:
            pd.DataFrame(
                {
                    "date": ["2026-01-01", "2026-01-02"],
                    "close": [10.0, 20.0],
                }
            ).to_csv(os.path.join(data_dir, "000001_hfq.csv"), index=False)

            result = buy_and_hold(data_dir).iloc[0]

        self.assertLess(result["bh"], 100.0)
        self.assertAlmostEqual(result["bh"], 99.855, places=3)


if __name__ == "__main__":
    unittest.main()
