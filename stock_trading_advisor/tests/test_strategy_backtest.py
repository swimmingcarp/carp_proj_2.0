import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategy import StrategyBase


class TestStrategyBacktest(unittest.TestCase):
    def test_holding_days_are_marked_to_close_without_changing_trade_accounting(self):
        df = pd.DataFrame(
            {
                'date': pd.date_range('2026-01-01', periods=4),
                'close': [100.0, 120.0, 60.0, 90.0],
                'buy_signal': [1, 1, 1, 0],
            }
        )

        result = StrategyBase(
            config={'commission_min_cn': 5.0},
            market='CN-A',
        ).backtest(df, initial_capital=10000.0)

        self.assertIsNotNone(result)
        self.assertEqual(result['total_trades'], 1)
        self.assertAlmostEqual(result['final_capital'], 8985.5)
        self.assertAlmostEqual(result['total_return'], -10.145)
        self.assertAlmostEqual(result['trades'][0]['profit_rate'] * 100, -10.145)
        self.assertEqual(df['capital'].tolist(), [9995.0, 11995.0, 5995.0, 8985.5])
        self.assertAlmostEqual(result['max_drawdown'], (5995.0 / 11995.0 - 1.0) * 100)


if __name__ == '__main__':
    unittest.main()
