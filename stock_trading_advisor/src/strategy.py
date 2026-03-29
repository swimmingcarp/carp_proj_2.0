"""
Strategy utilities shared by RSITrendStrategy.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyBase:
    """RSITrendStrategy 复用的基础配置与通用回测基座。"""

    def __init__(self, config: Optional[Dict] = None, validate_indicators: bool = True,
                 sell_strategy: str = 'auto', order: str = 'auto',
                 market: str = 'CN-A', use_simple_divergence: bool = False,
                 adaptive_oscillation: bool = False, stock_code: str = '',
                 precomputed_indicators: bool = False,
                 oscillation_driven: bool = False):
        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.validate_indicators = validate_indicators
        self.sell_strategy = sell_strategy
        self.order_mode = order
        self.optimal_strategy = None
        self.optimal_order = 'high_frequency'
        self.adaptive_oscillation = adaptive_oscillation
        self.stock_code = stock_code
        self.selected_oscillation_version = None
        self.market = market
        self.use_simple_divergence = use_simple_divergence
        self.precomputed_indicators = precomputed_indicators
        self.oscillation_driven = oscillation_driven
        self._oscillation_confirmed_periods: List[Tuple] = []

        if self.validate_indicators:
            try:
                from .indicator_validator import IndicatorValidator
            except ImportError:
                from indicator_validator import IndicatorValidator
            self.indicator_validator = IndicatorValidator()
        else:
            self.indicator_validator = None

    def _default_config(self) -> Dict:
        """保留通用回测/手续费默认配置。"""
        return {
            'commission_enabled': True,
            'commission_cn_a': 0.00015,
            'commission_hk': 0.0025,
            'commission_us': 0.0002,
            'commission_min_cn': 5.0,
            'stamp_duty_cn': 0.0005,
            'stamp_duty_hk': 0.0013,
        }

    def _calculate_commission(self, transaction_amount: float, is_buy: bool = True) -> float:
        """根据市场与双边规则计算交易手续费。"""
        if not self.config.get('commission_enabled', True):
            return 0.0

        if self.market == 'HK':
            commission_rate = self.config.get('commission_hk', 0.0025)
            stamp_duty_rate = self.config.get('stamp_duty_hk', 0.0013)
            return transaction_amount * (commission_rate + stamp_duty_rate)

        if self.market == 'US':
            commission_rate = self.config.get('commission_us', 0.0002)
            return transaction_amount * commission_rate

        commission_rate = self.config.get('commission_cn_a', 0.0003)
        commission_min = self.config.get('commission_min_cn', 5.0)
        commission = max(transaction_amount * commission_rate, commission_min)
        if not is_buy:
            stamp_duty_rate = self.config.get('stamp_duty_cn', 0.001)
            commission += transaction_amount * stamp_duty_rate
        return commission

    def get_oscillation_confirmed_periods(self) -> List[Tuple]:
        """返回可视化层使用的震荡确认区间缓存。"""
        return self._oscillation_confirmed_periods

    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Optional[Dict]:
        """通用回测实现，按收盘买卖并计入双边手续费。"""
        if df is None or 'buy_signal' not in df.columns:
            return None

        n = len(df)
        if n == 0:
            return None

        close_arr = df['close'].to_numpy()
        signal_arr = df['buy_signal'].to_numpy()
        if 'date' in df.columns:
            date_arr = df['date'].astype(str).to_numpy()
        else:
            date_arr = df.index.astype(str).to_numpy()

        capital = initial_capital
        capital_list = np.empty(n, dtype=float)
        trades = []
        holding = False
        buy_price = 0.0
        buy_date = None
        shares = 0.0
        buy_commission = 0.0
        total_commission = 0.0

        gross_capital = 10000.0
        gross_holding = False
        gross_buy_price = 0.0
        gross_shares = 0.0

        for i in range(n):
            curr_signal = signal_arr[i]
            curr_price = close_arr[i]
            curr_date = date_arr[i]

            if curr_signal == 1 and not holding:
                buy_price = curr_price
                buy_date = curr_date
                shares = capital / buy_price
                transaction_amount = shares * buy_price

                buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                capital -= buy_commission
                total_commission += buy_commission
                holding = True

            elif curr_signal == 0 and holding:
                sell_price = curr_price
                transaction_amount = shares * sell_price
                sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
                total_commission += sell_commission
                capital = transaction_amount - buy_commission - sell_commission

                initial_investment = shares * buy_price
                profit = (sell_price - buy_price) * shares - (buy_commission + sell_commission)
                profit_rate = profit / initial_investment if initial_investment > 0 else 0.0

                trades.append({
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': curr_date,
                    'sell_price': sell_price,
                    'profit_rate': profit_rate,
                    'capital': capital,
                    'commission': buy_commission + sell_commission,
                    'buy_commission': buy_commission,
                    'sell_commission': sell_commission,
                })

                holding = False
                shares = 0.0
                buy_commission = 0.0

            if curr_signal == 1 and not gross_holding:
                gross_buy_price = curr_price
                gross_shares = gross_capital / gross_buy_price
                gross_holding = True
            elif curr_signal == 0 and gross_holding:
                sell_price_gross = curr_price
                transaction_amount_gross = gross_shares * sell_price_gross
                gross_capital = transaction_amount_gross
                gross_holding = False
                gross_shares = 0.0

            capital_list[i] = capital

        if holding:
            sell_price = close_arr[-1]
            transaction_amount = shares * sell_price
            sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
            total_commission += sell_commission
            capital = transaction_amount - buy_commission - sell_commission

            initial_investment = shares * buy_price
            profit = (sell_price - buy_price) * shares - (buy_commission + sell_commission)
            profit_rate = profit / initial_investment if initial_investment > 0 else 0.0

            trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': date_arr[-1],
                'sell_price': sell_price,
                'profit_rate': profit_rate,
                'capital': capital,
                'commission': buy_commission + sell_commission,
                'buy_commission': buy_commission,
                'sell_commission': sell_commission,
            })

            capital_list[-1] = capital

        if gross_holding:
            sell_price_gross = df.iloc[-1]['close']
            transaction_amount_gross = gross_shares * sell_price_gross
            gross_capital = transaction_amount_gross

        df['capital'] = capital_list

        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100
        gross_return = (gross_capital - initial_capital) / initial_capital * 100

        capital_series = pd.Series(capital_list)
        running_max = capital_series.expanding().max()
        drawdown = (capital_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0.0

        if trades:
            trade_returns_series = pd.Series([t['profit_rate'] for t in trades])
            sharpe = (
                trade_returns_series.mean() / trade_returns_series.std() * np.sqrt(252)
                if trade_returns_series.std() > 0 else 0.0
            )
        else:
            sharpe = 0.0

        winning_trades = sum(1 for t in trades if t['profit_rate'] > 0)
        win_rate = (winning_trades / len(trades) * 100) if trades else 0.0

        sample_interval = max(1, len(df) // 10)
        capital_curve = []
        for i in range(0, len(df), sample_interval):
            row = df.iloc[i]
            capital_curve.append({
                'date': row['date'],
                'capital': capital_list[i],
                'return_pct': (capital_list[i] / initial_capital - 1) * 100,
            })

        if len(df) > 0 and (len(df) - 1) % sample_interval != 0:
            last_row = df.iloc[-1]
            capital_curve.append({
                'date': last_row['date'],
                'capital': capital_list[-1],
                'return_pct': (capital_list[-1] / initial_capital - 1) * 100,
            })

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'gross_return': gross_return,
            'total_commission': total_commission,
            'commission_rate': (total_commission / initial_capital * 100),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trading_days': len(df),
            'capital_curve': capital_curve,
            'trades': trades,
        }
