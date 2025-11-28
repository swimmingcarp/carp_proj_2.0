"""
RSI Trend Strategy
Based on https://github.com/zchancode/trading/blob/main/RSI趋势策略.tv

This module recreates the TradingView RSI Cross strategy inside the Python
backtest engine. The logic follows the original Pine Script:
    - dual RSI cross (fast RSI over slow RSI) triggers entries
    - ATR-based trailing stop controls the underlying trend direction
    - Heikin Ashi candles are used as a confirmation filter
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .indicators import (
        atr_indicator,
        ma_indicator,
        ema_indicator,
        rsi_indicator,
    )
    from .strategy import MixedStrategy
except ImportError:  # pragma: no cover - fallback for standalone usage
    from indicators import atr_indicator, rsi_indicator, ma_indicator, ema_indicator
    from strategy import MixedStrategy

logger = logging.getLogger(__name__)


class RSITrendStrategy(MixedStrategy):
    """
    RSI趋势策略

    特征:
        1. 使用 fast RSI (默认25) 与 slow RSI (默认100) 的金叉作为触发信号
        2. ATR × Multiplier 组成的动态止损决定多空方向（等价于SuperTrend思想）
        3. Heikin Ashi 阳线作为额外的趋势确认
    """

    def __init__(self, config: Optional[Dict] = None, market: str = 'CN-A',
                 stock_code: str = ''):
        defaults = {
            'trend_rsi_fast_period': 25,
            'trend_rsi_slow_period': 100,
            'trend_atr_period': 20,
            'trend_atr_multiplier': 3.0,
            'trend_use_close_for_extrema': True,
            'trend_relaxed_entry': True,
            'trend_relaxed_min_gap': 1.0,
            'trend_use_zigzag_filter': False,
            'trend_zigzag_tolerance_pct': 1.5,
            'trend_stop_loss_pct': 7.0,
            'trend_exit_use_ma_filter': True,
            'trend_exit_fast_ema_period': 16,
            'trend_exit_slow_ma_period': 45,
            'trend_exit_confirm_ma_period': 16,
            'trend_lr_filter_enabled': True,
            'trend_lr_lookback': 30,
            'trend_lr_max_slope_pct': 1.5,
        }
        if config:
            defaults.update(config)

        super().__init__(
            config=defaults,
            validate_indicators=False,
            sell_strategy='original',
            order='high_frequency',
            market=market,
            use_simple_divergence=False,
            adaptive_oscillation=False,
            stock_code=stock_code,
            precomputed_indicators=False,
            use_strategy_cache=False,
            oscillation_driven=False
        )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def analyze(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """执行RSI趋势策略分析"""
        if df is None or len(df) == 0:
            logger.warning("RSITrendStrategy: 数据为空，无法分析")
            return None, None

        data = self._prepare_dataframe(df)
        fast_period = int(self.config.get('trend_rsi_fast_period', 25))
        slow_period = int(self.config.get('trend_rsi_slow_period', 100))
        atr_period = int(self.config.get('trend_atr_period', 20))
        atr_multiplier = float(self.config.get('trend_atr_multiplier', 3.0))
        use_close = bool(self.config.get('trend_use_close_for_extrema', True))
        exit_ma_filter_enabled = bool(self.config.get('trend_exit_use_ma_filter', True))
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))
        exit_fast_ema_period = max(1, int(self.config.get('trend_exit_fast_ema_period', 16)))
        exit_slow_ma_period = max(1, int(self.config.get('trend_exit_slow_ma_period', 45)))
        exit_confirm_ma_period = max(
            1,
            int(self.config.get('trend_exit_confirm_ma_period', exit_fast_ema_period))
        )
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))
        lr_lookback = max(5, int(self.config.get('trend_lr_lookback', 30)))
        lr_max_slope_pct = max(0.0, float(self.config.get('trend_lr_max_slope_pct', 2.0)))

        data['fast_rsi'] = rsi_indicator(data['close'], period=fast_period)
        data['slow_rsi'] = rsi_indicator(data['close'], period=slow_period)

        atr_values = atr_indicator(data, period=atr_period) * atr_multiplier
        data['atr_trailing'] = atr_values

        long_stop, short_stop, direction = self._compute_trend_levels(
            data,
            atr_values,
            atr_period,
            use_close
        )
        data['long_stop'] = long_stop
        data['short_stop'] = short_stop
        data['trend_direction'] = direction

        ha_open, ha_close, ha_high, ha_low = self._calculate_heikin_ashi(data)
        data['ha_open'] = ha_open
        data['ha_close'] = ha_close
        data['ha_high'] = ha_high
        data['ha_low'] = ha_low
        data['is_heikin_bullish'] = ha_close > ha_open

        data['golden_cross'] = self._crossover(data['fast_rsi'], data['slow_rsi'])
        data['death_cross'] = self._crossunder(data['fast_rsi'], data['slow_rsi'])
        data['rsi_diff'] = data['fast_rsi'] - data['slow_rsi']

        relaxed_enabled = bool(self.config.get('trend_relaxed_entry', True))
        relaxed_gap = max(0.0, float(self.config.get('trend_relaxed_min_gap', 1.0)))
        if relaxed_enabled:
            rsi_relaxed_condition = (data['rsi_diff'] >= relaxed_gap)
        else:
            rsi_relaxed_condition = pd.Series(False, index=data.index)
        data['rsi_relaxed_condition'] = rsi_relaxed_condition

        if lr_filter_enabled:
            def _lr_slope(arr: np.ndarray) -> float:
                if len(arr) == 0 or np.isnan(arr).any():
                    return np.nan
                x = np.arange(len(arr), dtype=float)
                return np.polyfit(x, arr, 1)[0]

            def _lr_zscore(arr: np.ndarray) -> float:
                if len(arr) == 0 or np.isnan(arr).any():
                    return np.nan
                x = np.arange(len(arr), dtype=float)
                slope, intercept = np.polyfit(x, arr, 1)
                fitted = slope * x + intercept
                residuals = arr - fitted
                std = np.std(residuals, ddof=1)
                if std == 0 or np.isnan(std):
                    return 0.0
                last_idx = len(arr) - 1
                last_fit = slope * last_idx + intercept
                return (arr[-1] - last_fit) / std

            lr_slope = data['close'].rolling(
                lr_lookback,
                min_periods=lr_lookback
            ).apply(_lr_slope, raw=True)
            steep_down = (lr_slope < 0) & (lr_slope.abs() > lr_max_slope_pct)
            lr_filter_condition = (~steep_down)

            data['lr_slope'] = lr_slope
            data['lr_steep_down'] = steep_down.fillna(False)
            data['lr_filter_condition'] = lr_filter_condition.fillna(False)
        else:
            nan_series = pd.Series(np.nan, index=data.index)
            data['lr_slope'] = nan_series.copy()
            data['lr_steep_down'] = pd.Series(False, index=data.index)
            lr_filter_condition = pd.Series(True, index=data.index)
            data['lr_filter_condition'] = lr_filter_condition

        strong_ma_uptrend = pd.Series(False, index=data.index)
        close_below_confirm = pd.Series(False, index=data.index)
        if exit_ma_filter_enabled:
            exit_ema_fast = ema_indicator(data['close'], period=exit_fast_ema_period)
            exit_ma_slow = ma_indicator(data['close'], period=exit_slow_ma_period)
            exit_ma_confirm = ma_indicator(data['close'], period=exit_confirm_ma_period)
            data['exit_ema_fast'] = exit_ema_fast
            data['exit_ma_slow'] = exit_ma_slow
            data['exit_ma_confirm'] = exit_ma_confirm
            strong_ma_uptrend = exit_ema_fast > exit_ma_slow
            close_below_confirm = (data['close'] < exit_ma_confirm).fillna(False)
            close_below_arr = close_below_confirm.to_numpy(dtype=bool, copy=True)
            streak_arr = np.zeros(len(close_below_arr), dtype=int)
            if len(streak_arr) > 0 and close_below_arr[0]:
                streak_arr[0] = 1
            for i in range(1, len(streak_arr)):
                if close_below_arr[i]:
                    streak_arr[i] = streak_arr[i - 1] + 1
                else:
                    streak_arr[i] = 0
            streak_series = pd.Series(streak_arr, index=data.index)
            ma_filter_break = strong_ma_uptrend & (streak_series >= 3)
            data['exit_close_below_ma_confirm'] = close_below_confirm
            data['exit_ma_confirm_below_streak'] = streak_series
            data['exit_ma_filter_break'] = ma_filter_break
            data['exit_strong_ma_trend'] = strong_ma_uptrend
            data['exit_ma_filter_active'] = strong_ma_uptrend & (streak_series < 3)
        else:
            nan_series = pd.Series(np.nan, index=data.index)
            data['exit_ema_fast'] = nan_series.copy()
            data['exit_ma_slow'] = nan_series.copy()
            data['exit_ma_confirm'] = nan_series.copy()
            data['exit_close_below_ma_confirm'] = pd.Series(False, index=data.index)
            data['exit_ma_confirm_below_streak'] = pd.Series(0, index=data.index)
            data['exit_ma_filter_break'] = pd.Series(False, index=data.index)
            data['exit_strong_ma_trend'] = pd.Series(False, index=data.index)
            data['exit_ma_filter_active'] = pd.Series(False, index=data.index)

        zigzag_enabled = bool(self.config.get('trend_use_zigzag_filter', True))
        zigzag_tolerance = float(self.config.get('trend_zigzag_tolerance_pct', 0.0))
        if zigzag_enabled:
            last_low, prev_low = self._compute_zigzag_lows(data)
            data['zigzag_last_low'] = last_low
            data['zigzag_prev_low'] = prev_low
            zigzag_condition = ~(
                last_low.notna() &
                prev_low.notna() &
                (last_low < prev_low * (1 + zigzag_tolerance / 100.0))
            )
        else:
            zigzag_condition = pd.Series(True, index=data.index)
            data['zigzag_last_low'] = np.nan
            data['zigzag_prev_low'] = np.nan
        data['zigzag_condition'] = zigzag_condition

        stop_loss_pct = max(0.0, float(self.config.get('trend_stop_loss_pct', 7.0)))

        entry_condition = (
            (direction == 1) &
            data['is_heikin_bullish'] &
            zigzag_condition &
            (data['golden_cross'] | rsi_relaxed_condition) &
            lr_filter_condition
        )
        exit_condition = (direction != 1)
        if exit_ma_filter_enabled:
            exit_condition = exit_condition & (
                ~strong_ma_uptrend | data['exit_ma_filter_break']
            )

        position, entry_flags, exit_flags, stop_loss_flags = self._build_position_series(
            entry_condition,
            exit_condition,
            data['close'],
            stop_loss_pct
        )

        data['buy_signal'] = position
        data['entry_signal'] = entry_flags
        data['exit_signal'] = exit_flags
        data['stop_loss_exit'] = stop_loss_flags
        data['stop_loss_pct'] = stop_loss_pct if stop_loss_pct > 0 else np.nan
        data['entry_reason'] = self._build_entry_reasons(data, entry_flags)
        data['exit_reason'] = self._build_exit_reasons(data, exit_flags)

        return data, None

    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """输出最新的策略信号"""
        if df is None or len(df) == 0:
            return {'signal': 'NO_DATA', 'reason': '数据不足'}

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        exit_ma_filter_enabled = bool(self.config.get('trend_exit_use_ma_filter', True))
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))

        signal = 'HOLD'
        reasons: List[str] = []
        strength = 1

        stop_loss_pct = float(self.config.get('trend_stop_loss_pct', 7.0))

        if latest.get('entry_signal', 0) == 1:
            signal = 'BUY'
            if latest.get('golden_cross'):
                reasons.append("RSI快线金叉慢线")
            elif latest.get('rsi_relaxed_condition'):
                reasons.append("RSI保持在慢线之上（放宽入场）")
            else:
                reasons.append("RSI多头信号")
            if latest.get('is_heikin_bullish'):
                reasons.append("Heikin Ashi 阳线确认")
            if latest.get('trend_direction') == 1:
                reasons.append("ATR趋势仍为多头")
            strength = 2 + int(latest.get('is_heikin_bullish', False)) + int(latest.get('trend_direction', 0) == 1)

        elif latest.get('exit_signal', 0) == 1 or (
            previous.get('buy_signal', 0) == 1 and latest.get('buy_signal', 0) == 0
        ):
            signal = 'SELL'
            if latest.get('stop_loss_exit'):
                reasons.append(f"触发{stop_loss_pct:.1f}%止损")
            else:
                reasons.append("ATR趋势转空，触发止盈/止损")
                if latest.get('death_cross'):
                    reasons.append("RSI死叉确认转弱")
                if exit_ma_filter_enabled and latest.get('exit_ma_filter_break'):
                    reasons.append("EMA16>MA45但已连续3日跌破MA16")
            strength = 3 + int(latest.get('death_cross', False)) + int(latest.get('stop_loss_exit', False))

        elif latest.get('buy_signal', 0) == 1:
            signal = 'HOLD_BUY'
            reasons.append("多头持仓中")
            if latest.get('trend_direction') == 1:
                reasons.append("ATR多头趋势未破坏")
            if latest.get('is_heikin_bullish'):
                reasons.append("Heikin Ashi 保持阳线")
            if latest.get('fast_rsi', 0) > latest.get('slow_rsi', 0):
                reasons.append("RSI保持强势")
            if exit_ma_filter_enabled and latest.get('exit_ma_filter_active'):
                reasons.append("EMA16>MA45，均线保护持仓")
            strength = min(5, 1 + int(latest.get('trend_direction', 0) == 1) + int(latest.get('is_heikin_bullish', False)))
        else:
            reasons.append("等待下一次RSI金叉")
            if lr_filter_enabled and not bool(latest.get('lr_filter_condition', True)):
                reasons.append("线性回归趋势/偏离未满足，暂缓抄底")
            strength = 1

        return {
            'signal': signal,
            'strength': min(strength, 5),
            'reason': '；'.join(reasons) if reasons else '无明确信号',
            'price': latest.get('close', 0),
            'date': latest.get('date'),
            'fast_rsi': latest.get('fast_rsi'),
            'slow_rsi': latest.get('slow_rsi'),
            'trend_direction': latest.get('trend_direction'),
            'ha_bullish': latest.get('is_heikin_bullish'),
            # 兼容 signal analyzer 的字段
            'k': latest.get('fast_rsi', 0),
            'd': latest.get('slow_rsi', 0),
            'macd': 0.0,
            'diff': 0.0,
            'ema_16': latest.get('exit_ema_fast', np.nan),
            'ma_16': latest.get('exit_ma_confirm', np.nan),
            'ma_45': latest.get('exit_ma_slow', np.nan),
            'lr_slope': latest.get('lr_slope', np.nan),
            'lr_filter_ok': latest.get('lr_filter_condition', True),
        }

    def get_trading_signals(self, df: pd.DataFrame,
                            initial_capital: float = 10000.0) -> Dict:
        """返回可视化买卖点"""
        base_result = {
            'buy_points': [],
            'sell_points': [],
            'total_trades': 0,
            'initial_capital': initial_capital,
        }

        if df is None or 'buy_signal' not in df.columns:
            return base_result

        backtest_result = super().backtest(df, initial_capital)
        if not backtest_result or 'trades' not in backtest_result:
            return base_result

        trades = backtest_result['trades']
        if not trades:
            return base_result

        buy_points: List[Dict] = []
        sell_points: List[Dict] = []
        last_is_open = bool(df.iloc[-1]['buy_signal'] == 1)

        for idx, trade in enumerate(trades):
            buy_row = self._find_row_by_date(df, trade['buy_date'])
            sell_row = self._find_row_by_date(df, trade['sell_date'])
            is_last_trade_open = (idx == len(trades) - 1) and last_is_open

            if buy_row is not None:
                buy_points.append({
                    'date': trade['buy_date'],
                    'price': trade['buy_price'],
                    'reason': buy_row.get('entry_reason', 'RSI金叉信号'),
                    'fast_rsi': buy_row.get('fast_rsi'),
                    'slow_rsi': buy_row.get('slow_rsi'),
                    'trend_direction': buy_row.get('trend_direction'),
                    'ha_bullish': buy_row.get('is_heikin_bullish'),
                    'commission': trade.get('buy_commission', 0),
                })

            if sell_row is not None:
                reason = sell_row.get('exit_reason', 'ATR趋势转空')
                if is_last_trade_open:
                    reason = "持仓中：ATR趋势仍为多头"

                sell_points.append({
                    'date': trade['sell_date'],
                    'price': trade['sell_price'],
                    'reason': reason,
                    'fast_rsi': sell_row.get('fast_rsi'),
                    'slow_rsi': sell_row.get('slow_rsi'),
                    'trend_direction': sell_row.get('trend_direction'),
                    'ha_bullish': sell_row.get('is_heikin_bullish'),
                    'is_open': is_last_trade_open,
                    'commission': trade.get('sell_commission', 0),
                })

        return {
            'buy_points': buy_points,
            'sell_points': sell_points,
            'total_trades': len(trades),
            'initial_capital': initial_capital,
            'trades': trades,
            'gross_return': backtest_result.get('gross_return', 0),
        }

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        else:
            data = data.sort_index().reset_index(drop=True)

        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(
                f"RSI趋势策略缺少必需字段: {', '.join(missing)}，"
                "请确认数据源包含 open/high/low/close。"
            )
        return data

    def _compute_trend_levels(self, data: pd.DataFrame, atr_values: pd.Series,
                              period: int, use_close: bool) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算ATR止损线与趋势方向"""
        src_high = data['close'] if use_close else data['high']
        src_low = data['close'] if use_close else data['low']

        highest = src_high.rolling(period).max()
        lowest = src_low.rolling(period).min()
        long_base = highest - atr_values
        short_base = lowest + atr_values

        close_arr = data['close'].to_numpy()
        long_base_arr = long_base.to_numpy()
        short_base_arr = short_base.to_numpy()

        n = len(data)
        long_stop = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)
        direction = np.ones(n, dtype=int)

        for i in range(n):
            base_long = long_base_arr[i]
            base_short = short_base_arr[i]
            prev_long = long_stop[i - 1] if i > 0 else base_long
            prev_short = short_stop[i - 1] if i > 0 else base_short

            if np.isnan(prev_long):
                prev_long = base_long
            if np.isnan(prev_short):
                prev_short = base_short

            candidate_long = base_long
            if (
                i > 0
                and not np.isnan(prev_long)
                and not np.isnan(close_arr[i - 1])
                and close_arr[i - 1] > prev_long
            ):
                candidate_long = prev_long if np.isnan(candidate_long) else max(candidate_long, prev_long)
            if np.isnan(candidate_long):
                candidate_long = prev_long
            long_stop[i] = candidate_long

            candidate_short = base_short
            if (
                i > 0
                and not np.isnan(prev_short)
                and not np.isnan(close_arr[i - 1])
                and close_arr[i - 1] < prev_short
            ):
                candidate_short = prev_short if np.isnan(candidate_short) else min(candidate_short, prev_short)
            if np.isnan(candidate_short):
                candidate_short = prev_short
            short_stop[i] = candidate_short

            prev_dir = direction[i - 1] if i > 0 else 1
            dir_value = prev_dir
            if not np.isnan(prev_short) and not np.isnan(close_arr[i]) and close_arr[i] > prev_short:
                dir_value = 1
            elif not np.isnan(prev_long) and not np.isnan(close_arr[i]) and close_arr[i] < prev_long:
                dir_value = -1
            direction[i] = dir_value

        return (
            pd.Series(long_stop, index=data.index, name='long_stop'),
            pd.Series(short_stop, index=data.index, name='short_stop'),
            pd.Series(direction, index=data.index, name='trend_direction')
        )

    @staticmethod
    def _calculate_heikin_ashi(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """计算Heikin Ashi数据"""
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4.0
        ha_open = ha_close.copy()
        if len(data) > 0:
            ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2.0
            for i in range(1, len(data)):
                ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)
        return ha_open, ha_close, ha_high, ha_low

    @staticmethod
    def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
        return (fast > slow) & (fast.shift(1) <= slow.shift(1))

    @staticmethod
    def _crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
        return (fast < slow) & (fast.shift(1) >= slow.shift(1))

    @staticmethod
    def _build_position_series(entry_condition: pd.Series,
                               exit_condition: pd.Series,
                               price_series: pd.Series,
                               stop_loss_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """根据条件构造持仓序列"""
        n = len(entry_condition)
        position = np.zeros(n, dtype=int)
        entry_flags = np.zeros(n, dtype=int)
        exit_flags = np.zeros(n, dtype=int)
        stop_flags = np.zeros(n, dtype=int)
        in_position = False
        entry_price = None

        for i in range(n):
            entry_active = bool(entry_condition.iloc[i]) if not pd.isna(entry_condition.iloc[i]) else False
            exit_active = bool(exit_condition.iloc[i]) if not pd.isna(exit_condition.iloc[i]) else False
            curr_price = price_series.iloc[i] if i < len(price_series) else np.nan

            if not in_position and entry_active:
                in_position = True
                entry_flags[i] = 1
                entry_price = curr_price if not pd.isna(curr_price) else None

            if in_position and exit_active:
                in_position = False
                exit_flags[i] = 1
                entry_price = None

            elif in_position and stop_loss_pct > 0 and entry_price and not pd.isna(curr_price):
                threshold = entry_price * (1 - stop_loss_pct / 100.0)
                if curr_price <= threshold:
                    in_position = False
                    exit_flags[i] = 1
                    stop_flags[i] = 1
                    entry_price = None

            position[i] = 1 if in_position else 0

        return position, entry_flags, exit_flags, stop_flags

    @staticmethod
    def _build_entry_reasons(data: pd.DataFrame, entry_flags: np.ndarray) -> pd.Series:
        reasons = [''] * len(data)
        for idx, flag in enumerate(entry_flags):
            if flag:
                row = data.iloc[idx]
                parts = []
                if row.get('golden_cross'):
                    parts.append("RSI金叉")
                elif row.get('rsi_relaxed_condition'):
                    parts.append("RSI多头延续")
                else:
                    parts.append("RSI多头")
                if row.get('is_heikin_bullish'):
                    parts.append("Heikin Ashi 阳线")
                if row.get('trend_direction') == 1:
                    parts.append("ATR趋势多头")
                reasons[idx] = ' + '.join(parts)
        return pd.Series(reasons, index=data.index)

    @staticmethod
    def _build_exit_reasons(data: pd.DataFrame, exit_flags: np.ndarray) -> pd.Series:
        reasons = [''] * len(data)
        for idx, flag in enumerate(exit_flags):
            if flag:
                parts = ["ATR趋势转空"]
                if data.iloc[idx].get('death_cross'):
                    parts.append("RSI死叉确认")
                if data.iloc[idx].get('exit_ma_filter_break'):
                    parts.append("EMA16>MA45下连续3日跌破MA16")
                if data.iloc[idx].get('stop_loss_exit'):
                    stop_val = data.iloc[idx].get('stop_loss_pct')
                    if stop_val and not pd.isna(stop_val):
                        parts.append(f"触发{stop_val:.1f}%止损")
                    else:
                        parts.append("触发止损")
                reasons[idx] = ' + '.join(parts)
        return pd.Series(reasons, index=data.index)

    @staticmethod
    def _find_row_by_date(df: pd.DataFrame, target_date) -> Optional[pd.Series]:
        """根据日期字符串查找行"""
        if 'date' in df.columns:
            mask = df['date'].astype(str).str.split().str[0] == str(target_date).split()[0]
            if mask.any():
                return df.loc[mask].iloc[0]
        else:
            mask = df.index.astype(str).str.split().str[0] == str(target_date).split()[0]
            if mask.any():
                return df.loc[mask].iloc[0]
        return None

    @staticmethod
    def _compute_zigzag_lows(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """简单ZigZag低点检测：基于局部最小值"""
        lows = data['low'].to_numpy()
        n = len(lows)
        last_low = np.nan
        prev_low = np.nan
        last_arr = np.full(n, np.nan)
        prev_arr = np.full(n, np.nan)

        for i in range(2, n):
            l_prev = lows[i - 1]
            if np.isnan(l_prev) or np.isnan(lows[i]) or np.isnan(lows[i - 2]):
                last_arr[i] = last_low
                prev_arr[i] = prev_low
                continue

            if l_prev <= lows[i] and l_prev <= lows[i - 2]:
                prev_low = last_low
                last_low = l_prev

            last_arr[i] = last_low
            prev_arr[i] = prev_low

        return (
            pd.Series(last_arr, index=data.index),
            pd.Series(prev_arr, index=data.index)
        )
