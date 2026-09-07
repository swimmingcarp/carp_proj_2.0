"""Point-in-time RSI trend strategy used by live signals and offline backtests."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .indicators import (
        atr_indicator,
        ema_indicator,
        ma_indicator,
        macd_indicator,
        rsi_indicator,
    )
    from .strategy import StrategyBase
except ImportError:  # pragma: no cover - standalone execution
    from indicators import (
        atr_indicator,
        ema_indicator,
        ma_indicator,
        macd_indicator,
        rsi_indicator,
    )
    from strategy import StrategyBase


logger = logging.getLogger(__name__)


class RSITrendStrategy(StrategyBase):
    """RSI continuation entries with ATR trend and shared risk exits."""

    def __init__(
        self, config: Optional[Dict] = None, market: str = "CN-A", stock_code: str = ""
    ):
        fee_keys = {
            "commission_enabled",
            "commission_cn_a",
            "commission_hk",
            "commission_us",
            "commission_min_cn",
            "stamp_duty_cn",
            "stamp_duty_hk",
        }
        fee_config = {
            key: value for key, value in (config or {}).items() if key in fee_keys
        }

        super().__init__(
            config=fee_config,
            market=market,
            stock_code=stock_code,
        )

    def analyze(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Calculate point-in-time signals and the resulting holding state."""
        if df is None or len(df) == 0:
            logger.warning("RSITrendStrategy: empty input")
            return None, None

        data = self._prepare_dataframe(df)
        close = data["close"]

        data["fast_rsi"] = rsi_indicator(close, period=30)
        data["slow_rsi"] = rsi_indicator(close, period=65)
        data["golden_cross"] = self._crossover(data["fast_rsi"], data["slow_rsi"])
        data["rsi_diff"] = data["fast_rsi"] - data["slow_rsi"]
        data["rsi_momentum"] = data["fast_rsi"] - data["fast_rsi"].shift(3)

        if "volume" in data.columns:
            data["volume_ma20"] = data["volume"].rolling(20, min_periods=1).mean()
            data["volume_weak"] = data["volume"] < data["volume_ma20"] * 0.75
        else:
            data["volume_weak"] = False

        data["high_1y"] = data["high"].rolling(250, min_periods=100).max()
        data["position_vs_1y_high"] = (close / data["high_1y"] - 1.0) * 100.0
        data["prev_peak_60_120"] = (
            data["high"].shift(60).rolling(120, min_periods=30).max()
        )
        data["is_m_top"] = (
            close.div(data["prev_peak_60_120"]).between(0.92, 1.05)
            & (data["position_vs_1y_high"] < -3.0)
        ).fillna(False)

        atr_trailing = atr_indicator(data, period=20) * 3.0
        data["atr_trailing"] = atr_trailing
        data["atr"] = atr_indicator(data, period=14)
        data["atr_ma20"] = data["atr"].rolling(20, min_periods=10).mean()
        data["atr_expanding"] = (data["atr"] > data["atr_ma20"] * 1.5).fillna(False)

        long_stop, short_stop, direction = self._compute_trend_levels(
            data,
            atr_trailing,
            20,
            True,
        )
        data["long_stop"] = long_stop
        data["short_stop"] = short_stop
        data["trend_direction"] = direction

        ha_open, ha_close, ha_high, ha_low = self._calculate_heikin_ashi(data)
        data["ha_open"] = ha_open
        data["ha_close"] = ha_close
        data["ha_high"] = ha_high
        data["ha_low"] = ha_low
        data["is_heikin_bullish"] = ha_close > ha_open

        macd_diff, _, macd_hist = macd_indicator(close)
        data["macd_diff"] = macd_diff
        data["macd_hist"] = macd_hist

        data["ma_60"] = close.rolling(60).mean()
        data["dist_ma60"] = (close / data["ma_60"] - 1.0) * 100.0

        self._add_exit_ma_state(data)
        self._add_long_slope_state(data)

        data["impulse_trend"] = self._detect_impulse_trend(data)
        rolling_high = close.rolling(120, min_periods=1).max()
        rolling_low = close.rolling(120, min_periods=1).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        data["price_position"] = ((close - rolling_low) / price_range).fillna(0.5)

        data["rsi_relaxed_condition"] = data["rsi_diff"] >= 2.0

        common = (
            (direction == 1)
            & data["is_heikin_bullish"]
            & (~data["volume_weak"])
            & (~data["is_m_top"])
            & (~data["atr_expanding"])
        )
        standard_entry = common & (data["golden_cross"] | data["rsi_relaxed_condition"])
        discount_entry = (
            (data["price_position"] < 0.25)
            & (direction == 1)
            & data["is_heikin_bullish"]
        )
        momentum_entry = (
            common
            & (data["rsi_momentum"] > 10.0)
            & (data["fast_rsi"] < 60.0)
            & (data["fast_rsi"].shift(3) < 40.0)
        )
        # One shared entry-quality gate replaces per-family scoring and routing.
        ma60_block = data["dist_ma60"] > 18.0
        up_streak = self._consecutive_true(close > close.shift(1))
        standard_entry &= ~ma60_block & (up_streak <= 5)
        discount_entry &= ~ma60_block
        momentum_entry &= ~ma60_block & (up_streak <= 5)

        data["standard_entry_raw"] = standard_entry
        data["discount_zone_entry"] = discount_entry
        data["rsi_momentum_entry"] = momentum_entry

        entry_condition = standard_entry | discount_entry | momentum_entry
        entry_condition = self._apply_regime_filters(
            data,
            entry_condition,
            discount_entry,
        )

        basic_exit = direction != 1
        basic_exit &= ~data["exit_strong_ma_trend"] | data["exit_ma_filter_break"]
        recent_impulse = data["impulse_trend"].rolling(5, min_periods=1).sum() > 0
        impulse_protection = (
            recent_impulse
            & (data["exit_ema_fast"] > data["exit_ma_slow"])
            & (close > data["exit_ma_slow"] * 0.90)
        ).fillna(False)
        exit_condition = basic_exit & ~impulse_protection

        (
            position,
            entry_flags,
            exit_flags,
            stop_flags,
            profit_flags,
            entry_reasons,
            exit_reasons,
        ) = self._build_position_series(entry_condition, exit_condition, data)

        data["buy_signal"] = position
        data["entry_signal"] = entry_flags
        data["exit_signal"] = exit_flags
        data["stop_loss_exit"] = stop_flags
        data["profit_target_exit"] = profit_flags
        data["entry_reason"] = entry_reasons
        data["exit_reason"] = exit_reasons
        return data, None

    def _add_exit_ma_state(self, data: pd.DataFrame) -> None:
        close = data["close"]
        fast = ema_indicator(close, period=20)
        slow = ma_indicator(close, period=40)
        confirm = ma_indicator(close, period=15)
        strong = fast > slow
        below_streak = self._consecutive_true(close < confirm)

        data["exit_ema_fast"] = fast
        data["exit_ma_slow"] = slow
        data["exit_ma_confirm"] = confirm
        data["exit_close_below_ma_confirm"] = close < confirm
        data["exit_ma_confirm_below_streak"] = below_streak
        data["exit_ma_filter_break"] = strong & (below_streak >= 3)
        data["exit_strong_ma_trend"] = strong
        data["exit_ma_filter_active"] = strong & (below_streak < 3)

    def _add_long_slope_state(self, data: pd.DataFrame) -> None:
        log_close = np.log(data["close"].where(data["close"] > 0))
        slopes = pd.Series(np.nan, index=data.index, dtype=float)
        pearsons = pd.Series(np.nan, index=data.index, dtype=float)
        period = 180
        for i in range(period - 1, len(data)):
            values = log_close.iloc[i - period + 1 : i + 1].to_numpy(dtype=float)
            slope, pearson = self._slope_and_pearson(values)
            slopes.iloc[i] = slope
            pearsons.iloc[i] = pearson
        data["ultra_long_slope"] = slopes
        data["ultra_long_channel_pearson"] = pearsons

    def _apply_regime_filters(
        self,
        data: pd.DataFrame,
        entry: pd.Series,
        discount: pd.Series,
    ) -> pd.Series:
        result = entry.copy()

        # Do not mean-revert against a strongly persistent price path.
        hurst = pd.Series(0.5, index=data.index, dtype=float)
        for i in range(100, len(data), 20):
            value = self._calculate_hurst_exponent(data["close"].iloc[:i], 100)
            hurst.iloc[i : min(i + 20, len(data))] = value
        data["hurst_exponent"] = hurst
        result &= ~((hurst > 0.65) & discount)

        downtrend = (
            (data["ultra_long_slope"] < -0.003)
            & (data["ultra_long_channel_pearson"] > 0.70)
        ).fillna(False)
        result &= ~downtrend

        return result

    def _build_position_series(
        self, entry_condition: pd.Series, exit_condition: pd.Series, data: pd.DataFrame
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]
    ]:
        n = len(data)
        position = np.zeros(n, dtype=int)
        entry_flags = np.zeros(n, dtype=int)
        exit_flags = np.zeros(n, dtype=int)
        stop_flags = np.zeros(n, dtype=int)
        profit_flags = np.zeros(n, dtype=int)
        entry_reasons = [""] * n
        exit_reasons = [""] * n

        holding = False
        entry_price = None
        max_profit = 0.0
        trailing_active = False
        trailing_pending = False
        trailing_pending_days = 0
        pending_exit = False
        pending_exit_days = 0

        for i in range(n):
            entry_active = (
                bool(entry_condition.iloc[i])
                if not pd.isna(entry_condition.iloc[i])
                else False
            )
            exit_active = (
                bool(exit_condition.iloc[i])
                if not pd.isna(exit_condition.iloc[i])
                else False
            )
            price = data["close"].iloc[i]

            if not holding and entry_active:
                holding = True
                entry_flags[i] = 1
                entry_price = price if not np.isnan(price) else None
                entry_reason = self._entry_reason(data, i)
                entry_reasons[i] = entry_reason

                trade_stop = 6.0 if entry_reason in {"RSI金叉", "折价区补仓"} else 8.0
                trailing_trigger = {
                    "RSI金叉": 5.0,
                    "折价区补仓": 5.5,
                }.get(entry_reason, 6.0)
                trailing_level = {
                    "RSI金叉": 2.3,
                    "折价区补仓": 3.5,
                }.get(entry_reason, 1.5)

                max_profit = 0.0
                trailing_active = False
                trailing_pending = False
                trailing_pending_days = 0
                pending_exit = False
                pending_exit_days = 0

            if holding:
                if entry_price and not pd.isna(price) and entry_price > 0:
                    profit = (price / entry_price - 1.0) * 100.0
                    if profit > max_profit:
                        max_profit = profit

                    stop_threshold = entry_price * (1.0 - trade_stop / 100.0)
                    if price <= stop_threshold:
                        holding = False
                        exit_flags[i] = 1
                        stop_flags[i] = 1
                        exit_reasons[i] = f"硬性止损上限({trade_stop:.1f}%)"
                        entry_price = None
                        trailing_active = False
                        max_profit = 0.0
                        pending_exit = False
                        position[i] = 0
                        continue

                    if not trailing_active and max_profit >= trailing_trigger:
                        trailing_active = True
                    if trailing_active:
                        trailing_threshold = entry_price * (
                            1.0 + trailing_level / 100.0
                        )
                        if price <= trailing_threshold:
                            if not trailing_pending:
                                trailing_pending = True
                                trailing_pending_days = 0
                            else:
                                trailing_pending_days += 1
                            if trailing_pending_days >= 1:
                                holding = False
                                exit_flags[i] = 1
                                if profit < 0:
                                    stop_flags[i] = 1
                                    exit_reasons[i] = "Trailing止损-确认(1日)"
                                else:
                                    profit_flags[i] = 1
                                    exit_reasons[i] = "Trailing止盈-确认(1日)"
                                entry_price = None
                                trailing_active = False
                                max_profit = 0.0
                                pending_exit = False
                                trailing_pending = False
                                trailing_pending_days = 0
                                position[i] = 0
                                continue
                        elif trailing_pending:
                            trailing_pending = False
                            trailing_pending_days = 0

                    if not exit_active and profit >= 24.0 and "volume" in data.columns:
                        volume = data["volume"].iloc[i]
                        volume_ma = data["volume_ma20"].iloc[i]
                        close = data["close"].iloc[i]
                        open_price = data["open"].iloc[i]
                        is_distribution = (
                            not np.isnan(volume)
                            and not np.isnan(volume_ma)
                            and volume_ma > 0
                            and volume > volume_ma * 2.2
                            and close < open_price
                        )
                        if is_distribution and i >= 12:
                            ma13 = np.mean(data["close"].iloc[i - 12 : i + 1].values)
                            deviation = (
                                (close - ma13) / ma13 * 100.0 if ma13 > 0 else 0.0
                            )
                            if deviation >= 24.0:
                                holding = False
                                exit_flags[i] = 1
                                exit_reasons[i] = f"放量阴线+偏离MA13({deviation:.1f}%)"
                                entry_price = None
                                trailing_active = False
                                max_profit = 0.0
                                pending_exit = False
                                pending_exit_days = 0
                                position[i] = 0
                                continue

                if pending_exit:
                    if not exit_active:
                        pending_exit = False
                        pending_exit_days = 0

                if pending_exit:
                    pending_exit_days += 1
                    if pending_exit_days >= 1:
                        holding = False
                        exit_flags[i] = 1
                        exit_reasons[i] = "趋势转空-延迟1日退出"
                        entry_price = None
                        pending_exit = False
                        pending_exit_days = 0
                elif exit_active:
                    previous_close = data["close"].iloc[i - 1] if i > 0 else price
                    day_change = (
                        (price / previous_close - 1.0) * 100.0
                        if previous_close > 0
                        else 0.0
                    )
                    if i > 0 and day_change < -2.6:
                        pending_exit = True
                        pending_exit_days = 0
                    else:
                        holding = False
                        exit_flags[i] = 1
                        exit_reasons[i] = "趋势转空退出"
                        entry_price = None

            position[i] = 1 if holding else 0

        return (
            position,
            entry_flags,
            exit_flags,
            stop_flags,
            profit_flags,
            entry_reasons,
            exit_reasons,
        )

    @staticmethod
    def _entry_reason(data: pd.DataFrame, index: int) -> str:
        if bool(data["rsi_momentum_entry"].iloc[index]):
            return "RSI动量加速"
        if bool(data["discount_zone_entry"].iloc[index]):
            return "折价区补仓"
        if bool(data["golden_cross"].iloc[index]):
            return "RSI金叉"
        return "RSI多头延续"

    def _detect_impulse_trend(self, data: pd.DataFrame) -> pd.Series:
        """Identify a recent accelerating uptrend that merits exit confirmation."""
        result = pd.Series(False, index=data.index)

        for i in range(3, len(data)):
            gain_ok = data["close"].iloc[i] / data["close"].iloc[i - 3] - 1.0 >= 0.15
            rsi_ok = data["fast_rsi"].iloc[i] <= 80.0
            if i >= 20 and "volume" in data.columns:
                volume_ok = (
                    data["volume"].iloc[i]
                    >= data["volume"].iloc[i - 20 : i].mean() * 1.2
                )
            else:
                volume_ok = True
            trend_ok = data["trend_direction"].iloc[i] == 1
            ma_ok = data["exit_ema_fast"].iloc[i] > data["exit_ma_slow"].iloc[i]
            price_ok = data["close"].iloc[i] > data["exit_ma_slow"].iloc[i] * 0.95
            continuity_ok = data["close"].iloc[i] > data["close"].iloc[i - 1]
            result.iloc[i] = (
                trend_ok
                and ma_ok
                and sum((gain_ok, rsi_ok, volume_ok, price_ok, continuity_ok)) >= 3
            )
        return result

    @staticmethod
    def _calculate_hurst_exponent(prices: pd.Series, window: int) -> float:
        if len(prices) < window:
            return 0.5
        values = prices.iloc[-window:].to_numpy(dtype=float)
        if np.any(values <= 0) or np.isnan(values).any():
            return 0.5
        returns = np.diff(np.log(values))
        lags = list(range(2, min(20, len(returns) // 2)))
        rs_values = []
        used_lags = []
        for lag in lags:
            samples = []
            for start in range(0, len(returns) - lag + 1, lag):
                sample = returns[start : start + lag]
                deviation = np.cumsum(sample - sample.mean())
                spread = deviation.max() - deviation.min()
                standard_deviation = sample.std()
                if standard_deviation > 0:
                    samples.append(spread / standard_deviation)
            if samples:
                used_lags.append(lag)
                rs_values.append(np.mean(samples))
        if len(rs_values) < 2 or np.any(np.asarray(rs_values) <= 0):
            return 0.5
        slope = np.polyfit(np.log(used_lags), np.log(rs_values), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))

    @staticmethod
    def _slope_and_pearson(values: np.ndarray) -> Tuple[float, float]:
        if len(values) < 2 or np.isnan(values).any():
            return np.nan, np.nan
        x = np.arange(len(values), dtype=float)
        slope = np.polyfit(x, values, 1)[0]
        pearson = np.corrcoef(values, x)[0, 1]
        return float(slope), float(abs(pearson))

    @staticmethod
    def _consecutive_true(condition: pd.Series) -> pd.Series:
        values = condition.fillna(False).to_numpy(dtype=bool)
        result = np.zeros(len(values), dtype=int)
        for i, value in enumerate(values):
            if value:
                result[i] = result[i - 1] + 1 if i else 1
        return pd.Series(result, index=condition.index)

    @staticmethod
    def _compute_trend_levels(
        data: pd.DataFrame, atr_values: pd.Series, period: int, use_close: bool
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        source_high = data["close"] if use_close else data["high"]
        source_low = data["close"] if use_close else data["low"]
        long_base = source_high.rolling(period).max() - atr_values
        short_base = source_low.rolling(period).min() + atr_values

        close = data["close"].to_numpy(dtype=float)
        long_values = long_base.to_numpy(dtype=float)
        short_values = short_base.to_numpy(dtype=float)
        long_stop = np.full(len(data), np.nan)
        short_stop = np.full(len(data), np.nan)
        direction = np.ones(len(data), dtype=int)

        for i in range(len(data)):
            previous_long = long_stop[i - 1] if i else long_values[i]
            previous_short = short_stop[i - 1] if i else short_values[i]
            if np.isnan(previous_long):
                previous_long = long_values[i]
            if np.isnan(previous_short):
                previous_short = short_values[i]

            candidate_long = long_values[i]
            if i and np.isfinite(previous_long) and close[i - 1] > previous_long:
                candidate_long = (
                    previous_long
                    if np.isnan(candidate_long)
                    else max(candidate_long, previous_long)
                )
            long_stop[i] = previous_long if np.isnan(candidate_long) else candidate_long

            candidate_short = short_values[i]
            if i and np.isfinite(previous_short) and close[i - 1] < previous_short:
                candidate_short = (
                    previous_short
                    if np.isnan(candidate_short)
                    else min(candidate_short, previous_short)
                )
            short_stop[i] = (
                previous_short if np.isnan(candidate_short) else candidate_short
            )

            previous_direction = direction[i - 1] if i else 1
            if np.isfinite(previous_short) and close[i] > previous_short:
                direction[i] = 1
            elif np.isfinite(previous_long) and close[i] < previous_long:
                direction[i] = -1
            else:
                direction[i] = previous_direction

        return (
            pd.Series(long_stop, index=data.index),
            pd.Series(short_stop, index=data.index),
            pd.Series(direction, index=data.index),
        )

    @staticmethod
    def _calculate_heikin_ashi(
        data: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        ha_close = (data["open"] + data["high"] + data["low"] + data["close"]) / 4.0
        ha_open = ha_close.copy()
        if len(data):
            ha_open.iloc[0] = (data["open"].iloc[0] + data["close"].iloc[0]) / 2.0
            for i in range(1, len(data)):
                ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        ha_high = pd.concat((data["high"], ha_open, ha_close), axis=1).max(axis=1)
        ha_low = pd.concat((data["low"], ha_open, ha_close), axis=1).min(axis=1)
        return ha_open, ha_close, ha_high, ha_low

    @staticmethod
    def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
        return (fast > slow) & (fast.shift(1) <= slow.shift(1))

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if "date" in data.columns:
            data = data.sort_values("date").reset_index(drop=True)
        else:
            data = data.sort_index()
            data["date"] = data.index
            data = data.reset_index(drop=True)
        missing = [
            name
            for name in ("open", "high", "low", "close")
            if name not in data.columns
        ]
        if missing:
            raise ValueError(
                f"RSI trend strategy missing columns: {', '.join(missing)}"
            )
        return data

    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        if df is None or len(df) == 0:
            return {"signal": "NO_DATA", "reason": "数据不足"}

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        is_new_entry = (
            latest.get("entry_signal", 0) == 1 and previous.get("buy_signal", 0) == 0
        )
        if is_new_entry:
            signal = "BUY"
            reason = str(latest.get("entry_reason", "RSI趋势买入"))
            strength = 3
        elif latest.get("exit_signal", 0) == 1 or (
            previous.get("buy_signal", 0) == 1 and latest.get("buy_signal", 0) == 0
        ):
            signal = "SELL"
            reason = str(latest.get("exit_reason", "ATR趋势转空"))
            strength = 4 if latest.get("stop_loss_exit", 0) else 3
        elif latest.get("buy_signal", 0) == 1:
            signal = "HOLD_BUY"
            reason = "RSI与ATR多头持仓中"
            strength = 3
        else:
            signal = "HOLD"
            reason = "等待RSI趋势入场"
            strength = 1

        return {
            "signal": signal,
            "strength": strength,
            "reason": reason,
            "price": latest.get("close", 0),
            "date": latest.get("date"),
            "fast_rsi": latest.get("fast_rsi"),
            "slow_rsi": latest.get("slow_rsi"),
            "trend_direction": latest.get("trend_direction"),
            "ha_bullish": latest.get("is_heikin_bullish"),
            "k": latest.get("fast_rsi", 0),
            "d": latest.get("slow_rsi", 0),
            "macd": latest.get("macd_hist", 0.0),
            "diff": latest.get("macd_diff", 0.0),
            "ema_16": latest.get("exit_ema_fast", np.nan),
            "ma_16": latest.get("exit_ma_confirm", np.nan),
            "ma_45": latest.get("exit_ma_slow", np.nan),
        }

    def get_trading_signals(
        self, df: pd.DataFrame, initial_capital: float = 10000.0
    ) -> Dict:
        base_result = {
            "buy_points": [],
            "sell_points": [],
            "total_trades": 0,
            "initial_capital": initial_capital,
        }
        if df is None or "buy_signal" not in df.columns:
            return base_result
        result = super().backtest(df, initial_capital)
        if not result or not result.get("trades"):
            return base_result

        buy_points: List[Dict] = []
        sell_points: List[Dict] = []
        date_rows = {str(row["date"]): row for _, row in df.iterrows()}
        last_open = bool(df.iloc[-1]["buy_signal"] == 1)
        trades = result["trades"]
        for index, trade in enumerate(trades):
            buy_row = date_rows.get(str(trade["buy_date"]))
            sell_row = date_rows.get(str(trade["sell_date"]))
            if buy_row is not None:
                buy_points.append(
                    {
                        "date": trade["buy_date"],
                        "price": trade["buy_price"],
                        "reason": buy_row.get("entry_reason", "RSI趋势买入"),
                        "fast_rsi": buy_row.get("fast_rsi"),
                        "slow_rsi": buy_row.get("slow_rsi"),
                        "trend_direction": buy_row.get("trend_direction"),
                        "ha_bullish": buy_row.get("is_heikin_bullish"),
                        "commission": trade.get("buy_commission", 0),
                    }
                )
            if sell_row is not None:
                is_open = index == len(trades) - 1 and last_open
                sell_points.append(
                    {
                        "date": trade["sell_date"],
                        "price": trade["sell_price"],
                        "reason": (
                            "持仓中"
                            if is_open
                            else sell_row.get("exit_reason", "ATR趋势转空")
                        ),
                        "fast_rsi": sell_row.get("fast_rsi"),
                        "slow_rsi": sell_row.get("slow_rsi"),
                        "trend_direction": sell_row.get("trend_direction"),
                        "ha_bullish": sell_row.get("is_heikin_bullish"),
                        "is_open": is_open,
                        "commission": trade.get("sell_commission", 0),
                    }
                )
        return {
            "buy_points": buy_points,
            "sell_points": sell_points,
            "total_trades": len(trades),
            "initial_capital": initial_capital,
            "trades": trades,
            "gross_return": result.get("gross_return", 0),
        }

    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        result = super().backtest(df, initial_capital)
        trades = result.get("trades", [])
        positive = [trade for trade in trades if trade["profit_rate"] > 0]
        negative = [trade for trade in trades if trade["profit_rate"] < 0]
        total_profit = sum(trade["profit_rate"] for trade in positive)
        total_loss = abs(sum(trade["profit_rate"] for trade in negative))
        result["profit_factor"] = (
            total_profit / total_loss
            if total_loss
            else (999.0 if total_profit else 0.0)
        )
        result["win_count"] = len(positive)
        result["lose_count"] = len(negative)
        result["total_profit_pct"] = total_profit
        result["total_loss_pct"] = total_loss
        return result
