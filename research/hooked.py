"""Hookable re-implementation of the formal strategy's analyze().

`HookedStrategy` reproduces `RSITrendStrategy.analyze()` bar for bar while exposing hook points so a
candidate can change one thing without forking the whole strategy. `harness.py --selfcheck` asserts the
equivalence; if it fails, the formal strategy has changed and this file must be resynchronised before
any candidate result can be trusted.

Hooks:
  chase_gates(data, close, direction)        -> (block_standard, block_discount, block_momentum)
  extra_entry_block(data, close, direction)  -> bool Series, blocks every entry route
  extra_entry_route(data, close, direction, common) -> bool Series, an extra entry route
  extra_exit(data, close, direction)         -> bool Series, an extra exit trigger
  discount_route(data, close, direction)     -> bool Series, replaces the discount entry route
  volume_weak_gate(data)                     -> bool Series, replaces the low-volume gate
  exit_mask(data, close, direction)          -> bool Series, gates the trend exit
  _build_position_series(...)                -> full position state machine
Class attributes:
  REENTRY_FRESH_TREND, STOP_MODE, TRAILING, WINNER_K, REENTRY_S_K, REGIME_STOP
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT = os.environ.get("CARP_ROOT", "/home/jaden/carp/carp_proj_2.0")
APP = os.path.join(PROJECT, "stock_trading_advisor")
DATA_DIR = os.path.join(APP, "data", "backtest_data")
if os.path.join(APP, "src") not in sys.path:
    sys.path.insert(0, os.path.join(APP, "src"))

from new_strategy import RSITrendStrategy  # noqa: E402
from indicators import (  # noqa: E402,F401
    atr_indicator,
    ema_indicator,
    ma_indicator,
    macd_indicator,
    rsi_indicator,
)


class HookedStrategy(RSITrendStrategy):
    """Re-implementation of analyze() with hook points; identical to the formal strategy when hooks are no-ops."""
    REENTRY_FRESH_TREND = False   # after a stop/trailing exit, wait for direction to flip -1 then back to +1
    STOP_MODE = "fixed"           # fixed | none
    TRAILING = True
    WINNER_K = None               # Wilcox-Crittenden winner mode: once profit >= 2x initial risk, exit only on close < HighestClose - K*ATR42
    REENTRY_S_K = None
    STOP_ATR_K = None            # if set, the hard stop is K x the stock's own daily ATR% at entry
    STOP_WIDE_MODE = None        # "ma120"/"stack"/"nearhigh": widen ONLY inside a confirmed advance
    ANTICIPATION_ENABLED = True  # mirror the production overlay; candidates with their own set False
    KL_ARM = None                # bars armed for re-entry after a HARD STOP (Kaminski-Lo asymmetry)
    KL_MODE = "up"               # "up": first close above the prior close; "delta": > KL_DELTA; "ma20": reclaim MA20
    KL_DELTA = 0.0
    STOP_ATR_CLIP = (3.0, 25.0)  # floor/ceiling in percent, so the ATR scaling cannot run away            # Sepp C4(b): after a trend exit with (EMA20-EMA175)/ATR20 > K, allow RSI-free re-entry within 60 bars
    NO_ATRX = False               # if True the ATR-expansion gate never blocks an entry (judge phase)
    REGIME_STOP = None            # if set and data["regime_off"] is True on the entry day, use this hard stop (%) instead of 6/8

    # --- hooks ---
    def chase_gates(self, data, close, direction):
        """Return (block_standard, block_discount, block_momentum) boolean Series. Baseline: MA60 distance + up streak."""
        ma60_block = data["dist_ma60"] > 18.0
        up_streak = self._consecutive_true(close > close.shift(1))
        streak_block = up_streak > 5
        return ma60_block | streak_block, ma60_block, ma60_block | streak_block

    def extra_entry_block(self, data, close, direction):
        return pd.Series(False, index=data.index)

    def volume_weak_gate(self, data):
        return data["volume"] < data["volume_ma20"] * 0.75

    def exit_mask(self, data, close, direction):
        """Baseline protection: a trend exit is suppressed while EMA20>MA40 unless close has been below MA15 for 3 days."""
        return (~data["exit_strong_ma_trend"]) | data["exit_ma_filter_break"]

    def extra_exit(self, data, close, direction):
        """Additional exit trigger OR-ed into the exit condition (default none)."""
        return pd.Series(False, index=data.index)

    def extra_entry_route(self, data, close, direction, common):
        """Additional entry route OR-ed into the entry condition (reason falls back to 'RSI多头延续': 8% stop, 6/1.5 lock)."""
        return pd.Series(False, index=data.index)

    def discount_route(self, data, close, direction):
        """Baseline discount route: 120-day range position < 0.25 in an intact chandelier uptrend with HA bullish."""
        return ((data["price_position"] < 0.25) & (direction == 1) & data["is_heikin_bullish"])

    def analyze(self, df):
        if df is None or len(df) == 0:
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
            data["volume_weak"] = self.volume_weak_gate(data)
        else:
            data["volume_weak"] = False
        data["high_1y"] = data["high"].rolling(250, min_periods=100).max()
        data["position_vs_1y_high"] = (close / data["high_1y"] - 1.0) * 100.0
        data["prev_peak_60_120"] = data["high"].shift(60).rolling(120, min_periods=30).max()
        data["is_m_top"] = (close.div(data["prev_peak_60_120"]).between(0.92, 1.05) & (data["position_vs_1y_high"] < -3.0)).fillna(False)
        atr_trailing = atr_indicator(data, period=20) * 3.0
        data["atr_trailing"] = atr_trailing
        data["atr"] = atr_indicator(data, period=14)
        data["atr_ma20"] = data["atr"].rolling(20, min_periods=10).mean()
        data["atr_expanding"] = (data["atr"] > data["atr_ma20"] * 1.5).fillna(False)
        if getattr(self, "NO_ATRX", False):
            data["atr_expanding"] = pd.Series(False, index=data.index)
        long_stop, short_stop, direction = self._compute_trend_levels(data, atr_trailing, 20, True)
        data["long_stop"] = long_stop; data["short_stop"] = short_stop; data["trend_direction"] = direction
        ha_open, ha_close, ha_high, ha_low = self._calculate_heikin_ashi(data)
        data["ha_open"] = ha_open; data["ha_close"] = ha_close; data["ha_high"] = ha_high; data["ha_low"] = ha_low
        data["is_heikin_bullish"] = ha_close > ha_open
        macd_diff, _, macd_hist = macd_indicator(close)
        data["macd_diff"] = macd_diff; data["macd_hist"] = macd_hist
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
        common = ((direction == 1) & data["is_heikin_bullish"] & (~data["volume_weak"]) & (~data["is_m_top"]) & (~data["atr_expanding"]))
        standard_entry = common & (data["golden_cross"] | data["rsi_relaxed_condition"])
        discount_entry = self.discount_route(data, close, direction)
        momentum_entry = (common & (data["rsi_momentum"] > 10.0) & (data["fast_rsi"] < 60.0) & (data["fast_rsi"].shift(3) < 40.0))
        b_std, b_disc, b_mom = self.chase_gates(data, close, direction)
        standard_entry &= ~b_std; discount_entry &= ~b_disc; momentum_entry &= ~b_mom
        xb = self.extra_entry_block(data, close, direction).fillna(False)
        standard_entry &= ~xb; discount_entry &= ~xb; momentum_entry &= ~xb
        extra_route = self.extra_entry_route(data, close, direction, common).fillna(False) & ~b_std & ~xb
        data["standard_entry_raw"] = standard_entry; data["discount_zone_entry"] = discount_entry; data["rsi_momentum_entry"] = momentum_entry
        data["extra_route_entry"] = extra_route
        data["atr42"] = atr_indicator(data, period=42)
        entry_condition = standard_entry | discount_entry | momentum_entry | extra_route
        entry_condition = self._apply_regime_filters(data, entry_condition, discount_entry)
        basic_exit = direction != 1
        basic_exit &= self.exit_mask(data, close, direction)
        data["reentry_ok"] = (common & ~b_std & ~xb).fillna(False)
        recent_impulse = data["impulse_trend"].rolling(5, min_periods=1).sum() > 0
        impulse_protection = (recent_impulse & (data["exit_ema_fast"] > data["exit_ma_slow"]) & (close > data["exit_ma_slow"] * 0.90)).fillna(False)
        exit_condition = (basic_exit & ~impulse_protection) | self.extra_exit(data, close, direction).fillna(False)
        (position, entry_flags, exit_flags, stop_flags, profit_flags, entry_reasons, exit_reasons) = self._build_position_series(entry_condition, exit_condition, data)
        if self.ANTICIPATION_ENABLED:
            (position, entry_flags, exit_flags, stop_flags, entry_reasons, exit_reasons) = \
                self._apply_anticipation_overlay(
                    position, entry_flags, exit_flags, stop_flags, entry_reasons, exit_reasons, data)
        data["buy_signal"] = position; data["entry_signal"] = entry_flags; data["exit_signal"] = exit_flags
        data["stop_loss_exit"] = stop_flags; data["profit_target_exit"] = profit_flags
        data["entry_reason"] = entry_reasons; data["exit_reason"] = exit_reasons
        return data, None

    def _build_position_series(self, entry_condition, exit_condition, data):
        n = len(data)
        position = np.zeros(n, dtype=int); entry_flags = np.zeros(n, dtype=int); exit_flags = np.zeros(n, dtype=int)
        stop_flags = np.zeros(n, dtype=int); profit_flags = np.zeros(n, dtype=int)
        entry_reasons = [""] * n; exit_reasons = [""] * n
        close_arr = data["close"].to_numpy(float); dir_arr = data["trend_direction"].to_numpy()
        atr42_arr = data["atr42"].to_numpy(float) if "atr42" in data.columns else np.full(n, np.nan)
        entry_arr = entry_condition.fillna(False).to_numpy(bool); exit_arr = exit_condition.fillna(False).to_numpy(bool)
        holding = False; entry_price = None; max_profit = 0.0
        hc = 0.0; winner = False; wstop = -np.inf
        s_arr = data["sepp_s"].to_numpy(float) if "sepp_s" in data.columns else np.full(n, np.nan)
        atrpct_arr = (data["atr"] / data["close"] * 100.0).to_numpy(float) if "atr" in data.columns else np.full(n, np.nan)
        if self.STOP_WIDE_MODE:
            _c = data["close"]; _m120 = _c.rolling(120).mean()
            _rising = (_m120 / _m120.shift(20) - 1) > 0
            if self.STOP_WIDE_MODE == "ma120":
                _ok = (_c > _m120) & _rising
            elif self.STOP_WIDE_MODE == "stack":
                _m20, _m60 = _c.rolling(20).mean(), _c.rolling(60).mean()
                _ok = (_c > _m20) & (_m20 > _m60) & (_m60 > _m120) & _rising
            elif self.STOP_WIDE_MODE == "nearhigh":
                _ok = (_c > _m120) & _rising & (_c / _c.rolling(250, min_periods=120).max() > 0.85)
            else:
                raise ValueError(self.STOP_WIDE_MODE)
            wide_arr = _ok.fillna(False).to_numpy(bool)
        else:
            wide_arr = np.ones(n, bool)
        reentry_ok = data["reentry_ok"].to_numpy(bool) if "reentry_ok" in data.columns else np.zeros(n, bool)
        arm_until = -1
        kl_arm_until = -1
        ma20_arr = data["close"].rolling(20).mean().to_numpy(float)
        prev_close = np.concatenate(([np.nan], close_arr[:-1]))
        trailing_active = trailing_pending = pending_exit = False
        trailing_pending_days = pending_exit_days = 0
        trade_stop = 8.0; trailing_trigger = 6.0; trailing_level = 1.5
        wait_fresh = False   # REENTRY_FRESH_TREND state: set after a stop/trailing exit, cleared once direction has been -1
        for i in range(n):
            if wait_fresh and dir_arr[i] != 1:
                wait_fresh = False
            entry_active = bool(entry_arr[i]) and not wait_fresh
            if self.REENTRY_S_K and not holding and not entry_active and i <= arm_until and reentry_ok[i]:
                entry_active = True
            if self.KL_ARM and not holding and not entry_active and i <= kl_arm_until:
                # Kaminski & Lo (2014) S(gamma, delta, J): the exit reads a cumulative loss but the
                # re-entry reads a SINGLE bar - the asymmetry is the point. Bypasses every entry gate.
                _px = close_arr[i]   # NOT `price`: that is only assigned further down, still holding bar i-1
                if self.KL_MODE == "ma20":
                    _fire = (not np.isnan(ma20_arr[i])) and (not np.isnan(_px)) and _px > ma20_arr[i]
                else:
                    _thr = self.KL_DELTA if self.KL_MODE == "delta" else 0.0
                    _fire = (not np.isnan(prev_close[i])) and prev_close[i] > 0 and (not np.isnan(_px)) \
                        and (_px / prev_close[i] - 1.0) > _thr
                if _fire:
                    entry_active = True; kl_arm_until = -1
            exit_active = bool(exit_arr[i])
            price = close_arr[i]
            if not holding and entry_active:
                holding = True; entry_flags[i] = 1; entry_price = price if not np.isnan(price) else None
                entry_reason = self._entry_reason(data, i); entry_reasons[i] = entry_reason
                trade_stop = 6.0 if entry_reason in {"RSI金叉", "折价区补仓"} else 8.0
                if self.REGIME_STOP and "regime_off" in data.columns and bool(data["regime_off"].iloc[i]): trade_stop = min(trade_stop, self.REGIME_STOP)
                if wide_arr[i] and self.STOP_ATR_K and not np.isnan(atrpct_arr[i]) and atrpct_arr[i] > 0:
                    # root-cause fix under test: a fixed-percent stop is only ~1.55 daily ATRs on the most
                    # volatile stocks (noise level) and ~3.12 on the quietest, so scale it to the stock
                    trade_stop = float(np.clip(self.STOP_ATR_K * atrpct_arr[i], *self.STOP_ATR_CLIP))
                if self.STOP_MODE == "none": trade_stop = 1e9
                trailing_trigger = {"RSI金叉": 5.0, "折价区补仓": 5.5}.get(entry_reason, 6.0)
                trailing_level = {"RSI金叉": 2.3, "折价区补仓": 3.5}.get(entry_reason, 1.5)
                max_profit = 0.0; trailing_active = trailing_pending = pending_exit = False
                trailing_pending_days = pending_exit_days = 0
                hc = price; winner = False; wstop = -np.inf; arm_until = -1
            if holding:
                if entry_price and not np.isnan(price) and entry_price > 0:
                    profit = (price / entry_price - 1.0) * 100.0
                    max_profit = max(max_profit, profit)
                    if price <= entry_price * (1.0 - trade_stop / 100.0):
                        holding = False; exit_flags[i] = 1; stop_flags[i] = 1; exit_reasons[i] = f"硬性止损上限({trade_stop:.1f}%)"
                        kl_arm_until = i + self.KL_ARM if self.KL_ARM else -1
                        entry_price = None; trailing_active = False; max_profit = 0.0; pending_exit = False; position[i] = 0
                        wait_fresh = self.REENTRY_FRESH_TREND
                        continue
                    if self.WINNER_K:
                        hc = max(hc, price)
                        if not winner and max_profit >= 2.0 * trade_stop:
                            winner = True; wstop = -np.inf
                        if winner:
                            if not np.isnan(atr42_arr[i]): wstop = max(wstop, hc - self.WINNER_K * atr42_arr[i])
                            exit_active = False; pending_exit = False; pending_exit_days = 0
                            if price < wstop:
                                holding = False; exit_flags[i] = 1; exit_reasons[i] = "WC宽止损"; entry_price = None
                                trailing_active = False; max_profit = 0.0; position[i] = 0
                                continue
                    if self.TRAILING:
                        if not trailing_active and max_profit >= trailing_trigger:
                            trailing_active = True
                        if trailing_active:
                            if price <= entry_price * (1.0 + trailing_level / 100.0):
                                if not trailing_pending:
                                    trailing_pending = True; trailing_pending_days = 0
                                else:
                                    trailing_pending_days += 1
                                if trailing_pending_days >= 1:
                                    holding = False; exit_flags[i] = 1
                                    if profit < 0: stop_flags[i] = 1; exit_reasons[i] = "Trailing止损-确认(1日)"
                                    else: profit_flags[i] = 1; exit_reasons[i] = "Trailing止盈-确认(1日)"
                                    entry_price = None; trailing_active = False; max_profit = 0.0
                                    pending_exit = False; trailing_pending = False; trailing_pending_days = 0; position[i] = 0
                                    wait_fresh = self.REENTRY_FRESH_TREND
                                    continue
                            elif trailing_pending:
                                trailing_pending = False; trailing_pending_days = 0
                    if not exit_active and profit >= 24.0 and "volume" in data.columns:
                        volume = data["volume"].iloc[i]; volume_ma = data["volume_ma20"].iloc[i]; open_price = data["open"].iloc[i]
                        is_distribution = (not np.isnan(volume) and not np.isnan(volume_ma) and volume_ma > 0 and volume > volume_ma * 2.2 and price < open_price)
                        if is_distribution and i >= 12:
                            ma13 = np.mean(close_arr[i - 12 : i + 1]); deviation = (price - ma13) / ma13 * 100.0 if ma13 > 0 else 0.0
                            if deviation >= 24.0:
                                holding = False; exit_flags[i] = 1; exit_reasons[i] = f"放量阴线+偏离MA13({deviation:.1f}%)"
                                entry_price = None; trailing_active = False; max_profit = 0.0; pending_exit = False; pending_exit_days = 0; position[i] = 0
                                continue
                if pending_exit and not exit_active:
                    pending_exit = False; pending_exit_days = 0
                if pending_exit:
                    pending_exit_days += 1
                    if pending_exit_days >= 1:
                        holding = False; exit_flags[i] = 1; exit_reasons[i] = "趋势转空-延迟1日退出"; entry_price = None; pending_exit = False; pending_exit_days = 0
                        if self.REENTRY_S_K and s_arr[i] > self.REENTRY_S_K: arm_until = i + 60
                elif exit_active:
                    previous_close = close_arr[i - 1] if i > 0 else price
                    day_change = (price / previous_close - 1.0) * 100.0 if previous_close > 0 else 0.0
                    if i > 0 and day_change < -2.6:
                        pending_exit = True; pending_exit_days = 0
                    else:
                        holding = False; exit_flags[i] = 1; exit_reasons[i] = "趋势转空退出"; entry_price = None
                        if self.REENTRY_S_K and s_arr[i] > self.REENTRY_S_K: arm_until = i + 60
            position[i] = 1 if holding else 0
        return position, entry_flags, exit_flags, stop_flags, profit_flags, entry_reasons, exit_reasons

class Baseline(RSITrendStrategy):
    """The formal strategy, untouched. Used as the equivalence reference."""


def make_ma60(threshold):
    """Build a candidate with a different MA60-distance threshold."""

    class MA60(HookedStrategy):
        def chase_gates(self, data, close, direction):
            ma60_block = data["dist_ma60"] > threshold
            up_streak = self._consecutive_true(close > close.shift(1))
            streak_block = up_streak > 5
            return ma60_block | streak_block, ma60_block, ma60_block | streak_block

    MA60.__name__ = "MA60_%s" % threshold
    return MA60


REGISTRY = {
    "formal": Baseline,
    "baseline": HookedStrategy,
    "ma60_12": make_ma60(12.0),
    "ma60_18": make_ma60(18.0),
}


def load_registry():
    """REGISTRY plus anything defined in research/candidates_local.py, if present."""
    registry = dict(REGISTRY)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates_local.py")
    if os.path.exists(local):
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidates_local", local)
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidates_local"] = module
        spec.loader.exec_module(module)
        additions = getattr(module, "REGISTRY", {})
        reserved = {"formal", "baseline"} & set(additions)
        if reserved:
            raise ValueError("candidates_local.py cannot override reserved candidates: %s"
                             % ", ".join(sorted(reserved)))
        registry.update(additions)
    return registry
