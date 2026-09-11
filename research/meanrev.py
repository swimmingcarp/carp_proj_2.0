#!/usr/bin/env python3
"""A standalone mean-reversion strategy, to be compared against the trend strategy state by state.

This is NOT a bolt-on to the existing system. It is an independent buy-low-sell-high rule, so that the
question "which regime suits which strategy" can actually be asked. Deliberately crude - three coarse
parameters, all classic values, none tuned here:

  entry  close below its 20-bar mean by more than k x ATR20/close, while the 120-bar trend is not broken
         (close above its own 120-bar mean), i.e. a dip inside a longer uptrend, not a collapse
  exit   close back above the 20-bar mean (target reached), or a hard stop, or a time stop

It shares the project's execution model exactly: decisions and fills at the same close, T+1, real fees,
via the same StrategyBase.backtest the formal reports use.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from hooked import APP  # noqa: E402
sys.path.insert(0, os.path.join(APP, "src"))
from indicators import atr_indicator  # noqa: E402
from new_strategy import RSITrendStrategy  # noqa: E402

# _prepare_dataframe (column normalisation, date parsing, sorting) lives on RSITrendStrategy, so both
# strategies inherit it and then replace analyze() entirely. Nothing of the trend logic survives.


class MeanReversion(RSITrendStrategy):
    """Buy dips below a short mean inside an intact long trend; sell on reversion to that mean."""

    DIP_K = 1.5          # how far below MA20, in ATR20 units
    STOP_PCT = 8.0       # same order as the trend strategy's hard stop
    TIME_STOP = 20       # abandon a dip that does not revert
    TREND_FILTER = True  # require close above MA120

    def analyze(self, df):
        if df is None or len(df) == 0:
            return None, None
        data = self._prepare_dataframe(df)
        close = data["close"]
        ma20 = close.rolling(20).mean()
        ma120 = close.rolling(120).mean()
        atr = atr_indicator(data, period=20)

        below = (ma20 - close) / atr.replace(0, np.nan)
        intact = (close > ma120) if self.TREND_FILTER else pd.Series(True, index=data.index)
        entry = (below >= self.DIP_K) & intact.fillna(False)
        target = close >= ma20

        data["ma20"] = ma20
        data["ma120"] = ma120
        data["dip_atr"] = below

        n = len(data)
        px = close.to_numpy(float)
        e = entry.fillna(False).to_numpy(bool)
        t = target.fillna(False).to_numpy(bool)
        pos = np.zeros(n, int)
        ef = np.zeros(n, int); xf = np.zeros(n, int); sf = np.zeros(n, int); pf = np.zeros(n, int)
        er = [""] * n; xr = [""] * n

        holding = False
        entry_px = 0.0
        held = 0
        for i in range(n):
            if not holding:
                if e[i]:
                    holding = True; entry_px = px[i]; held = 0
                    ef[i] = 1; er[i] = "回归买入"
            else:
                held += 1
                if px[i] <= entry_px * (1 - self.STOP_PCT / 100.0):
                    holding = False; xf[i] = 1; sf[i] = 1; xr[i] = "硬性止损"
                elif t[i]:
                    holding = False; xf[i] = 1; pf[i] = 1; xr[i] = "回归目标"
                elif held >= self.TIME_STOP:
                    holding = False; xf[i] = 1; xr[i] = "时间退出"
            pos[i] = 1 if holding else 0

        data["buy_signal"] = pos
        data["entry_signal"] = ef; data["exit_signal"] = xf
        data["stop_loss_exit"] = sf; data["profit_target_exit"] = pf
        data["entry_reason"] = er; data["exit_reason"] = xr
        return data, None


class BuyHold(RSITrendStrategy):
    """Always in. The benchmark a long-only regime switch must beat when it says 'trend'."""

    def analyze(self, df):
        if df is None or len(df) == 0:
            return None, None
        data = self._prepare_dataframe(df)
        n = len(data)
        data["buy_signal"] = np.ones(n, int)
        data["entry_signal"] = np.array([1] + [0] * (n - 1))
        data["exit_signal"] = np.zeros(n, int)
        data["stop_loss_exit"] = np.zeros(n, int)
        data["profit_target_exit"] = np.zeros(n, int)
        data["entry_reason"] = ["买入持有"] + [""] * (n - 1)
        data["exit_reason"] = [""] * n
        return data, None


def make_meanrev(dip_k=1.5, stop=8.0, time_stop=20, trend_filter=True):
    class MR(MeanReversion):
        DIP_K = dip_k
        STOP_PCT = stop
        TIME_STOP = time_stop
        TREND_FILTER = trend_filter
    MR.__name__ = "MeanRev_k%.1f_t%d%s" % (dip_k, time_stop, "" if trend_filter else "_nofilter")
    return MR


REGISTRY = {
    "meanrev": make_meanrev(),
    "meanrev_deep": make_meanrev(dip_k=2.5),
    "meanrev_nofilter": make_meanrev(trend_filter=False),
    "buyhold": BuyHold,
}
