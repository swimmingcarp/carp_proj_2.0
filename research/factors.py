#!/usr/bin/env python3
"""Price-volume factor library. Every factor is causal and has a stated economic reading.

We only have OHLCV, so this is deliberately a price-volume library - no fundamentals, no analyst data.
Factors are grouped by the economic story they tell, because grouping is what later decides which ones
may be combined and which are near-duplicates of each other.

Sign convention: every factor is oriented so that HIGHER = expected BETTER forward return, according to
the literature it comes from. Where the literature says the anomaly is negative (e.g. the MAX effect,
short-horizon reversal in A-shares), the factor is negated here. That orientation is a HYPOTHESIS, and
the evaluation is free to reject it.

    python3 research/factors.py --pool dev     # -> research/out/factors_dev.csv
    python3 research/factors.py --pool oos
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from harness import OUT, resolve_pool  # noqa: E402

# name -> (group, one-line economic reading)
FACTOR_DOC = {
    "mom_12_1":   ("momentum", "12-month return skipping the last month (Jegadeesh-Titman); weak in A-shares"),
    "mom_6_1":    ("momentum", "6-month return skipping the last month"),
    "mom_3_1":    ("momentum", "3-month return skipping the last month"),
    "rev_1m":     ("reversal", "NEGATED 1-month return; short-horizon reversal, strong in A-shares"),
    "rev_1w":     ("reversal", "NEGATED 1-week return; very short-horizon reversal"),
    "max5":       ("lottery", "NEGATED mean of the 5 largest daily returns in 20 days (Bali MAX effect)"),
    "skew60":     ("lottery", "NEGATED 60-day return skewness; lottery-like payoffs are overpriced"),
    "ivol60":     ("volatility", "NEGATED 60-day return volatility; the low-volatility anomaly"),
    "ivol_rel":   ("volatility", "NEGATED volatility relative to the stock's own 250-day history"),
    "downside_vol": ("volatility", "NEGATED annualised 120-day downside deviation; lower downside risk is preferred"),
    "turn_low":   ("liquidity", "NEGATED 20-day mean turnover proxy volume/volume_250; low turnover outperforms in A-shares"),
    "turn_trend": ("liquidity", "20-day mean volume over 60-day mean volume; participation building"),
    "trend_ma":   ("trend", "distance of close above its own 60-day mean"),
    "trend_slope": ("trend", "annualised slope of a 60-bar log-price regression"),
    "trend_r2":   ("trend", "R-squared of that regression; how orderly the path is"),
    "trend_qual": ("trend", "slope x R-squared (Clenow); orderly trends only"),
    "near_high":  ("anchor", "close divided by its own 250-day high (George-Hwang nearness)"),
    "dd_250":     ("anchor", "drawdown from the 250-day high; the mirror of nearness"),
    "path_eff":   ("quality", "Kaufman efficiency ratio over 20 bars; net move over total travel"),
    "up_ratio":   ("quality", "share of up days over 60 bars"),
    "vol_price_corr": ("quality", "20-day correlation of volume with absolute return; NEGATED, churn is bad"),
    "gap_ratio":  ("quality", "NEGATED share of bars with a gap over 2%; jumpy names are harder to hold"),
    "range_pos":  ("position", "position of the close inside the 120-day high-low range"),
    "ma_stack":   ("trend", "how many of MA5>MA20>MA60>MA120 hold, as a 0-1 score"),
}


def _reg_slope_r2(logp, window):
    """Rolling OLS of log price on time. Returns (annualised slope, r-squared), both causal."""
    n = len(logp)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x -= x.mean()
    sxx = (x * x).sum()
    values = logp.to_numpy(float)
    for i in range(window - 1, n):
        seg = values[i - window + 1:i + 1]
        if np.isnan(seg).any():
            continue
        sm = seg.mean()
        sxy = (x * (seg - sm)).sum()
        syy = ((seg - sm) ** 2).sum()
        b = sxy / sxx
        slope[i] = (np.exp(b * 252) - 1) * 100
        r2[i] = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return slope, r2


def build_stock(df):
    """All factors for one stock. df must be sorted by date and contain OHLCV."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    ret = close.pct_change()
    logp = np.log(close)
    out = pd.DataFrame(index=df.index)

    out["mom_12_1"] = (close.shift(21) / close.shift(252) - 1) * 100
    out["mom_6_1"] = (close.shift(21) / close.shift(126) - 1) * 100
    out["mom_3_1"] = (close.shift(21) / close.shift(63) - 1) * 100
    out["rev_1m"] = -(close / close.shift(21) - 1) * 100
    out["rev_1w"] = -(close / close.shift(5) - 1) * 100

    out["max5"] = -ret.rolling(20).apply(lambda s: np.sort(s)[-5:].mean(), raw=True) * 100
    out["skew60"] = -ret.rolling(60).skew()

    vol60 = ret.rolling(60).std()
    out["ivol60"] = -vol60 * np.sqrt(252) * 100
    out["ivol_rel"] = -(vol60 / vol60.rolling(250, min_periods=120).mean())
    downside_squared = ret.clip(upper=0).pow(2)
    out["downside_vol"] = -np.sqrt(
        downside_squared.rolling(120, min_periods=60).mean()
    ) * np.sqrt(252) * 100

    vmean250 = volume.rolling(250, min_periods=120).mean()
    out["turn_low"] = -(volume.rolling(20).mean() / vmean250)
    out["turn_trend"] = volume.rolling(20).mean() / volume.rolling(60).mean()

    ma60 = close.rolling(60).mean()
    out["trend_ma"] = (close / ma60 - 1) * 100
    slope, r2 = _reg_slope_r2(logp, 60)
    out["trend_slope"] = slope
    out["trend_r2"] = r2
    out["trend_qual"] = slope * r2

    high250 = close.rolling(250, min_periods=250).max()
    out["near_high"] = close / high250
    out["dd_250"] = (close / high250 - 1) * 100

    move = (close - close.shift(20)).abs()
    travel = close.diff().abs().rolling(20).sum()
    out["path_eff"] = move / travel.replace(0, np.nan)
    out["up_ratio"] = (ret > 0).rolling(60).mean()
    out["vol_price_corr"] = -ret.abs().rolling(20).corr(volume)
    gap = (df["open"] / close.shift(1) - 1).abs()
    out["gap_ratio"] = -(gap > 0.02).rolling(60).mean()

    hi120 = high.rolling(120).max()
    lo120 = low.rolling(120).min()
    out["range_pos"] = (close - lo120) / (hi120 - lo120).replace(0, np.nan)

    ma5, ma20, ma120 = (close.rolling(w).mean() for w in (5, 20, 120))
    out["ma_stack"] = ((ma5 > ma20).astype(float) + (ma20 > ma60).astype(float)
                       + (ma60 > ma120).astype(float)) / 3.0
    return out


def build(data_dir, horizons=(20, 40, 60)):
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv"))):
        code = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
        if len(df) < 300:
            continue
        fac = build_stock(df)
        fac["code"] = code
        fac["market"] = "HK" if len(code) == 5 else "CN-A"
        fac["date"] = pd.to_datetime(df["date"])
        fac["close"] = df["close"].values
        for h in horizons:
            fac["fwd%d" % h] = (df["close"].shift(-h) / df["close"] - 1).values * 100
            fac["fwd%d_date" % h] = pd.to_datetime(df["date"]).shift(-h).values
        frames.append(fac)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


def factor_names():
    return list(FACTOR_DOC)


def path(pool):
    return os.path.join(OUT, "factors_%s.csv" % pool)


def load(pool="dev"):
    p = path(pool)
    if not os.path.exists(p):
        raise SystemExit("missing: python3 research/factors.py --pool %s" % pool)
    panel = pd.read_csv(p, dtype={"code": str}, parse_dates=["date"])
    for column in ("fwd20_date", "fwd40_date", "fwd60_date"):
        if column in panel:
            panel[column] = pd.to_datetime(panel[column])
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="dev")
    args = parser.parse_args()
    panel = build(resolve_pool(args.pool))
    os.makedirs(OUT, exist_ok=True)
    panel.to_csv(path(args.pool), index=False)
    print("factors written: %s" % path(args.pool))
    print("  rows %d  stocks %d  dates %d  factors %d"
          % (len(panel), panel.code.nunique(), panel.date.nunique(), len(FACTOR_DOC)))
    cover = panel[factor_names()].notna().mean().sort_values()
    print("  lowest coverage:", ", ".join("%s %.0f%%" % (k, v * 100) for k, v in cover.head(4).items()))


if __name__ == "__main__":
    main()
