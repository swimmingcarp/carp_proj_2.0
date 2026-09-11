#!/usr/bin/env python3
"""A stock's PHASE, measured on a rolling recent window - not a permanent label.

The same stock is choppy for months and then trends for months. Classifying it once from three years of
history answers the wrong question and cannot be traded: by the time the label is earned it is stale.
This module asks, every day, "what phase is this stock in RIGHT NOW", using only the recent past.

For each stock and day, from a trailing window (default 120 bars, about six months):
  eff        Kaufman efficiency ratio: |net move| / total travel. High = travelling, low = churning.
  vr20       variance ratio at q=20. Below 1 = mean-reverting, above 1 = trending.
  hurst      rescaled-range exponent on returns.
  adx        Wilder's ADX(14), the practitioner's trend-strength standard.
  cross      moving-average crossings per 100 bars: a direct count of chop.
  atr_pct    ATR20 / close: the volatility level.
  range_hold share of bars inside the window's own middle half - a range-bound stock stays inside it.
  bb_width   Bollinger bandwidth (upper-lower)/middle: narrow = coiled/ranging, wide = expanding
  bb_pctb    %B, where the close sits inside the bands
  ene_pos    position inside the ENE envelope ((1+M/100)*MA25, (1-M/100)*MA25), the CN retail standard
  ma_spread  dispersion of MA5/10/20/60 divided by price - the "moving averages entangled" measure.
             Small = averages glued together = no trend / pre-breakout coil. Large = fanned out = trend.
  ma_order   how close MA5>MA10>MA20>MA60 is to a perfect bullish stack, scored 0..1
  ma_slope   slope of MA20 over 20 bars, in percent - trend steepness
  sr_pos     position between the last CONFIRMED swing low and swing high (support and resistance)
  sr_touch   how many times price has revisited those levels - a well-tested range is a real range
  sr_width   the range width divided by price - how much room there is to trade inside it

All are then converted to that stock's OWN trailing percentile, so thresholds are scale-free and adapt
to each stock instead of one global cut point.

CRITICAL: nothing here touches any strategy's signals, so the phase label cannot be contaminated by the
entry logic being evaluated - the error made on 2026-09-08.

    python3 research/phase_state.py --pool dev [--fast]
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
from harness import OUT, resolve_pool, fast_files  # noqa: E402

WIN = 120
PCT_WIN = 250


def _vr(ret, window, q=20):
    n = len(ret)
    out = np.full(n, np.nan)
    r = ret.to_numpy(float)
    for i in range(window, n):
        seg = r[i - window + 1:i + 1]
        seg = seg[~np.isnan(seg)]
        m = len(seg)
        if m < window * 0.8:
            continue
        mu = seg.mean()
        v1 = ((seg - mu) ** 2).sum() / (m - 1)
        if v1 <= 0:
            continue
        cs = np.cumsum(seg)
        qr = cs[q - 1:] - np.concatenate(([0.0], cs[:-q]))
        k = len(qr)
        vq = ((qr - q * mu) ** 2).sum() / (k * q) * (m / (m - q + 1))
        out[i] = vq / v1
    return out


def _hurst(ret, window):
    n = len(ret)
    out = np.full(n, np.nan)
    r = ret.to_numpy(float)
    sizes = [10, 20, 40]
    for i in range(window, n):
        seg = r[i - window + 1:i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) < window * 0.8:
            continue
        pts = []
        for s in sizes:
            vals = []
            for j in range(len(seg) // s):
                c = seg[j * s:(j + 1) * s]
                z = c - c.mean()
                sd = c.std(ddof=1)
                if sd > 0:
                    vals.append(np.ptp(z.cumsum()) / sd)
            if vals:
                pts.append((np.log(s), np.log(np.mean(vals))))
        if len(pts) >= 3:
            a = np.array(pts)
            out[i] = np.polyfit(a[:, 0], a[:, 1], 1)[0]
    return out


def _adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _swing_levels(df, k=5, lookback=WIN):
    """Support and resistance from CONFIRMED swing points only.

    A swing high at bar i can only be recognised once k later bars exist, so the level is published at
    i+k, never at i. Reading it earlier would be a lookahead - the classic trap in support/resistance
    code. Returns (support, resistance, touches) as arrays aligned to the bars.
    """
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    sup = np.full(n, np.nan)
    res = np.full(n, np.nan)
    touch = np.zeros(n)

    confirmed_high = []   # (bar_index_available_from, level)
    confirmed_low = []
    for i in range(k, n - k):
        if high[i] == max(high[i - k:i + k + 1]):
            confirmed_high.append((i + k, high[i]))
        if low[i] == min(low[i - k:i + k + 1]):
            confirmed_low.append((i + k, low[i]))

    hi_idx = 0
    lo_idx = 0
    hi_live = []
    lo_live = []
    for t in range(n):
        while hi_idx < len(confirmed_high) and confirmed_high[hi_idx][0] <= t:
            hi_live.append(confirmed_high[hi_idx]); hi_idx += 1
        while lo_idx < len(confirmed_low) and confirmed_low[lo_idx][0] <= t:
            lo_live.append(confirmed_low[lo_idx]); lo_idx += 1
        hs = [lv for av, lv in hi_live if av > t - lookback]
        ls = [lv for av, lv in lo_live if av > t - lookback]
        if hs:
            res[t] = max(hs)
        if ls:
            sup[t] = min(ls)
        if hs and ls and res[t] > sup[t]:
            band = (res[t] - sup[t]) * 0.10
            lo_b, hi_b = sup[t], res[t]
            seg = close[max(0, t - lookback):t + 1]
            touch[t] = int(np.sum((np.abs(seg - lo_b) < band) | (np.abs(seg - hi_b) < band)))
    return sup, res, touch


def build_stock(df, win=WIN, pct_win=PCT_WIN):
    close = df["close"]
    ret = np.log(close).diff()
    out = pd.DataFrame(index=df.index)

    move = (close - close.shift(win)).abs()
    travel = close.diff().abs().rolling(win).sum()
    out["eff"] = move / travel.replace(0, np.nan)
    out["vr20"] = _vr(ret, win)
    out["hurst"] = _hurst(ret, win)
    out["adx"] = _adx(df)
    ma = close.rolling(20).mean()
    above = (close > ma).astype(float)
    out["cross"] = above.diff().abs().rolling(win).sum() / win * 100
    tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift()).abs(),
                    (df["low"] - close.shift()).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.rolling(20).mean() / close * 100
    hi = close.rolling(win).max()
    lo = close.rolling(win).min()
    mid_hi = lo + (hi - lo) * 0.75
    mid_lo = lo + (hi - lo) * 0.25
    out["range_hold"] = ((close > mid_lo) & (close < mid_hi)).rolling(win).mean()

    # Bollinger: width says coiled vs expanding, %B says where in the band we are
    bb_mid = close.rolling(20).mean()
    bb_sd = close.rolling(20).std()
    upper, lower = bb_mid + 2 * bb_sd, bb_mid - 2 * bb_sd
    out["bb_width"] = (upper - lower) / bb_mid
    out["bb_pctb"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # ENE envelope, the parameters used on Chinese retail platforms (N=25, M=6)
    ene_ma = close.rolling(25).mean()
    ene_up, ene_dn = ene_ma * 1.06, ene_ma * 0.94
    out["ene_pos"] = (close - ene_dn) / (ene_up - ene_dn).replace(0, np.nan)

    # moving-average entanglement: glued together = no trend, fanned out = trend
    mas = pd.concat([close.rolling(w).mean() for w in (5, 10, 20, 60)], axis=1)
    out["ma_spread"] = mas.std(axis=1) / close * 100
    m5, m10, m20, m60 = (mas.iloc[:, i] for i in range(4))
    out["ma_order"] = ((m5 > m10).astype(float) + (m10 > m20).astype(float)
                       + (m20 > m60).astype(float)) / 3.0
    out["ma_slope"] = (m20 / m20.shift(20) - 1) * 100

    sup, res, touch = _swing_levels(df)
    width = pd.Series(res - sup, index=df.index)
    out["sr_pos"] = (close - pd.Series(sup, index=df.index)) / width.replace(0, np.nan)
    out["sr_touch"] = touch
    out["sr_width"] = width / close * 100

    pct = out.rolling(pct_win, min_periods=pct_win // 2).rank(pct=True)
    pct.columns = [c + "_p" for c in out.columns]
    return pd.concat([out, pct], axis=1)


def build(files):
    frames = []
    for path in files:
        code = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
        if len(df) < WIN + PCT_WIN // 2 + 50:
            continue
        f = build_stock(df)
        f["code"] = code
        f["market"] = "HK" if len(code) == 5 else "CN-A"
        f["date"] = pd.to_datetime(df["date"])
        f["close"] = df["close"].values
        for h in (20, 40, 60):
            f["fwd%d" % h] = (df["close"].shift(-h) / df["close"] - 1).values * 100
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="dev")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    files = fast_files() if args.fast else sorted(
        glob.glob(os.path.join(resolve_pool(args.pool), "*_hfq.csv")))
    panel = build(files)
    os.makedirs(OUT, exist_ok=True)
    tag = "fast" if args.fast else args.pool
    p = os.path.join(OUT, "phase_%s.csv" % tag)
    panel.to_csv(p, index=False)
    print("written %s   rows %d   stocks %d" % (p, len(panel), panel.code.nunique()))

    raw = ["eff", "vr20", "hurst", "adx", "cross", "atr_pct", "range_hold",
           "bb_width", "bb_pctb", "ene_pos", "ma_spread", "ma_order", "ma_slope",
           "sr_pos", "sr_touch", "sr_width"]
    print("\nDo these phase measures agree with each other? (Spearman on their own-history percentiles)")
    corr = panel[[c + "_p" for c in raw]].corr(method="spearman")
    print(corr.round(2).to_string())

    print("\nHow persistent is a phase? (median run length of the top/bottom tercile of each measure)")
    for c in raw:
        col = panel[c + "_p"]
        runs = []
        for _, g in panel.groupby("code", sort=False):
            v = g[c + "_p"]
            state = pd.cut(v, [0, 1 / 3, 2 / 3, 1.0], labels=[0, 1, 2])
            s = state.dropna()
            if len(s) < 50:
                continue
            grp = (s != s.shift()).cumsum()
            runs.extend(s.groupby(grp).size().tolist())
        if runs:
            print("  %-11s median run %3.0f bars   mean %5.1f" % (c, np.median(runs), np.mean(runs)))


if __name__ == "__main__":
    main()
