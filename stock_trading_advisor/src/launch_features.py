#!/usr/bin/env python3
"""Causal launch-detection features.

Every column is computable at the CLOSE of bar t from that stock's own OHLCV history alone - no
future data, no cross-sectional information, no strategy signal. Shared by the offline detection
study (/tmp/run_detect_launch) and by the tradeable candidates in candidates_launch.py, so the
backtested rule and the measured detector are the same function.
"""
import numpy as np
import pandas as pd

PCT = 250
SWING_K = 5
SWING_LB = 120


def _rank(s, win=PCT, minp=120):
    return s.rolling(win, min_periods=minp).rank(pct=True)


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
    return dx.ewm(alpha=1 / period, adjust=False).mean(), tr


def _swing(df, k=SWING_K, lookback=SWING_LB):
    """Confirmed swing support/resistance; a swing at bar i is published only at i+k (causal)."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    sup = np.full(n, np.nan); res = np.full(n, np.nan); touch = np.zeros(n)
    ch = []; cl = []
    for i in range(k, n - k):
        if high[i] == max(high[i - k:i + k + 1]):
            ch.append((i + k, high[i]))
        if low[i] == min(low[i - k:i + k + 1]):
            cl.append((i + k, low[i]))
    hi_idx = lo_idx = 0
    hi_live = []; lo_live = []
    for t in range(n):
        while hi_idx < len(ch) and ch[hi_idx][0] <= t:
            hi_live.append(ch[hi_idx]); hi_idx += 1
        while lo_idx < len(cl) and cl[lo_idx][0] <= t:
            lo_live.append(cl[lo_idx]); lo_idx += 1
        hs = [lv for av, lv in hi_live if av > t - lookback]
        ls = [lv for av, lv in lo_live if av > t - lookback]
        if hs:
            res[t] = max(hs)
        if ls:
            sup[t] = min(ls)
        if hs and ls and res[t] > sup[t]:
            band = (res[t] - sup[t]) * 0.10
            seg = close[max(0, t - lookback):t + 1]
            touch[t] = int(np.sum((np.abs(seg - sup[t]) < band) | (np.abs(seg - res[t]) < band)))
    return sup, res, touch


def _streak(mask):
    out = np.zeros(len(mask)); run = 0
    for i, v in enumerate(mask.to_numpy()):
        run = run + 1 if v else 0
        out[i] = run
    return out


def build_features(df):
    """OHLCV DataFrame (date, open, close, high, low, volume) -> causal feature frame."""
    c = df["close"].astype(float); h = df["high"].astype(float)
    lo = df["low"].astype(float); o = df["open"].astype(float)
    v = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=c.index)
    ret = c.pct_change()
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    mas = {w: c.rolling(w).mean() for w in (5, 10, 20, 60, 120)}
    for w in (5, 20, 60, 120):
        f["ma%d_slope20" % w] = (mas[w] / mas[w].shift(20) - 1) * 100
    f["ma20_slope5"] = (mas[20] / mas[20].shift(5) - 1) * 100
    f["ma_stack"] = ((c > mas[5]).astype(float) + (mas[5] > mas[10]).astype(float)
                     + (mas[10] > mas[20]).astype(float) + (mas[20] > mas[60]).astype(float)
                     + (mas[60] > mas[120]).astype(float)) / 5.0
    ms = pd.concat([mas[w] for w in (5, 10, 20, 60)], axis=1)
    f["ma_spread"] = ms.std(axis=1) / c * 100
    f["ma_spread_pct"] = _rank(f["ma_spread"])
    for w in (20, 60, 120):
        f["dist_ma%d" % w] = (c / mas[w] - 1) * 100
    for w in (60, 120, 250):
        mx = c.rolling(w, min_periods=int(w * 0.6)).max()
        mn = c.rolling(w, min_periods=int(w * 0.6)).min()
        f["pos_%d" % w] = (c / mx - 1) * 100
        f["rngpos_%d" % w] = (c - mn) / (mx - mn).replace(0, np.nan)
        f["up_from_low_%d" % w] = (c / mn - 1) * 100
    adx, tr = _adx(df)
    f["adx"] = adx
    f["adx_pct"] = _rank(adx)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct_price"] = atr / c * 100
    f["atr_pctile"] = _rank(f["atr_pct_price"])
    f["atr_ratio60"] = atr / atr.rolling(60, min_periods=30).mean()
    rv = lr.rolling(20).std() * np.sqrt(252) * 100
    f["rvol20"] = rv
    f["rvol20_pctile"] = _rank(rv)
    f["rvol_ratio"] = rv / (lr.rolling(120, min_periods=60).std() * np.sqrt(252) * 100)
    bmid = mas[20]; bsd = c.rolling(20).std()
    bw = (4 * bsd) / bmid
    f["bb_width"] = bw
    f["bb_width_pctile"] = _rank(bw)
    f["bb_pctb"] = (c - (bmid - 2 * bsd)) / (4 * bsd).replace(0, np.nan)
    prior_sq = f["bb_width_pctile"].shift(1).rolling(10, min_periods=5).min() < 0.10
    f["squeeze_release"] = (prior_sq & (bw > bw.shift(5))).astype(float)
    vma20 = v.rolling(20).mean(); vma60 = v.rolling(60, min_periods=30).mean()
    f["vol_ratio20"] = v / vma20.replace(0, np.nan)
    f["vol_ratio60"] = v / vma60.replace(0, np.nan)
    f["vol_pctile"] = _rank(v)
    f["vol_trend"] = v.rolling(5).mean() / vma60.replace(0, np.nan)
    turn = v * c
    f["turn_ratio20"] = turn / turn.rolling(20).mean().replace(0, np.nan)
    f["turn_pctile"] = _rank(turn)
    dry = (v / vma20.replace(0, np.nan)).shift(3).rolling(8, min_periods=4).min() < 0.75
    f["vol_dryup_surge"] = (dry & (f["vol_ratio20"] > 1.8)).astype(float)
    f["up_streak"] = _streak(c > c.shift(1))
    f["up_days_20"] = (c > c.shift(1)).rolling(20).sum()
    f["big_up_20"] = (ret >= 0.05).rolling(20).sum()
    f["limitup_20"] = (ret >= 0.095).rolling(20).sum()
    f["limitup20_20"] = (ret >= 0.19).rolling(20).sum()
    f["gapup_20"] = ((o / c.shift(1) - 1) > 0.02).rolling(20).sum()
    rngp = (h - lo) / c
    f["range_exp"] = rngp / rngp.rolling(20).mean().replace(0, np.nan)
    f["range_exp_pctile"] = _rank(rngp)
    for w in (5, 10, 20, 60):
        f["ret_%d" % w] = (c / c.shift(w) - 1) * 100
    f["runup_20"] = (c / c.rolling(20).min() - 1) * 100
    for w in (20, 60):
        move = (c - c.shift(w)).abs()
        travel = c.diff().abs().rolling(w).sum()
        f["eff_%d" % w] = move / travel.replace(0, np.nan)
    sup, res, touch = _swing(df)
    sup = pd.Series(sup, index=c.index); res = pd.Series(res, index=c.index)
    f["base_dist_sup"] = (c / sup - 1) * 100
    f["base_dist_res"] = (c / res - 1) * 100
    f["base_touch"] = touch
    f["base_width"] = (res - sup) / c * 100
    f["base_breakout"] = (c > res).astype(float)
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l_ = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + g / l_.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c * 100
    return f


def score(df, model):
    """Frozen logistic score in [0,1] per bar. `model` = dict(cols, mu, sd, coef, intercept)."""
    f = build_features(df)
    cols = model["cols"]
    mu = pd.Series(model["mu"])[cols]
    sd = pd.Series(model["sd"])[cols]
    co = pd.Series(model["coef"])[cols].to_numpy(float)
    z = ((f[cols] - mu) / sd).clip(-5, 5)
    ok = z.notna().all(axis=1)
    lin = np.full(len(f), np.nan)
    lin[ok.to_numpy()] = z.loc[ok].to_numpy(float) @ co + model["intercept"]
    return pd.Series(1.0 / (1.0 + np.exp(-lin)), index=f.index)
