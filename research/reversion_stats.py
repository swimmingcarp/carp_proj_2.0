#!/usr/bin/env python3
"""Is this stock statistically mean-reverting right now, and how fast?

Three classical measures, all rolling and causal, computed per stock per day:

  VR(q)      Lo-MacKinlay variance ratio. Var(q-period return)/(q*Var(1-period return)).
             VR < 1 => mean reversion, VR = 1 => random walk, VR > 1 => trending.
             Reported with the heteroskedasticity-robust z statistic.
  AR1        first-order autocorrelation of daily returns. Negative => reversion.
  half_life  from an Ornstein-Uhlenbeck fit of log price to its own moving level:
             dX = theta*(mu - X)dt + sigma dW, discretised as dX_t = a + b*X_{t-1} + e,
             half_life = -ln(2)/ln(1+b). This is the horizon a mean-reversion trade should target,
             and it is the natural adaptive parameter: it is estimated from the stock's own history.

Everything uses a trailing window only. No stock is looked at as a whole.

    python3 research/reversion_stats.py --pool dev
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

WINDOW = 250


def _rolling_vr(ret, window, q):
    """Variance ratio VR(q) and its heteroskedasticity-robust z, rolling."""
    n = len(ret)
    vr = np.full(n, np.nan)
    z = np.full(n, np.nan)
    r = ret.to_numpy(float)
    for i in range(window, n):
        seg = r[i - window + 1:i + 1]
        seg = seg[~np.isnan(seg)]
        m = len(seg)
        if m < window * 0.8:
            continue
        mu = seg.mean()
        var1 = ((seg - mu) ** 2).sum() / (m - 1)
        if var1 <= 0:
            continue
        # overlapping q-period returns
        cs = np.cumsum(seg)
        qret = cs[q - 1:] - np.concatenate(([0.0], cs[:-q]))
        k = len(qret)
        varq = ((qret - q * mu) ** 2).sum() / (k * q) * (m / (m - q + 1))
        ratio = varq / var1
        vr[i] = ratio
        # robust standard error (Lo-MacKinlay 1988, heteroskedasticity-consistent)
        d = seg - mu
        theta = 0.0
        for j in range(1, q):
            num = (d[j:] ** 2 * d[:-j] ** 2).sum()
            den = (d ** 2).sum() ** 2 / m
            delta = num / den if den > 0 else 0.0
            theta += (2.0 * (q - j) / q) ** 2 * delta
        z[i] = (ratio - 1.0) / np.sqrt(theta / m) if theta > 0 else np.nan
    return vr, z


def _rolling_half_life(logp, window):
    """OU half-life from a rolling AR(1) of the log price on its own lag."""
    n = len(logp)
    hl = np.full(n, np.nan)
    x = logp.to_numpy(float)
    for i in range(window, n):
        seg = x[i - window + 1:i + 1]
        if np.isnan(seg).any():
            continue
        lag = seg[:-1]
        d = np.diff(seg)
        lm = lag.mean()
        denom = ((lag - lm) ** 2).sum()
        if denom <= 0:
            continue
        b = ((lag - lm) * (d - d.mean())).sum() / denom
        if -1 < b < 0:
            hl[i] = -np.log(2) / np.log(1 + b)
    return hl


def build(data_dir, window=WINDOW):
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv"))):
        code = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
        if len(df) < window + 60:
            continue
        close = df["close"]
        ret = np.log(close).diff()
        out = pd.DataFrame({"code": code, "date": pd.to_datetime(df["date"])})
        out["market"] = "HK" if len(code) == 5 else "CN-A"
        for q in (5, 10, 20):
            vr, z = _rolling_vr(ret, window, q)
            out["vr%d" % q] = vr
            out["vrz%d" % q] = z
        out["ar1"] = ret.rolling(window).apply(lambda s: pd.Series(s).autocorr(1), raw=False)
        out["half_life"] = _rolling_half_life(np.log(close), window)
        for h in (10, 20, 40):
            out["fwd%d" % h] = (close.shift(-h) / close - 1).values * 100
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="dev")
    args = parser.parse_args()
    panel = build(resolve_pool(args.pool))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "reversion_%s.csv" % args.pool)
    panel.to_csv(p, index=False)
    print("written: %s   rows %d   stocks %d" % (p, len(panel), panel.code.nunique()))

    v = panel.dropna(subset=["vr20"])
    print("\nAre these stocks mean-reverting? Variance ratio VR(q): <1 reverting, >1 trending")
    for q in (5, 10, 20):
        col = "vr%d" % q
        sub = panel[col].dropna()
        zcol = panel["vrz%d" % q].dropna()
        print("  VR(%2d): median %.3f  share below 1: %5.1f%%   "
              "significantly reverting (z<-1.96): %4.1f%%   significantly trending (z>1.96): %4.1f%%"
              % (q, sub.median(), (sub < 1).mean() * 100,
                 (zcol < -1.96).mean() * 100, (zcol > 1.96).mean() * 100))
    print("\n  AR(1) of daily returns: median %.4f, share negative %.1f%%"
          % (panel.ar1.median(), (panel.ar1 < 0).mean() * 100))
    hl = panel.half_life.dropna()
    print("  OU half-life (bars), where an AR(1) fit implies reversion at all: "
          "defined on %.1f%% of days, median %.1f, quartiles %.1f / %.1f"
          % (len(hl) / len(panel) * 100, hl.median(), hl.quantile(.25), hl.quantile(.75)))
    print("\n  by market:")
    for mk, g in panel.groupby("market"):
        print("    %-5s VR(20) median %.3f  share<1 %5.1f%%   AR1 median %+.4f  half-life median %.1f"
              % (mk, g.vr20.median(), (g.vr20 < 1).mean() * 100, g.ar1.median(), g.half_life.median()))


if __name__ == "__main__":
    main()
