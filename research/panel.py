#!/usr/bin/env python3
"""Cross-sectional market panel: the one dimension a single-stock backtest cannot see.

Builds, causally, for every (stock, date):
  r20            20-bar return
  cs20           percentile rank of r20 among stocks of the SAME market that day
  cs60           same for the 60-bar return
  above_ma50     close above its own 50-bar mean
  breadth50      share of that market above its own MA50 that day
  breadth_chg20  breadth50 minus its value 20 trading days earlier
  rs_ibd         IBD-style strength percentile: 0.4*r63 + 0.2*(r126 + r189 + r252), ranked per market
  F              cs20 >= 0.90 and breadth50 >= 0.50
  F2             F and breadth_chg20 > 0

Everything uses bars up to and including that day only. Written once, reused by every experiment.

    python3 research/panel.py                 # development pool -> research/out/panel_dev.csv
    python3 research/panel.py --pool oos      # revealed OOS230 -> research/out/panel_oos.csv
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


def load_closes(data_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv"))):
        code = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path, usecols=["date", "close"])
        df["code"] = code
        df["market"] = "HK" if len(code) == 5 else "CN-A"
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)


def build(data_dir):
    panel = load_closes(data_dir)
    grouped = panel.groupby("code", sort=False)["close"]
    for window in (20, 60, 63, 126, 189, 252):
        panel["r%d" % window] = grouped.transform(lambda s, w=window: s / s.shift(w) - 1.0)
    panel["ma50"] = grouped.transform(lambda s: s.rolling(50).mean())
    panel["above_ma50"] = (panel["close"] > panel["ma50"]).where(panel["ma50"].notna(), False)

    by_day = panel.groupby(["date", "market"], sort=False)
    panel["cs20"] = by_day["r20"].rank(pct=True)
    panel["cs60"] = by_day["r60"].rank(pct=True)
    panel["strength"] = (0.4 * panel["r63"] + 0.2 * panel["r126"]
                         + 0.2 * panel["r189"] + 0.2 * panel["r252"])
    panel["rs_ibd"] = by_day["strength"].rank(pct=True)

    breadth = by_day["above_ma50"].mean().rename("breadth50").reset_index()
    breadth = breadth.sort_values(["market", "date"], kind="mergesort")
    breadth["breadth_chg20"] = breadth.groupby("market", sort=False)["breadth50"].transform(
        lambda s: s - s.shift(20))
    panel = panel.merge(breadth, on=["date", "market"], how="left")

    panel["F"] = (panel["cs20"] >= 0.90) & (panel["breadth50"] >= 0.50)
    panel["F2"] = panel["F"] & (panel["breadth_chg20"] > 0)
    cols = ["code", "market", "date", "close", "r20", "r60", "cs20", "cs60", "rs_ibd",
            "above_ma50", "breadth50", "breadth_chg20", "F", "F2"]
    return panel[cols]


def panel_path(pool_name):
    return os.path.join(OUT, "panel_%s.csv" % pool_name)


def load(pool_name="dev"):
    path = panel_path(pool_name)
    if not os.path.exists(path):
        raise SystemExit("panel missing: run python3 research/panel.py --pool %s" % pool_name)
    panel = pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="dev")
    args = parser.parse_args()
    panel = build(resolve_pool(args.pool))
    os.makedirs(OUT, exist_ok=True)
    path = panel_path(args.pool)
    panel.to_csv(path, index=False)
    print("panel written: %s  rows=%d  stocks=%d  dates=%d"
          % (path, len(panel), panel.code.nunique(), panel.date.nunique()))
    for market, sub in panel.groupby("market"):
        valid = sub[sub.cs20.notna()]
        print("  %-5s stocks=%3d  F fires %.2f%%  F2 fires %.2f%%  mean breadth %.3f"
              % (market, sub.code.nunique(), valid.F.mean() * 100, valid.F2.mean() * 100,
                 sub.breadth50.mean()))


if __name__ == "__main__":
    main()
