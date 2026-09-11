#!/usr/bin/env python3
"""Strategy-character diagnosis: what kind of system is this, and where does its return come from?

Answers the questions AGENTS.md now requires before any improvement is proposed:
  - versus buy-and-hold per stock: median excess, mean excess, share of stocks beaten
  - excess by buy-and-hold quintile: does it protect weak stocks and truncate strong ones?
  - return attribution: which layer of stocks contributes the summed return
  - capture ratio: how much of a rising stock's move is kept
  - drawdown protection by layer
  - trade-level concentration: how much of the profit sits in the right tail

    python3 research/diagnose.py ma60_12
    python3 research/diagnose.py ma60_12 --data-dir /path/to/out_of_pool
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hooked import DATA_DIR, load_registry  # noqa: E402
from harness import artifact_path, evaluate, resolve_pool  # noqa: E402


def buy_and_hold(data_dir):
    """Buy first close and sell last close under the same fee model as the strategy."""
    rows = []
    strategies = {
        market: load_registry()["formal"](market=market)
        for market in ("CN-A", "HK")
    }
    for path in sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv"))):
        code = os.path.basename(path).split("_")[0]
        market = "HK" if len(code) == 5 else "CN-A"
        frame = pd.read_csv(path, usecols=["date", "close"])
        close = frame.close.astype(float)
        initial = 10000.0
        shares = initial / close.iloc[0]
        strategy = strategies[market]
        buy_fee = strategy._calculate_commission(initial, is_buy=True)
        equity = shares * close - buy_fee
        sell_amount = shares * close.iloc[-1]
        equity.iloc[-1] -= strategy._calculate_commission(sell_amount, is_buy=False)
        peak = np.maximum.accumulate(np.r_[initial, equity.to_numpy(float)])[1:]
        rows.append(dict(code=code, market=market,
                         bh=(equity.iloc[-1] / initial - 1) * 100,
                         bh_mdd=(equity.to_numpy(float) / peak - 1).min() * 100,
                         vol=close.pct_change().std() * np.sqrt(252) * 100))
    return pd.DataFrame(rows)


def diagnose(name, data_dir=DATA_DIR, workers=8):
    data_dir = resolve_pool(data_dir)
    per_stock = artifact_path(name, "stocks.csv", data_dir)
    if not os.path.exists(per_stock):
        evaluate(name, data_dir=data_dir, workers=workers)
    strat = pd.read_csv(per_stock, dtype={"code": str})
    frame = buy_and_hold(data_dir).merge(strat[["code", "ret", "mdd", "trades"]], on="code")
    frame["excess"] = frame.ret - frame.bh
    pd.set_option("display.width", 200)

    print("=== %s: strategy versus buy-and-hold (per stock, same fees) ===" % name)
    for label, sub in [("ALL", frame), ("CN-A", frame[frame.market == "CN-A"]),
                       ("HK", frame[frame.market == "HK"])]:
        if sub.empty:
            continue
        print("\n%s  n=%d" % (label, len(sub)))
        print("  strategy   median %8.2f  mean %8.2f  avg_mdd %7.2f"
              % (sub.ret.median(), sub.ret.mean(), sub.mdd.mean()))
        print("  buy&hold   median %8.2f  mean %8.2f  avg_mdd %7.2f"
              % (sub.bh.median(), sub.bh.mean(), sub.bh_mdd.mean()))
        print("  excess     median %8.2f  mean %8.2f  beats B&H on %.1f%% of stocks"
              % (sub.excess.median(), sub.excess.mean(), (sub.excess > 0).mean() * 100))

    frame["q"] = pd.qcut(frame.bh, 5, labels=["Q1 worst", "Q2", "Q3", "Q4", "Q5 best"])
    grouped = frame.groupby("q", observed=True)
    print("\n=== excess by buy-and-hold quintile (protect weak / truncate strong?) ===")
    print(grouped.agg(n=("code", "size"), bh_med=("bh", "median"), strat_med=("ret", "median"),
                      excess_med=("excess", "median"),
                      beat=("excess", lambda x: (x > 0).mean() * 100)).round(2).to_string())

    print("\n=== return attribution: where the summed return comes from ===")
    attribution = grouped.agg(n=("code", "size"), strat_sum=("ret", "sum"), bh_sum=("bh", "sum"),
                              excess_sum=("excess", "sum"))
    attribution["strat_share_%"] = attribution.strat_sum / attribution.strat_sum.sum() * 100
    print(attribution.round(1).to_string())

    print("\n=== drawdown protection by layer ===")
    print(grouped.agg(strat_mdd=("mdd", "mean"), bh_mdd=("bh_mdd", "mean")).round(1).to_string())

    rising = frame[frame.bh > 20].copy()
    if not rising.empty:
        rising["capture"] = rising.ret / rising.bh
        print("\n=== capture ratio on rising stocks (strategy return / buy-and-hold return) ===")
        print(rising.groupby("q", observed=True).capture.describe()[["count", "25%", "50%", "75%"]]
              .round(2).to_string())

    total = frame.ret.sum()
    top15 = frame.nlargest(15, "ret")
    print("\n=== concentration ===")
    print("  top 15 of %d stocks contribute %.1f%% of the summed strategy return"
          % (len(frame), top15.ret.sum() / total * 100))
    print("  their buy-and-hold contributes %.1f%% of summed buy-and-hold"
          % (top15.bh.sum() / frame.bh.sum() * 100))

    ledger = artifact_path(name, "trades.json", data_dir)
    if os.path.exists(ledger):
        returns = np.array([t["pr"] * 100 for trades in json.load(open(ledger)).values()
                            for t in trades])
        if len(returns):
            order = np.sort(returns)[::-1]
            positive, negative = returns[returns > 0].sum(), -returns[returns <= 0].sum()
            print("\n=== trade-level right tail (n=%d trades) ===" % len(returns))
            print("  win rate %.2f%% | tPF %.3f" % ((returns > 0).mean() * 100, positive / negative))
            for pct in (1, 5, 10):
                cut = max(1, int(len(order) * pct / 100))
                kept = returns[returns > 0].sum() - order[:cut].sum()
                print("  drop best %2d%% of trades -> tPF %.3f (top %d%% hold %.1f%% of gross profit)"
                      % (pct, max(kept, 0) / negative, pct, order[:cut].sum() / positive * 100))

    output = artifact_path(name, "diagnosis.csv", data_dir)
    frame.to_csv(output, index=False)
    print("\nper-stock diagnosis written to %s" % output)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    diagnose(args.candidate, data_dir=args.data_dir, workers=args.workers)


if __name__ == "__main__":
    main()
