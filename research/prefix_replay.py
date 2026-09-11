#!/usr/bin/env python3
"""Day-by-day prefix replay: prove a candidate reads no information it could not have had.

At each tested bar the strategy is rebuilt from scratch on the stock's bars [0..t] and, when a market
series is injected, on that series truncated to the same date. Signals, holding state, trigger reasons,
key intermediate columns, the completed trade ledger and the open position are compared against the
full-history computation at the same bar. Any difference is a lookahead.

    python3 research/prefix_replay.py ma60_12 --stocks 300274 00700
    python3 research/prefix_replay.py my_gate --market-series /tmp/hsi.csv --stocks 00700

By default every prefix is checked. ``--points`` enables an explicitly exploratory sample and must not
be reported as a complete lookahead pass.

A market series CSV needs columns date,close and is exposed to the candidate as the class attribute
`market_series` (a pandas Series indexed by date), already truncated at each cutoff.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from hooked import DATA_DIR, load_registry  # noqa: E402

COMPARE = ["buy_signal", "entry_signal", "exit_signal", "stop_loss_exit", "profit_target_exit",
           "entry_reason", "exit_reason", "trend_direction", "dist_ma60", "atr_trailing"]


def same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    try:
        return (pd.isna(a) and pd.isna(b)) or np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9)
    except (TypeError, ValueError):
        return a == b


def build(cls, market, code, market_series):
    strategy = cls(market=market, stock_code=code)
    if market_series is not None:
        strategy.market_series = market_series
    return strategy


def select_prefixes(full, points=0, start=0, seed=11):
    all_points = list(range(start, len(full)))
    if points <= 0 or len(all_points) <= points:
        return all_points

    signals = np.where(
        full["entry_signal"].to_numpy(int) | full["exit_signal"].to_numpy(int)
    )[0]
    signal_points = [int(i) for i in signals if i >= start]
    rng = np.random.default_rng(seed)
    if len(signal_points) >= points:
        return sorted(int(i) for i in rng.choice(signal_points, size=points, replace=False))
    quiet = sorted(set(all_points) - set(signal_points))
    picked = rng.choice(quiet, size=points - len(signal_points), replace=False)
    return sorted(signal_points + [int(i) for i in picked])


def ledger(strategy, data):
    bt = strategy.backtest(data, 10000.0)
    return [(str(t["buy_date"])[:10], round(float(t["buy_price"]), 10),
             str(t["sell_date"])[:10], round(float(t["sell_price"]), 10),
             round(float(t["profit_rate"]), 10))
            for t in bt["trades"]]


def replay(name, code, points=0, market_series=None, start=0, seed=11, extra_cols=()):
    cls = load_registry()[name]
    market = "HK" if len(code) == 5 else "CN-A"
    df = pd.read_csv(os.path.join(DATA_DIR, "%s_hfq.csv" % code))
    full_strategy = build(cls, market, code, market_series)
    full, _ = full_strategy.analyze(df)
    n = len(full)
    dates = pd.to_datetime(full["date"])
    columns = [c for c in list(COMPARE) + list(extra_cols) if c in full.columns]

    tested = select_prefixes(full, points=points, start=start, seed=seed)

    diffs = []
    for t in tested:
        cutoff = dates.iloc[t]
        series = market_series[market_series.index <= cutoff] if market_series is not None else None
        part_strategy = build(cls, market, code, series)
        part, _ = part_strategy.analyze(df.iloc[: t + 1].copy())
        if part is None or len(part) != t + 1:
            diffs.append((t, "length", None, t + 1))
            continue
        for col in columns:
            if not same(part[col].iloc[-1], full[col].iloc[t]):
                diffs.append((t, col, part[col].iloc[-1], full[col].iloc[t]))
        if ledger(part_strategy, part) != ledger(full_strategy, full.iloc[: t + 1].copy()):
            diffs.append((t, "trade_ledger", None, None))
        if int(part["buy_signal"].iloc[-1]) != int(full["buy_signal"].iloc[t]):
            diffs.append((t, "open_position", int(part["buy_signal"].iloc[-1]),
                          int(full["buy_signal"].iloc[t])))
    return len(tested), diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("--stocks", nargs="+", required=True)
    parser.add_argument("--points", type=int, default=0,
                        help="sample this many prefixes (0 checks every prefix; only 0 is a formal pass)")
    parser.add_argument("--market-series", default=None,
                        help="CSV with date,close injected as `market_series` and truncated per cutoff")
    parser.add_argument("--extra-cols", nargs="*", default=[],
                        help="additional strategy columns to compare, e.g. a gate state")
    args = parser.parse_args()

    series = None
    if args.market_series:
        raw = pd.read_csv(args.market_series)
        series = pd.Series(raw["close"].to_numpy(float),
                           index=pd.to_datetime(raw["date"])).sort_index()

    total_points = total_diffs = 0
    if args.points > 0:
        print("WARNING: sampled prefix replay; this is exploratory and not a formal lookahead pass.\n")
    for code in args.stocks:
        points, diffs = replay(args.candidate, code, points=args.points, market_series=series,
                               extra_cols=args.extra_cols)
        total_points += points
        total_diffs += len(diffs)
        print("%-8s points=%4d diffs=%d" % (code, points, len(diffs)))
        for d in diffs[:6]:
            print("      ", d)
    print("\n=== %s: %d prefix points, %d differences ===" % (args.candidate, total_points, total_diffs))
    raise SystemExit(1 if total_diffs else 0)


if __name__ == "__main__":
    main()
