#!/usr/bin/env python3
"""Portfolio backtest: one capital pool, a cap on concurrent positions, an explicit ranking rule.

The single-stock harness gives every stock its own money and lets 60% of it sit idle. That framework
cannot express the question "given limited capital, which of today's signals do I buy?" - which is the
only question a cross-sectional signal can answer. This engine can.

Rules: decisions at the close, equal weight per slot, the project's real fee model, cash earns nothing.
When more entry signals fire than there are free slots, candidates are ordered by the chosen ranking
and the best are taken. A ranking is only meaningful if it beats RANDOM ranking of the same signals -
that comparison is built in. Missed entry events expire; they are never bought later merely because a
slot becomes free. Returns remain in each stock's local currency, so mixed CN-A/HK results exclude FX.

    python3 research/portfolio.py baseline --rank cs20 --slots 20
    python3 research/portfolio.py baseline --rank cs20 random rs_ibd none --slots 10 20 30
    python3 research/portfolio.py baseline --rank cs20 --pool oos --slippage 10
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from hooked import load_registry  # noqa: E402
from harness import OUT, resolve_pool, selfcheck  # noqa: E402
import panel as panel_mod  # noqa: E402

# panel-derived rankings plus every factor in factors.py; "random"/"none" are the null benchmarks
RANKINGS = ("cs20", "cs60", "rs_ibd", "F2", "random", "none", "inv_vol")


def _signals_one(args):
    """Per-stock entry/exit events and closes from the same strategy as the formal report."""
    name, path = args
    code = os.path.basename(path).split("_")[0]
    market = "HK" if len(code) == 5 else "CN-A"
    strategy = load_registry()[name](market=market, stock_code=code)
    data, _ = strategy.analyze(pd.read_csv(path))
    return code, market, pd.DataFrame({
        "date": pd.to_datetime(data["date"]),
        "entry": data["entry_signal"].to_numpy(int),
        "exit": data["exit_signal"].to_numpy(int),
        "close": data["close"].to_numpy(float),
    })


def build_signals(name, data_dir, workers=8):
    files = sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv")))
    with Pool(workers) as pool:
        rows = pool.map(_signals_one, [(name, f) for f in files], chunksize=4)
    closes, entries, exits, markets = {}, {}, {}, {}
    for code, market, frame in rows:
        frame = frame.set_index("date")
        closes[code] = frame["close"]
        entries[code] = frame["entry"]
        exits[code] = frame["exit"]
        markets[code] = market
    close = pd.DataFrame(closes).sort_index()
    entry = pd.DataFrame(entries).reindex(close.index).fillna(0).astype(int)
    exit_ = pd.DataFrame(exits).reindex(close.index).fillna(0).astype(int)
    return close, entry, exit_, markets


def fee(strategy, amount, is_buy, slippage_bps):
    return strategy._calculate_commission(amount, is_buy=is_buy) + amount * slippage_bps / 10000.0


def run(close, entry, exit_, markets, rank_table, slots, slippage_bps=0.0, seed=0,
        initial=1_000_000.0, random_rank=False):
    """rank_table: DataFrame indexed like close with a rank score per stock (higher = preferred).

    CN-A and HK trade on different calendars, so a merged frame is full of holes. A missing bar means
    "no session for this stock today": the holding is still worth its last close (carried forward for
    valuation) but cannot be bought or sold. Conflating the two silently liquidates every suspended
    position at zero, so the two prices are kept separate throughout.
    """
    registry = load_registry()
    strategies = {m: registry["baseline"](market=m) for m in ("CN-A", "HK")}
    codes = list(close.columns)
    dates = close.index
    rng = np.random.default_rng(seed)

    tradable = close.notna().to_numpy()            # a real session for that stock
    px_trade = close.to_numpy(float)               # execution price, NaN when not tradable
    px_value = close.ffill().to_numpy(float)       # valuation price, carried over holidays/halts
    entry_np = entry.reindex(index=close.index, columns=close.columns).fillna(0).to_numpy(int)
    exit_np = exit_.reindex(index=close.index, columns=close.columns).fillna(0).to_numpy(int)
    rank_np = rank_table.to_numpy(float) if rank_table is not None else None
    col = {c: i for i, c in enumerate(codes)}
    last_tradable = {
        code: int(np.flatnonzero(tradable[:, j])[-1])
        for j, code in enumerate(codes)
        if tradable[:, j].any()
    }

    cash = initial
    shares = {}
    equity_curve = np.empty(len(dates))
    invested_frac = np.empty(len(dates))
    fees_paid = 0.0
    traded_value = 0.0

    for t in range(len(dates)):
        for code in list(shares):
            j = col[code]
            must_exit = exit_np[t, j] == 1 or t == last_tradable[code]
            if must_exit and tradable[t, j]:
                amount = shares[code] * px_trade[t, j]
                del shares[code]
                f = fee(strategies[markets[code]], amount, False, slippage_bps)
                cash += amount - f
                fees_paid += f
                traded_value += amount

        free = slots - len(shares)
        if free > 0:
            cands = [codes[j] for j in range(len(codes))
                     if entry_np[t, j] == 1 and codes[j] not in shares and tradable[t, j]
                     and t < last_tradable.get(codes[j], t)]
            if cands:
                if rank_np is None:
                    scores = rng.random(len(cands)) if random_rank else None
                else:
                    scores = np.array([rank_np[t, col[c]] for c in cands], float)
                    ranked = np.isfinite(scores)
                    cands = [code for code, keep in zip(cands, ranked) if keep]
                    scores = scores[ranked]
                if scores is None:
                    picked = cands[:free]
                elif not len(scores):
                    picked = []
                else:
                    order = np.argsort(-scores, kind="stable")
                    picked = [cands[i] for i in order[:free]]
                held_value = sum(shares[c] * px_value[t, col[c]] for c in shares)
                budget = (cash + held_value) / slots
                for code in picked:
                    amount = min(budget, cash)
                    f = fee(strategies[markets[code]], amount, True, slippage_bps)
                    if amount + f > cash:
                        amount = max(0.0, cash - f)
                        f = fee(strategies[markets[code]], amount, True, slippage_bps)
                    if amount <= 0:
                        break
                    shares[code] = amount / px_trade[t, col[code]]
                    cash -= amount + f
                    fees_paid += f
                    traded_value += amount

        held = sum(shares[c] * px_value[t, col[c]] for c in shares)
        equity_curve[t] = cash + held
        invested_frac[t] = held / equity_curve[t] if equity_curve[t] > 0 else 0.0

    curve = pd.Series(equity_curve, index=dates)
    daily = curve.pct_change()
    daily.iloc[0] = curve.iloc[0] / initial - 1.0
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = ((curve.iloc[-1] / initial) ** (1 / years) - 1) * 100
    peak = curve.cummax().clip(lower=initial)
    mdd = (curve / peak - 1).min() * 100
    return {
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0,
        "calmar": cagr / abs(mdd) if mdd else np.nan,
        "invested": float(np.mean(invested_frac)) * 100,
        "final": curve.iloc[-1],
        "fees": fees_paid,
        "turnover_x": traded_value / initial / years,
        "curve": curve,
    }


def compare(name, pool="dev", slots=(10, 20, 30), rankings=("cs20", "random", "none"),
            slippage_bps=0.0, workers=8, random_seeds=10, start=None, end=None, market=None):
    data_dir = resolve_pool(pool)
    close, entry, exit_, markets = build_signals(name, data_dir, workers=workers)
    if market:
        keep = [c for c in close.columns if markets[c] == market]
        close, entry, exit_ = close[keep], entry[keep], exit_[keep]
        markets = {c: markets[c] for c in keep}
    pan = panel_mod.load("oos" if data_dir == resolve_pool("oos") else "dev")
    tables = {}
    for key in ("cs20", "cs60", "rs_ibd", "F2"):
        wide = pan.pivot_table(index="date", columns="code", values=key, aggfunc="last")
        tables[key] = wide.reindex(index=close.index, columns=close.columns).astype(float)
    inv = 1.0 / close.pct_change(fill_method=None).rolling(20).std()
    tables["inv_vol"] = inv.reindex(index=close.index, columns=close.columns)

    # any factor from factors.py may be used as a ranking, oriented so higher = better
    import factors as F
    wanted = [r for r in rankings if r in F.factor_names() or r.startswith("combo:")]
    if wanted:
        fac = F.load("oos" if data_dir == resolve_pool("oos") else "dev")
        needed = set()
        for r in wanted:
            needed.update(r[6:].split("+") if r.startswith("combo:") else [r])
        wide = {}
        for col in needed:
            w = fac.pivot_table(index="date", columns="code", values=col, aggfunc="last")
            wide[col] = w.reindex(index=close.index, columns=close.columns).astype(float)
        for r in wanted:
            if r.startswith("combo:"):
                parts = r[6:].split("+")
                # equal-weight average of cross-sectional ranks: no fitted weights, nothing to overfit
                ranks = [wide[c].rank(axis=1, pct=True) for c in parts]
                tables[r] = sum(ranks) / len(ranks)
            else:
                tables[r] = wide[r]

    if start or end:
        # slice AFTER every rolling feature is built, so no window is truncated by the split
        mask = pd.Series(True, index=close.index)
        if start:
            mask &= close.index >= pd.Timestamp(start)
        if end:
            mask &= close.index <= pd.Timestamp(end)
        close, entry, exit_ = close[mask], entry[mask], exit_[mask]
        tables = {k: v[mask] for k, v in tables.items()}

    rows = []
    for n in slots:
        for rank in rankings:
            if rank == "random":
                sub = [run(close, entry, exit_, markets, None, n, slippage_bps, seed=s,
                           random_rank=True)
                       for s in range(random_seeds)]
                rows.append(dict(slots=n, rank="random(%d seeds)" % random_seeds,
                                 cagr=np.mean([r["cagr"] for r in sub]),
                                 cagr_sd=np.std([r["cagr"] for r in sub]),
                                 mdd=np.mean([r["mdd"] for r in sub]),
                                 sharpe=np.mean([r["sharpe"] for r in sub]),
                                 calmar=np.mean([r["calmar"] for r in sub]),
                                 invested=np.mean([r["invested"] for r in sub]),
                                 fees=np.mean([r["fees"] for r in sub]),
                                 turnover_x=np.mean([r["turnover_x"] for r in sub])))
            else:
                table = None if rank == "none" else tables[rank]
                r = run(close, entry, exit_, markets, table, n, slippage_bps, seed=0)
                rows.append(dict(slots=n, rank=rank, cagr=r["cagr"], cagr_sd=np.nan, mdd=r["mdd"],
                                 sharpe=r["sharpe"], calmar=r["calmar"], invested=r["invested"],
                                 fees=r["fees"], turnover_x=r["turnover_x"]))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("--pool", default="dev")
    parser.add_argument("--slots", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--rank", nargs="+", default=["cs20", "random", "none"],
                        help="panel rankings %s, any factor from factors.py, or combo:f1+f2+f3"
                             % (RANKINGS,))
    parser.add_argument("--slippage", type=float, default=0.0, help="basis points per side")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start", default=None, help="restrict the evaluation window, e.g. 2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--market", default=None, choices=("CN-A", "HK"))
    args = parser.parse_args()

    if not selfcheck(workers=args.workers):
        raise SystemExit(1)
    table = compare(args.candidate, pool=args.pool, slots=tuple(args.slots),
                    rankings=tuple(args.rank), slippage_bps=args.slippage,
                    workers=args.workers, random_seeds=args.seeds,
                    start=args.start, end=args.end, market=args.market)
    pd.set_option("display.width", 200)
    print("\n##### %s | pool=%s | market=%s | %s..%s | slippage=%gbps per side"
          % (args.candidate, args.pool, args.market or "both", args.start or "start",
             args.end or "end", args.slippage))
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
