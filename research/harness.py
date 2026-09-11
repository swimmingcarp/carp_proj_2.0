#!/usr/bin/env python3
"""Isolated candidate evaluation: any candidate x the full offline pool x 7 cohorts.

Never touches stock_trading_advisor/. Results go to research/out/.

    python3 research/harness.py --selfcheck              # assert hooked == formal strategy
    python3 research/harness.py baseline my_candidate    # evaluate on the 305 development pool
    python3 research/harness.py baseline --fast          # screen on 30 representative stocks (fast)
    python3 research/harness.py baseline --pool oos      # evaluate on the revealed OOS230 pool
    python3 research/harness.py --list                   # show available candidates
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from hooked import APP, DATA_DIR, load_registry  # noqa: E402

OUT = os.path.join(HERE, "out")
BENCH = os.path.join(APP, "backtest_benchmark")
OOS_DIR = os.path.join(APP, "data", "oos_data")

METRIC_COLS = ["n", "mean", "trim10", "median", "p25", "p75", "profitable", "losers", "avg_mdd",
               "med_mdd", "avg_win", "tPF", "aPF", "avg_trades", "trade_win", "hold_ann_med",
               "top5", "port_cagr", "port_mdd", "port_sharpe"]


FAST_SET = os.path.join(HERE, "fast_set.txt")


def fast_files():
    """A 30-stock stratified subset for rapid iteration.

    Stratified by market, volatility tercile and path-efficiency half, so a quick read reflects the
    full pool rather than a lucky corner. Use it to SCREEN many candidates cheaply. A candidate that
    looks good here still has to be confirmed on the full pool: 30 stocks cannot resolve a few percent
    of edge, and screening on them repeatedly will eventually fit them.
    """
    codes = [l.strip() for l in open(FAST_SET) if l.strip()]
    return [os.path.join(DATA_DIR, "%s_hfq.csv" % c) for c in codes]


def cohorts():
    """ALL / DEV250 / OLD128 / NEW122 / HK55 / HK99 / CN-A, read from the benchmark manifest."""
    manifest = json.load(open(os.path.join(BENCH, "manifest.json"), encoding="utf-8"))
    selected = manifest["selected"]
    dev250 = {l.strip() for l in open(os.path.join(BENCH, "codes_250.txt")) if l.strip()}
    old128 = {s["code"] for s in selected if s.get("historical_132")}
    hk99 = {s["code"] for s in selected if s["market"] == "HK"}
    hk55 = {s["code"] for s in selected
            if s.get("holdout_20260907") or s.get("holdout_20260907_round2")}
    return {"ALL": None, "DEV250": dev250, "OLD128": old128, "NEW122": dev250 - old128,
            "HK55": hk55, "HK99": hk99, "CN-A": "CN-A"}


def resolve_pool(name):
    """Resolve the development pool, the revealed OOS pool, or a custom data directory."""
    if name in (None, "dev", "development", DATA_DIR):
        return DATA_DIR
    if name in ("oos", OOS_DIR):
        return OOS_DIR
    # Pool B: 584 selection-clean stocks split into two same-composition halves. Unlike the 305-stock
    # dev pool it carries no hindsight price-path labels, so a candidate developed on pool_b_dev and
    # accepted on pool_b_acc escapes the failure mode that killed three candidates on 2026-09-09.
    if name in ("pool_b", "pool_b_dev", "pool_b_acc", "pool_c"):
        return os.path.join(APP, "data", name)
    return name


def pool_label(data_dir):
    """Return a stable artifact label without leaking an absolute path into filenames."""
    resolved = os.path.abspath(resolve_pool(data_dir))
    if resolved == os.path.abspath(DATA_DIR):
        return "dev"
    if resolved == os.path.abspath(OOS_DIR):
        return "oos_revealed"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(resolved)) or "pool"
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]
    return "custom_%s_%s" % (base, digest)


def artifact_path(name, kind, data_dir):
    return os.path.join(OUT, "%s__%s__%s" % (name, pool_label(data_dir), kind))


def run_one(args):
    name, path, data_dir = args
    registry = load_registry()
    cls = registry[name]
    code = os.path.basename(path).split("_")[0]
    market = "HK" if len(code) == 5 else "CN-A"
    df = pd.read_csv(path)
    strategy = cls(market=market, stock_code=code)
    data, _ = strategy.analyze(df)
    bt = strategy.backtest(data, 10000.0)
    equity = data["capital"].to_numpy(float) / 10000.0
    trades = [dict(buy=str(t["buy_date"])[:10], sell=str(t["sell_date"])[:10],
                   pr=float(t["profit_rate"]), cap=float(t["capital"])) for t in bt["trades"]]
    return dict(code=code, market=market, ret=bt["total_return"], mdd=bt["max_drawdown"],
                win=bt["win_rate"], pf=bt["profit_factor"], trades=bt["total_trades"],
                tp=bt["total_profit_pct"], tl=bt["total_loss_pct"], final=bt["final_capital"],
                eq=equity.tolist(), dates=pd.to_datetime(data["date"]).astype(str).tolist(),
                tlist=trades)


def metrics(rows):
    frame = pd.DataFrame([{k: r[k] for k in
                           ("code", "market", "ret", "mdd", "win", "pf", "trades", "tp", "tl", "final")}
                          for r in rows])
    if frame.empty:
        return {c: float("nan") for c in METRIC_COLS}
    ret = frame.ret
    amount_pos = amount_neg = 0.0
    hold_ann, all_pr = [], []
    for r in rows:
        prev = 10000.0
        for t in r["tlist"]:
            change = t["cap"] - prev
            if change > 0:
                amount_pos += change
            else:
                amount_neg -= change
            prev = t["cap"]
            days = max(1, (pd.Timestamp(t["sell"]) - pd.Timestamp(t["buy"])).days)
            hold_ann.append((1 + t["pr"]) ** (365.0 / days) - 1)
            all_pr.append(t["pr"])
    positive = ret[ret > 0]
    top5 = positive.nlargest(5).sum() / positive.sum() * 100 if len(positive) else 0.0
    daily = pd.DataFrame({r["code"]: pd.Series(r["eq"], index=pd.to_datetime(r["dates"])).pct_change()
                          for r in rows}).fillna(0.0).mean(axis=1)
    curve = (1 + daily).cumprod()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    lo, hi = ret.quantile(0.1), ret.quantile(0.9)
    return dict(
        n=len(frame), mean=ret.mean(), trim10=ret[(ret > lo) & (ret < hi)].mean(), median=ret.median(),
        p25=ret.quantile(0.25), p75=ret.quantile(0.75), profitable=(ret > 0).mean() * 100,
        losers=int((ret <= 0).sum()), avg_mdd=frame.mdd.mean(), med_mdd=frame.mdd.median(),
        avg_win=frame.win.mean(), tPF=frame.tp.sum() / frame.tl.sum() if frame.tl.sum() else np.nan,
        aPF=amount_pos / amount_neg if amount_neg else np.nan, avg_trades=frame.trades.mean(),
        trade_win=float(np.mean([p > 0 for p in all_pr]) * 100) if all_pr else 0.0,
        hold_ann_med=float(np.median(hold_ann) * 100) if hold_ann else 0.0, top5=top5,
        port_cagr=(curve.iloc[-1] ** (1 / years) - 1) * 100,
        port_mdd=(curve / curve.cummax() - 1).min() * 100,
        port_sharpe=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0)


def evaluate(name, data_dir=DATA_DIR, workers=8, save=True, fast=False):
    files = fast_files() if fast else sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv")))
    if not files:
        raise SystemExit("no *_hfq.csv under %s" % data_dir)
    with Pool(workers) as pool:
        rows = pool.map(run_one, [(name, f, data_dir) for f in files], chunksize=4)
    os.makedirs(OUT, exist_ok=True)
    if save:
        pd.DataFrame([{k: r[k] for k in ("code", "market", "ret", "mdd", "win", "pf", "trades",
                                         "tp", "tl", "final")} for r in rows]
                     ).to_csv(artifact_path(name, "stocks.csv", data_dir), index=False)
        json.dump({r["code"]: r["tlist"] for r in rows},
                  open(artifact_path(name, "trades.json", data_dir), "w"))
    if os.path.abspath(data_dir) == os.path.abspath(DATA_DIR):
        groups = cohorts()
    else:
        # An out-of-pool universe has none of the development cohorts; split by market only.
        groups = {"ALL": None, "CN-A": "CN-A", "HK": "HK"}
    result = {}
    for cohort, codes in groups.items():
        if codes is None:
            subset = rows
        elif codes in ("CN-A", "HK"):
            subset = [r for r in rows if r["market"] == codes]
        else:
            subset = [r for r in rows if r["code"] in codes]
        result[cohort] = metrics(subset)
    if save:
        json.dump(result, open(artifact_path(name, "metrics.json", data_dir), "w"), indent=1)
    return result


def selfcheck(workers=8, sample=None):
    """Assert the hookable re-implementation matches the formal strategy trade for trade."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_hfq.csv")))
    if sample is not None:
        files = files[:sample]
    with Pool(workers) as pool:
        formal = pool.map(run_one, [("formal", f, DATA_DIR) for f in files], chunksize=2)
        hooked = pool.map(run_one, [("baseline", f, DATA_DIR) for f in files], chunksize=2)
    bad = []
    for a, b in zip(formal, hooked):
        if abs(a["ret"] - b["ret"]) > 1e-9 or a["trades"] != b["trades"] or a["tlist"] != b["tlist"]:
            bad.append(a["code"])
    print("selfcheck: %d/%d stocks identical" % (len(files) - len(bad), len(files)))
    if bad:
        print("MISMATCH on: %s" % ", ".join(bad))
        print("The formal strategy has changed; resynchronise research/hooked.py before trusting any"
              " candidate result.")
        return False
    return True


def show(name, result):
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("\n##### %s" % name)
    print(pd.DataFrame(result).T[METRIC_COLS].round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", nargs="*")
    parser.add_argument("--selfcheck", action="store_true", help="verify hooked == formal, then exit")
    parser.add_argument("--list", action="store_true", help="list available candidates")
    parser.add_argument("--pool", default="dev",
                        help="'dev' (305 development stocks), 'oos' (230 revealed regression stocks), "
                             "or a directory of *_hfq.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fast", action="store_true",
                        help="screen on the 30-stock stratified subset instead of the whole pool")
    args = parser.parse_args()

    if args.list:
        print("\n".join(sorted(load_registry())))
        return
    pool = resolve_pool(args.pool)
    if pool == OOS_DIR:
        print("NOTE: OOS230 is already revealed. Use it only for regression checks; never tune a\n"
              "      parameter against it or describe it as an untouched acceptance result.\n")
    if args.selfcheck or not args.candidates:
        ok = selfcheck(workers=args.workers, sample=40 if args.fast else None)
        if not args.candidates:
            raise SystemExit(0 if ok else 1)
        if not ok:
            raise SystemExit(1)
    else:
        if not selfcheck(workers=args.workers, sample=40 if args.fast else None):
            raise SystemExit(1)

    for name in args.candidates:
        show(name, evaluate(name, data_dir=pool, workers=args.workers, fast=args.fast))


if __name__ == "__main__":
    main()
