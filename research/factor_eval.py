#!/usr/bin/env python3
"""Single-factor evaluation with a research design that resists the failure mode of this project.

Split by TIME, and purge labels that cross the boundary:
  discover  2020-01..2023-12  (development pool)
  validate  2024-01..2026-09  (development pool; revealed after first use)
  external  the already-revealed 230-stock pool   (regression evidence only)

A factor is only interesting if its sign is the SAME in discover and validate. Everything else is noise
that happened to fit. Reported per factor:
  IC     mean cross-sectional Spearman with the forward return, per market, per window
  ICIR   IC divided by its standard deviation across days
  Q5-Q1  forward-return spread between the top and bottom cross-sectional quintile

    python3 research/factor_eval.py                 # discover vs validate, dev pool
    python3 research/factor_eval.py --pool oos      # revealed external regression
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import factors as F  # noqa: E402
from harness import OUT  # noqa: E402

DISCOVER = ("2020-01-01", "2023-12-31")
VALIDATE = ("2024-01-01", "2026-12-31")


def _ic_frame(panel, name, fwd, market):
    sub = panel[panel.market == market][["date", name, fwd]].dropna()
    if sub.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ic = sub.groupby("date").apply(
        lambda g: g[name].corr(g[fwd], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False).dropna()

    def spread(g):
        if len(g) < 25:
            return np.nan
        q = pd.qcut(g[name], 5, labels=False, duplicates="drop")
        if q is None or pd.Series(q).nunique() < 5:
            return np.nan
        return g[fwd][q == 4].mean() - g[fwd][q == 0].mean()

    sp = panel[panel.market == market][["date", name, fwd]].dropna().groupby("date").apply(
        spread, include_groups=False).dropna()
    return ic, sp


def period_mask(panel, fwd, lower, upper):
    """Keep observations whose feature date and forward-return endpoint stay inside a period."""
    target = fwd + "_date"
    if target not in panel:
        raise ValueError("missing %s; rebuild factors with research/factors.py" % target)
    return (panel.date >= lower) & (panel[target] <= upper)


def evaluate(panel, window, markets=("CN-A", "HK"), periods=(("discover", DISCOVER), ("validate", VALIDATE))):
    fwd = "fwd%d" % window
    rows = []
    for name in F.factor_names():
        row = {"factor": name, "group": F.FACTOR_DOC[name][0]}
        for label, (lo, hi) in periods:
            mask = period_mask(panel, fwd, lo, hi)
            for market in markets:
                ic, sp = _ic_frame(panel[mask], name, fwd, market)
                tag = "%s_%s" % (label, market)
                row[tag + "_IC"] = ic.mean() * 100 if len(ic) else np.nan
                row[tag + "_ICIR"] = (ic.mean() / ic.std()) if len(ic) > 20 and ic.std() > 0 else np.nan
                row[tag + "_Q5Q1"] = sp.mean() if len(sp) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="dev")
    parser.add_argument("--window", type=int, default=20, choices=(20, 40, 60))
    args = parser.parse_args()

    panel = F.load(args.pool)
    if args.pool == "oos":
        print("REVEALED OOS230 regression run. Do not tune factors against this output.\n")
        table = evaluate(panel, args.window, periods=(("external", ("2020-01-01", "2026-12-31")),))
        cols = [c for c in table.columns if c.startswith("external")]
    else:
        table = evaluate(panel, args.window)
        cols = [c for c in table.columns if c.startswith(("discover", "validate"))]

    pd.set_option("display.width", 250)
    print("Forward window: %d bars   (IC in %%, Q5Q1 in %% forward return)\n" % args.window)
    show = table[["factor", "group"] + cols].copy()

    if args.pool != "oos":
        for market in ("CN-A", "HK"):
            d, v = "discover_%s_IC" % market, "validate_%s_IC" % market
            show["consistent_%s" % market] = np.sign(show[d]) == np.sign(show[v])
        signs = show[["discover_CN-A_IC", "validate_CN-A_IC",
                      "discover_HK_IC", "validate_HK_IC"]].apply(np.sign)
        show["consistent"] = signs.notna().all(axis=1) & signs.eq(signs.iloc[:, 0], axis=0).all(axis=1)
        show = show.sort_values("validate_CN-A_IC", ascending=False)
    print(show.round(3).to_string(index=False))

    if args.pool != "oos":
        keep = show[show.consistent & (show["validate_CN-A_IC"].abs() > 0.5)]
        print("\nSign-consistent in BOTH markets across BOTH periods, |validate IC| > 0.5%%: %d of %d"
              % (len(keep), len(show)))
        if len(keep):
            print("  " + ", ".join(keep.factor))
    out = os.path.join(OUT, "factor_eval_%s_%d.csv" % (args.pool, args.window))
    show.to_csv(out, index=False)
    print("\nwritten: %s" % out)


if __name__ == "__main__":
    main()
