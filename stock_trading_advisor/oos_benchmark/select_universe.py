"""Historical audit code for the revealed out-of-pool universe selection.

The 2026-09-08 Yahoo screener snapshot is not present in this repository, so this script documents the
selection algorithm but cannot reproduce the exact draw by itself. The fixed selected identities live
in manifest.json and the codes_oos_*.txt files; build_oos.py rebuilds bars for those identities.

Historical rule:
  Source snapshot : Yahoo Finance equity screener, region=cn (7432 rows) and region=hk (3485 rows),
                    pulled 2026-09-08, sorted by market cap desc.
  Eligibility (no return / performance input whatsoever):
    CN-A : symbol *.SS/*.SZ, 6-digit code with board prefix in
           600/601/603/605/688 (SSE main+STAR) or 000/001/002/003/300/301 (SZSE main+SME+ChiNext);
           quoteType EQUITY; excludes B-shares (200/900), ETFs, Beijing Stock Exchange.
    HK   : symbol *.HK, numeric code 1..9999 (main board + GEM); quoteType EQUITY;
           excludes 11xxx-14xxx equity-linked instruments and 8xxxx dual/RMB counters.
    Both : firstTradeDate <= 2019-07-01 (so a full 2020-01-01 history is possible),
           marketCap present, regularMarketPrice > 0,
           liquidity floor on 3-month average daily turnover
             (averageDailyVolume3Month * regularMarketPrice):
             CN-A >= 50,000,000 CNY ; HK >= 10,000,000 HKD,
           code NOT among the 305 development-pool codes.
  Sampling:
    Eligible names are ranked by market cap and cut into 10 equal-count deciles.
    Within each decile the names are shuffled with numpy default_rng(20260908) and taken
    in shuffled order; the first N that pass the data-quality filter are kept
    (CN-A 15/decile = 150, HK 8/decile = 80).
  Nothing about historical return, drawdown, trend or strategy behaviour enters the rule.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "source_snapshot")
POOL = set(l.strip() for l in open(
    os.path.join(os.path.dirname(HERE), "backtest_benchmark", "codes_305.txt")) if l.strip())
SEED = 20260908
CN_TURN_FLOOR = 5e7
HK_TURN_FLOOR = 1e7
LIST_BY = pd.Timestamp("2019-07-01")
CN_PFX = ("600", "601", "603", "605", "688", "000", "001", "002", "003", "300", "301")


def eligible():
    snapshot = os.path.join(WORK, "screener_raw.pkl")
    if not os.path.exists(snapshot):
        raise FileNotFoundError(
            "historical Yahoo screener snapshot is not retained; use manifest.json for the fixed pool"
        )
    o = pickle.load(open(snapshot, "rb"))
    cn = pd.DataFrame(o["cn"]).drop_duplicates("symbol")
    hk = pd.DataFrame(o["hk"]).drop_duplicates("symbol")
    for d in (cn, hk):
        d["ft"] = pd.to_datetime(d.firstTradeDateMilliseconds, unit="ms", errors="coerce")
        d["turn"] = d.averageDailyVolume3Month * d.regularMarketPrice

    cn["code"] = cn.symbol.str.split(".").str[0]
    c = cn[(cn.quoteType == "EQUITY") & cn.code.str.len().eq(6) & cn.code.str.startswith(CN_PFX)].copy()
    c = c[c.ft.notna() & (c.ft <= LIST_BY) & c.marketCap.notna() & (c.regularMarketPrice > 0)]
    c = c[c.turn >= CN_TURN_FLOOR]
    c = c[~c.code.isin(POOL)]
    c["market"] = "CN-A"

    hk["n"] = pd.to_numeric(hk.symbol.str.split(".").str[0], errors="coerce")
    h = hk[(hk.quoteType == "EQUITY") & hk.n.notna()].copy()
    h = h[(h.n >= 1) & (h.n <= 9999)]
    h["code"] = h.n.astype(int).astype(str).str.zfill(5)
    h = h[h.ft.notna() & (h.ft <= LIST_BY) & h.marketCap.notna() & (h.regularMarketPrice > 0)]
    h = h[h.turn >= HK_TURN_FLOOR]
    h = h[~h.code.isin(POOL)]
    h["market"] = "HK"
    return c.reset_index(drop=True), h.reset_index(drop=True)


def decile_order(df, seed=SEED):
    """Rank by market cap, cut into 10 equal-count deciles, shuffle inside each."""
    d = df.sort_values("marketCap", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * 10 // len(d)) + 1          # 1 = largest cap
    rng = np.random.default_rng(seed)
    order = {}
    for k, g in d.groupby("decile"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        order[int(k)] = d.loc[idx].reset_index(drop=True)
    return d, order


if __name__ == "__main__":
    c, h = eligible()
    print("eligible CN-A:", len(c), " HK:", len(h))
    for name, d in (("CN-A", c), ("HK", h)):
        dd, order = decile_order(d)
        print(f"{name} mcap deciles (bn local ccy):")
        print(dd.groupby("decile").marketCap.agg(["count", "min", "max"]).div(1e9).round(2).assign(
            count=dd.groupby("decile").size()).to_string())
    json.dump({"cn_eligible": len(c), "hk_eligible": len(h)}, open(f"{WORK}/eligible_counts.json", "w"))
    c.to_pickle(f"{WORK}/cn_eligible.pkl"); h.to_pickle(f"{WORK}/hk_eligible.pkl")
