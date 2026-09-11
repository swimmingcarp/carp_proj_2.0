"""Pool C universe selection: liquidity + data-quality only, stratified by market-cap decile.

Exclusions are the union of every pool this project has ever looked at:
  backtest_benchmark/codes_305.txt, oos_benchmark/codes_oos_230.txt and the FULL pool_b universe.
The pool_b exclusion is taken from the file listing of data/pool_b/ (the dev+acc union), so
pool_b_acc.txt is never opened and the dev/acc membership is never observed.
"""
import os, pickle
import numpy as np, pandas as pd

WORK = "/tmp/exp_fresh_pool"
APP = "/home/jaden/carp/carp_proj_2.0/stock_trading_advisor"

SEED = 20260911
CN_TURN_FLOOR = 5e7
HK_FLOOR_LADDER = [1e7, 5e6, 3e6, 2e6, 1e6]   # tried largest-first; see pick_hk_floor()
HK_SCAN_FLOOR = min(HK_FLOOR_LADDER)
LIST_BY = pd.Timestamp("2019-07-01")
CN_PFX = ("600", "601", "603", "605", "688", "000", "001", "002", "003", "300", "301")


def exclusions():
    p305 = {l.strip() for l in open(f"{APP}/backtest_benchmark/codes_305.txt") if l.strip()}
    p230 = {l.strip() for l in open(f"{APP}/oos_benchmark/codes_oos_230.txt") if l.strip()}
    pb = {f.split("_")[0] for f in os.listdir(f"{APP}/data/pool_b") if f.endswith("_hfq.csv")}
    return p305, p230, pb


def eligible(hk_floor=HK_SCAN_FLOOR):
    p305, p230, pb = exclusions()
    ex = p305 | p230 | pb
    o = pickle.load(open(f"{WORK}/screener_raw.pkl", "rb"))
    cn = pd.DataFrame(o["cn"]).drop_duplicates("symbol")
    hk = pd.DataFrame(o["hk"]).drop_duplicates("symbol")
    for d in (cn, hk):
        d["ft"] = pd.to_datetime(d.firstTradeDateMilliseconds, unit="ms", errors="coerce")
        d["turn"] = d.averageDailyVolume3Month * d.regularMarketPrice

    cn["code"] = cn.symbol.str.split(".").str[0]
    c = cn[(cn.quoteType == "EQUITY") & cn.code.str.len().eq(6) & cn.code.str.startswith(CN_PFX)].copy()
    c = c[c.ft.notna() & (c.ft <= LIST_BY) & c.marketCap.notna() & (c.regularMarketPrice > 0)]
    c = c[c.turn >= CN_TURN_FLOOR]
    c = c[~c.code.isin(ex)]
    c["market"] = "CN-A"

    hk["n"] = pd.to_numeric(hk.symbol.str.split(".").str[0], errors="coerce")
    h = hk[(hk.quoteType == "EQUITY") & hk.n.notna()].copy()
    h = h[(h.n >= 1) & (h.n <= 9999)]
    h["code"] = h.n.astype(int).astype(str).str.zfill(5)
    h = h[h.ft.notna() & (h.ft <= LIST_BY) & h.marketCap.notna() & (h.regularMarketPrice > 0)]
    h = h[h.turn >= hk_floor]
    h = h[~h.code.isin(ex)]
    h["market"] = "HK"
    return c.reset_index(drop=True), h.reset_index(drop=True)


def decile_order(df, seed=SEED):
    """Rank by market cap, cut into 10 equal-count deciles (1 = largest), shuffle inside each."""
    d = df.sort_values("marketCap", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * 10 // len(d)) + 1
    rng = np.random.default_rng(seed)
    order = {}
    for k, g in d.groupby("decile"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        order[int(k)] = d.loc[idx].reset_index(drop=True)
    return d, order


if __name__ == "__main__":
    c, h = eligible()
    print("eligible CN-A(>=%.0e): %d   HK(>=%.0e): %d" % (CN_TURN_FLOOR, len(c), HK_SCAN_FLOOR, len(h)))
    for name, d in (("CN-A", c), ("HK", h)):
        dd, _ = decile_order(d)
        g = dd.groupby("decile").marketCap.agg(["count", "min", "max"])
        g[["min", "max"]] = (g[["min", "max"]] / 1e9).round(3)
        print(name); print(g.to_string())
    c.to_pickle(f"{WORK}/cn_eligible.pkl"); h.to_pickle(f"{WORK}/hk_scanset.pkl")
