"""Pool B universe selection: liquidity + data-quality only, stratified by market-cap decile."""
import json, os, pickle
import numpy as np, pandas as pd

WORK = "/tmp/cleanpool"
APP = "/home/jaden/carp/carp_proj_2.0/stock_trading_advisor"
POOL305 = {l.strip() for l in open(os.path.join(APP,"backtest_benchmark","codes_305.txt")) if l.strip()}
POOL230 = {l.strip() for l in open(os.path.join(APP,"oos_benchmark","codes_oos_230.txt")) if l.strip()}
EXCLUDE = POOL305 | POOL230

SEED = 20260909
CN_TURN_FLOOR = 5e7
HK_TURN_FLOOR = 5e6
LIST_BY = pd.Timestamp("2019-07-01")
CN_PFX = ("600","601","603","605","688","000","001","002","003","300","301")

def eligible():
    o = pickle.load(open(f"{WORK}/screener_raw.pkl","rb"))
    cn = pd.DataFrame(o["cn"]).drop_duplicates("symbol")
    hk = pd.DataFrame(o["hk"]).drop_duplicates("symbol")
    for d in (cn, hk):
        d["ft"] = pd.to_datetime(d.firstTradeDateMilliseconds, unit="ms", errors="coerce")
        d["turn"] = d.averageDailyVolume3Month * d.regularMarketPrice

    cn["code"] = cn.symbol.str.split(".").str[0]
    c = cn[(cn.quoteType=="EQUITY") & cn.code.str.len().eq(6) & cn.code.str.startswith(CN_PFX)].copy()
    c = c[c.ft.notna() & (c.ft<=LIST_BY) & c.marketCap.notna() & (c.regularMarketPrice>0)]
    c = c[c.turn >= CN_TURN_FLOOR]
    c = c[~c.code.isin(EXCLUDE)]
    c["market"] = "CN-A"

    hk["n"] = pd.to_numeric(hk.symbol.str.split(".").str[0], errors="coerce")
    h = hk[(hk.quoteType=="EQUITY") & hk.n.notna()].copy()
    h = h[(h.n>=1)&(h.n<=9999)]
    h["code"] = h.n.astype(int).astype(str).str.zfill(5)
    h = h[h.ft.notna() & (h.ft<=LIST_BY) & h.marketCap.notna() & (h.regularMarketPrice>0)]
    h = h[h.turn >= HK_TURN_FLOOR]
    h = h[~h.code.isin(EXCLUDE)]
    h["market"] = "HK"
    return c.reset_index(drop=True), h.reset_index(drop=True)

def decile_order(df, seed=SEED):
    d = df.sort_values("marketCap", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d))*10//len(d)) + 1     # 1 = largest cap
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
        dd, _ = decile_order(d)
        g = dd.groupby("decile").marketCap.agg(["count","min","max"])
        g[["min","max"]] = (g[["min","max"]]/1e9).round(3)
        print(name); print(g.to_string())
    c.to_pickle(f"{WORK}/cn_eligible.pkl"); h.to_pickle(f"{WORK}/hk_eligible.pkl")
