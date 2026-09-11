"""Build Pool B: fetch Yahoo bars, reconstruct HFQ, quality-filter with same-decile replacement."""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd, requests

WORK = "/tmp/cleanpool"
APP = "/home/jaden/carp/carp_proj_2.0/stock_trading_advisor"
OUT = os.path.join(APP, "data", "pool_b")
START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-09-04")
P1, P2 = 1577232000, 1789171200          # 2019-12-25 .. 2026-09-08 (fetch window)
MIN_ROWS, MAX_ABS_MOVE, MAX_REPAIR_ERR = 510, 0.90, 0.05
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_tl = threading.local()


def session():
    if not hasattr(_tl, "s"):
        s = requests.Session(); s.headers.update(UA); _tl.s = s
    return _tl.s


def fetch(symbol):
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s"
           "?period1=%d&period2=%d&interval=1d&events=div%%2Csplit&includeAdjustedClose=true"
           % (symbol, P1, P2))
    last = None
    for attempt in range(3):
        try:
            r = session().get(url, timeout=60)
            if r.status_code == 200:
                return r.json()["chart"]["result"][0]
            last = "http%d" % r.status_code
        except Exception as exc:
            last = type(exc).__name__
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last or "unknown")


def build(symbol, code):
    res = fetch(symbol)
    meta, ind = res["meta"], res["indicators"]
    q = ind["quote"][0]
    adj = ind.get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        raise RuntimeError("no_adjclose")
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(meta["exchangeTimezoneName"])
    d = pd.DataFrame({"date": pd.to_datetime(ts.date), "open": q["open"], "close": q["close"],
                      "high": q["high"], "low": q["low"], "volume": q["volume"], "adjusted": adj})
    d = d[(d.date >= START) & (d.date <= END)]
    d = d.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    n0 = len(d)
    px = ["open", "close", "high", "low"]
    d = d.dropna(subset=px + ["adjusted"])
    d = d[(d[px] > 0).all(axis=1) & (d.adjusted > 0)]
    if d.empty:
        raise RuntimeError("empty_after_clean")
    f = (d.adjusted / d.close).to_numpy(float)
    d = d.assign(**{c: d[c].to_numpy(float) * f / f[0] for c in px})
    d["volume"] = pd.to_numeric(d.volume, errors="coerce").fillna(0.0).astype(float)
    ch, cl = d[px].max(axis=1), d[px].min(axis=1)
    he = ((ch - d.high) / d.high).abs().max() * 100
    le = ((d.low - cl) / d.low).abs().max() * 100
    info = {"dropped_nonpositive_or_nan_rows": int(n0 - len(d)),
            "repaired_high_rows": int((ch > d.high + 1e-12).sum()),
            "repaired_low_rows": int((cl < d.low - 1e-12).sum()),
            "max_repair_error_pct": round(float(max(he, le)), 4)}
    d["high"], d["low"] = ch, cl
    for c in px:
        d[c] = d[c].round(4)
    d["code"] = code
    d = d[["date", "open", "close", "high", "low", "volume", "code"]].reset_index(drop=True)
    info.update(rows=len(d), first=str(d.date.iloc[0].date()), last=str(d.date.iloc[-1].date()),
                max_abs_daily_change_pct=round(float(d.close.pct_change().abs().max() * 100), 2),
                min_price=round(float(d[px].to_numpy().min()), 6))
    return d, info


def reject(i):
    if i["rows"] < MIN_ROWS:
        return "rows<%d (%d)" % (MIN_ROWS, i["rows"])
    if i["min_price"] <= 0:
        return "nonpositive_price"
    if i["max_abs_daily_change_pct"] > MAX_ABS_MOVE * 100:
        return "daily_move>90%% (%s%%)" % i["max_abs_daily_change_pct"]
    if i["max_repair_error_pct"] > MAX_REPAIR_ERR * 100:
        return "invalid_ohlc (%s%%)" % i["max_repair_error_pct"]
    if i["last"] < str(END.date()):
        return "last_date<%s (%s)" % (END.date(), i["last"])
    if i["first"] > "2020-02-10":
        return "first_date>2020-02-10 (%s)" % i["first"]
    return None


def attempt(row):
    try:
        frame, info = build(row["symbol"], row["code"])
        return row["code"], frame, info, reject(info)
    except Exception as exc:
        return row["code"], None, None, "fetch_error:%s:%s" % (type(exc).__name__, str(exc)[:60])


def draw(order, per_decile, tag):
    """Walk each decile in fixed shuffled order, keep the first `per_decile` that pass quality."""
    kept, drops, shortfall = [], [], {}
    for dec in sorted(order):
        g = order[dec]
        rows = g.to_dict("records")
        got, i = [], 0
        while len(got) < per_decile and i < len(rows):
            chunk = rows[i:i + max(4, (per_decile - len(got)) + 8)]
            with ThreadPoolExecutor(max_workers=8) as ex:
                res = {c: (f, inf, rej) for c, f, inf, rej in ex.map(attempt, chunk)}
            for r in chunk:
                if len(got) >= per_decile:
                    break
                f, inf, rej = res[r["code"]]
                if rej:
                    drops.append({"code": r["code"], "market": tag, "decile": int(dec),
                                  "reason": rej, "symbol": r["symbol"]})
                else:
                    got.append((r, f, inf))
            i += len(chunk)
        if len(got) < per_decile:
            shortfall[int(dec)] = per_decile - len(got)
        kept.extend((r, f, inf, int(dec)) for r, f, inf in got)
        print("  %s decile %d: kept %d, attempts %d" % (tag, dec, len(got), i), flush=True)
    return kept, drops, shortfall


if __name__ == "__main__":
    sys.path.insert(0, WORK)
    from select_pool_b import decile_order, SEED
    cn = pd.read_pickle(f"{WORK}/cn_eligible.pkl")
    hk = pd.read_pickle(f"{WORK}/hk_eligible.pkl")
    hk = hk  # already floored in select_pool_b
    out = {}
    for tag, df, per in (("CN-A", cn, 40), ("HK", hk, 20)):
        _, order = decile_order(df, seed=SEED)
        kept, drops, short = draw(order, per, tag)
        out[tag] = {"kept": kept, "drops": drops, "shortfall": short}
        print("%s kept=%d drops=%d shortfall=%s" % (tag, len(kept), len(drops), short), flush=True)
    os.makedirs(OUT, exist_ok=True)
    records = []
    for tag in ("CN-A", "HK"):
        for r, f, inf, dec in out[tag]["kept"]:
            f.to_csv(os.path.join(OUT, "%s_hfq.csv" % r["code"]), index=False)
            records.append({"code": r["code"], "market": tag, "yahoo_symbol": r["symbol"],
                            "name": r.get("longName") or r.get("shortName"),
                            "rows": inf["rows"], "first_date": inf["first"], "last_date": inf["last"],
                            "market_cap": float(r["marketCap"]),
                            "avg_daily_turnover_3m": round(float(r["turn"]), 1),
                            "market_cap_decile": dec,
                            "first_trade_date": str(pd.Timestamp(r["ft"]).date()),
                            "sector": r.get("sector"), "industry": r.get("industry"),
                            "currency": r.get("currency"),
                            "max_abs_daily_change_pct": inf["max_abs_daily_change_pct"],
                            "dropped_nonpositive_or_nan_rows": inf["dropped_nonpositive_or_nan_rows"],
                            "repaired_high_rows": inf["repaired_high_rows"],
                            "repaired_low_rows": inf["repaired_low_rows"],
                            "max_repair_error_pct": inf["max_repair_error_pct"]})
    json.dump({"records": records,
               "drops": out["CN-A"]["drops"] + out["HK"]["drops"],
               "shortfall": {"CN-A": out["CN-A"]["shortfall"], "HK": out["HK"]["shortfall"]}},
              open(f"{WORK}/build_result.json", "w"), indent=1)
    print("wrote %d files to %s" % (len(records), OUT))
