"""Yahoo -> HFQ bar reconstruction, byte-for-byte the same procedure as oos_benchmark/build_oos.py
and pool_b_source/build_pool_b.py. Frames are cached on disk so the draw never refetches."""
import json, os, threading, time
import numpy as np, pandas as pd, requests

WORK = "/tmp/exp_fresh_pool"
CACHE = os.path.join(WORK, "cache")
START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-09-04")
P1, P2 = 1577232000, 1789171200          # identical fetch window to pool_b
MIN_ROWS, MAX_ABS_MOVE, MAX_REPAIR_ERR = 510, 0.90, 0.05
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_tl = threading.local()
os.makedirs(CACHE, exist_ok=True)


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
    """Returns (code, info_or_None, reject_reason_or_None). Frame is cached to disk, not returned."""
    code, symbol = row["code"], row["symbol"]
    ipath = os.path.join(CACHE, "%s.info.json" % code)
    if os.path.exists(ipath):
        rec = json.load(open(ipath))
        return code, rec["info"], rec["reject"]
    try:
        frame, info = build(symbol, code)
        rej = reject(info)
        frame.to_csv(os.path.join(CACHE, "%s_hfq.csv" % code), index=False)
    except Exception as exc:
        info, rej = None, "fetch_error:%s:%s" % (type(exc).__name__, str(exc)[:60])
    json.dump({"info": info, "reject": rej, "symbol": symbol}, open(ipath, "w"))
    return code, info, rej
