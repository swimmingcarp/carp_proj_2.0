"""Build Pool C: stratified draw with same-decile replacement, then write HFQ bars.

NO STRATEGY IS EVER RUN HERE. The only per-stock quantities computed are data-quality statistics
(row count, first/last date, adjustment-repair error, largest single-session move).
"""
import json, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

WORK = "/tmp/exp_fresh_pool"
sys.path.insert(0, WORK)
from select_pool_c import eligible, decile_order, SEED, CN_TURN_FLOOR
from bars import attempt, CACHE

APP = "/home/jaden/carp/carp_proj_2.0/stock_trading_advisor"
OUT = os.path.join(APP, "data", "pool_c")
HK_TURN_FLOOR = 2e6          # see manifest: chosen from the ladder on liquidity+quality counts only
CN_PER_DECILE = 28           # 280 CN-A target
HK_PER_DECILE = 12           # 120 HK target


def draw(order, per_decile, tag):
    kept, drops, shortfall, attempts = [], [], {}, {}
    for dec in sorted(order):
        rows = order[dec].to_dict("records")
        got, i = [], 0
        while len(got) < per_decile and i < len(rows):
            chunk = rows[i:i + max(4, (per_decile - len(got)) + 8)]
            with ThreadPoolExecutor(max_workers=8) as ex:
                res = {c: (inf, rej) for c, inf, rej in ex.map(attempt, chunk)}
            for r in chunk:
                if len(got) >= per_decile:
                    break
                inf, rej = res[r["code"]]
                if rej:
                    drops.append({"code": r["code"], "market": tag, "decile": int(dec),
                                  "reason": rej, "symbol": r["symbol"]})
                else:
                    got.append((r, inf))
            i += len(chunk)
        if len(got) < per_decile:
            shortfall[int(dec)] = per_decile - len(got)
        attempts[int(dec)] = i
        kept.extend((r, inf, int(dec)) for r, inf in got)
        print("  %s decile %2d: kept %2d / %2d, walked %d of %d" %
              (tag, dec, len(got), per_decile, i, len(rows)), flush=True)
    return kept, drops, shortfall, attempts


if __name__ == "__main__":
    cn, hk = eligible(hk_floor=HK_TURN_FLOOR)
    print("eligible CN-A=%d  HK=%d" % (len(cn), len(hk)))
    out = {}
    for tag, df, per in (("CN-A", cn, CN_PER_DECILE), ("HK", hk, HK_PER_DECILE)):
        _, order = decile_order(df, seed=SEED)
        kept, drops, short, att = draw(order, per, tag)
        out[tag] = dict(kept=kept, drops=drops, shortfall=short, attempts=att)
        print("%s kept=%d drops=%d shortfall=%s" % (tag, len(kept), len(drops), short), flush=True)

    os.makedirs(OUT, exist_ok=True)
    records = []
    for tag in ("CN-A", "HK"):
        for r, inf, dec in out[tag]["kept"]:
            # verbatim copy: a pandas round-trip would strip the leading zeros from the code column
            shutil.copy(os.path.join(CACHE, "%s_hfq.csv" % r["code"]),
                        os.path.join(OUT, "%s_hfq.csv" % r["code"]))
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
               "shortfall": {"CN-A": out["CN-A"]["shortfall"], "HK": out["HK"]["shortfall"]},
               "attempts": {"CN-A": out["CN-A"]["attempts"], "HK": out["HK"]["attempts"]},
               "eligible": {"CN-A": len(cn), "HK": len(hk)}},
              open(f"{WORK}/build_result.json", "w"), indent=1)
    print("wrote %d files to %s" % (len(records), OUT))
