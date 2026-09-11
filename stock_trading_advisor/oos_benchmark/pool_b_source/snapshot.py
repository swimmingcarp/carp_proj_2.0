"""Pull a full Yahoo Finance equity-screener snapshot for region=cn and region=hk."""
import pickle, time, sys
import yfinance as yf
from yfinance import EquityQuery

FIELDS = ["symbol","quoteType","marketCap","regularMarketPrice","averageDailyVolume3Month",
          "firstTradeDateMilliseconds","sector","industry","longName","shortName","exchange",
          "fullExchangeName","currency","region","sharesOutstanding"]

def pull(region):
    q = EquityQuery("eq", ["region", region])
    out, offset, size = [], 0, 250
    total = None
    while True:
        for attempt in range(4):
            try:
                r = yf.screen(q, size=size, offset=offset, sortField="intradaymarketcap", sortAsc=False)
                break
            except Exception as e:
                print("retry", region, offset, type(e).__name__, e, file=sys.stderr)
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError("screener failed at offset %d" % offset)
        total = r.get("total", total)
        quotes = r.get("quotes", [])
        if not quotes:
            break
        out.extend({k: q0.get(k) for k in FIELDS} for q0 in quotes)
        offset += len(quotes)
        print(region, offset, "/", total, file=sys.stderr)
        if offset >= (total or 0):
            break
        time.sleep(0.35)
    return out, total

res = {}
for reg in ("cn", "hk"):
    rows, total = pull(reg)
    res[reg] = rows
    res[reg + "_total"] = total
    print("PULLED", reg, len(rows), "of", total)
pickle.dump(res, open("/tmp/cleanpool/screener_raw.pkl", "wb"))
