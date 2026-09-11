"""Rebuild the fixed revealed-OOS daily bars listed in manifest.json.

This is a research data-maintenance tool, not part of the formal offline report path. Yahoo may revise
historical bars, so a later rebuild is not guaranteed to be byte-identical to the committed snapshot.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests


HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
OUT = os.path.join(APP, "data", "oos_data")
MANIFEST = os.path.join(HERE, "manifest.json")
START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2026-09-04")
P1, P2 = 1577232000, 1789084800
MIN_ROWS, MAX_ABS_MOVE, MAX_REPAIR_ERR = 510, 0.90, 0.05
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

_thread_local = threading.local()


def session():
    if not hasattr(_thread_local, "session"):
        value = requests.Session()
        value.headers.update(UA)
        _thread_local.session = value
    return _thread_local.session


def fetch(symbol):
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/%s"
        "?period1=%d&period2=%d&interval=1d&events=div%%2Csplit&includeAdjustedClose=true"
        % (symbol, P1, P2)
    )
    last_error = None
    for attempt in range(3):
        try:
            response = session().get(url, timeout=60)
            if response.status_code == 200:
                return response.json()["chart"]["result"][0]
            last_error = "http%d" % response.status_code
        except Exception as exc:  # network failures are reported after bounded retries
            last_error = type(exc).__name__
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_error or "unknown")


def build(symbol, code):
    result = fetch(symbol)
    meta, indicators = result["meta"], result["indicators"]
    quote = indicators["quote"][0]
    adjusted = indicators.get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        raise RuntimeError("no_adjclose")

    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    timestamps = timestamps.tz_convert(meta["exchangeTimezoneName"])
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps.date),
            "open": quote["open"],
            "close": quote["close"],
            "high": quote["high"],
            "low": quote["low"],
            "volume": quote["volume"],
            "adjusted": adjusted,
        }
    )
    data = data[(data.date >= START) & (data.date <= END)]
    data = data.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    original_rows = len(data)
    prices = ["open", "close", "high", "low"]
    data = data.dropna(subset=prices + ["adjusted"])
    data = data[(data[prices] > 0).all(axis=1) & (data.adjusted > 0)]
    if data.empty:
        raise RuntimeError("empty_after_clean")

    factor = (data.adjusted / data.close).to_numpy(float)
    data = data.assign(
        **{column: data[column].to_numpy(float) * factor / factor[0] for column in prices}
    )
    data["volume"] = pd.to_numeric(data.volume, errors="coerce").fillna(0.0).astype(float)

    corrected_high = data[prices].max(axis=1)
    corrected_low = data[prices].min(axis=1)
    high_error = ((corrected_high - data.high) / data.high).abs().max() * 100
    low_error = ((data.low - corrected_low) / data.low).abs().max() * 100
    info = {
        "dropped_nonpositive_or_nan_rows": int(original_rows - len(data)),
        "repaired_high_rows": int((corrected_high > data.high + 1e-12).sum()),
        "repaired_low_rows": int((corrected_low < data.low - 1e-12).sum()),
        "max_repair_error_pct": round(float(max(high_error, low_error)), 4),
    }
    data["high"], data["low"] = corrected_high, corrected_low
    for column in prices:
        data[column] = data[column].round(4)
    data["code"] = code
    data = data[["date", "open", "close", "high", "low", "volume", "code"]]
    info.update(
        rows=len(data),
        first=str(data.date.iloc[0].date()),
        last=str(data.date.iloc[-1].date()),
        max_abs_daily_change_pct=round(float(data.close.pct_change().abs().max() * 100), 2),
    )
    return data.reset_index(drop=True), info


def quality_reject(info):
    if info["rows"] < MIN_ROWS:
        return "rows<%d (%d)" % (MIN_ROWS, info["rows"])
    if info["max_abs_daily_change_pct"] > MAX_ABS_MOVE * 100:
        return "daily_move>%.0f%% (%s)" % (MAX_ABS_MOVE * 100, info["max_abs_daily_change_pct"])
    if info["max_repair_error_pct"] > MAX_REPAIR_ERR * 100:
        return "invalid_ohlc (%s%%)" % info["max_repair_error_pct"]
    if info["last"] < str(END.date()):
        return "last_date<%s (%s)" % (END.date(), info["last"])
    if info["first"] > "2020-02-10":
        return "first_date>2020-02-10 (%s)" % info["first"]
    return None


def try_build(stock):
    try:
        frame, info = build(stock["yahoo_symbol"], str(stock["code"]))
        return stock, frame, quality_reject(info)
    except Exception as exc:
        return stock, None, "%s:%s" % (type(exc).__name__, str(exc)[:80])


def main():
    with open(MANIFEST, encoding="utf-8") as stream:
        stocks = json.load(stream)["stocks"]
    codes = [str(stock["code"]) for stock in stocks]
    if len(codes) != len(set(codes)):
        raise SystemExit("manifest contains duplicate stock codes")

    expected = {"%s_hfq.csv" % code for code in codes}
    existing = set(os.listdir(OUT)) if os.path.isdir(OUT) else set()
    extras = sorted(name for name in existing if name.endswith("_hfq.csv") and name not in expected)
    if extras:
        raise SystemExit(
            "refusing rebuild while OOS directory contains files outside manifest: %s"
            % ", ".join(extras)
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(try_build, stocks))
    failures = [(stock["code"], error) for stock, _, error in results if error]
    if failures:
        for code, error in failures:
            print("%s: %s" % (code, error))
        raise SystemExit("refusing partial OOS rebuild: %d/%d failed" % (len(failures), len(stocks)))

    os.makedirs(OUT, exist_ok=True)
    for stock, frame, _ in results:
        frame.to_csv(os.path.join(OUT, "%s_hfq.csv" % stock["code"]), index=False)
    print("rebuilt %d fixed revealed-OOS files under %s" % (len(results), OUT))


if __name__ == "__main__":
    main()
