"""Write oos_benchmark/pool_c_manifest.json + code lists + the reproducibility bundle."""
import json, os, shutil, datetime, collections
import pandas as pd

WORK = "/tmp/exp_fresh_pool"
APP = "/home/jaden/carp/carp_proj_2.0/stock_trading_advisor"
BENCH = os.path.join(APP, "oos_benchmark")
SRC = os.path.join(BENCH, "pool_c_source")
OUT = os.path.join(APP, "data", "pool_c")

br = json.load(open(f"{WORK}/build_result.json"))
ladder = json.load(open(f"{WORK}/hk_floor_ladder.json"))
recs = sorted(br["records"], key=lambda r: (r["market"], r["code"]))
df = pd.DataFrame(recs)

p305 = {l.strip() for l in open(f"{APP}/backtest_benchmark/codes_305.txt") if l.strip()}
p230 = {l.strip() for l in open(f"{APP}/oos_benchmark/codes_oos_230.txt") if l.strip()}
pb = {f.split("_")[0] for f in os.listdir(f"{APP}/data/pool_b") if f.endswith("_hfq.csv")}
pbdev = {l.strip() for l in open(f"{APP}/oos_benchmark/pool_b_dev.txt") if l.strip()}
codes = {r["code"] for r in recs}

SELECTION_RULE = """Pool C universe selection (performance-blind; liquidity and data quality only).

Source snapshot: Yahoo Finance equity screener (yfinance 1.7.0 EquityQuery), region=cn 7,436 rows and
  region=hk 9,189 rows (9,175 unique symbols), pulled 2026-09-11, sorted by intraday market cap
  descending. The raw snapshot is retained at oos_benchmark/pool_c_source/screener_raw.pkl together
  with the four scripts that produced this pool (snapshot.py, select_pool_c.py, bars.py,
  build_pool_c.py), so the draw can be regenerated exactly.

Eligibility (no return, drawdown, volatility, trend or strategy input whatsoever):
  CN-A : symbol *.SS/*.SZ, 6-digit code with board prefix in 600/601/603/605/688 (SSE main+STAR)
         or 000/001/002/003/300/301 (SZSE main+SME+ChiNext); quoteType EQUITY;
         excludes B-shares (200/900), ETFs and the Beijing Stock Exchange.
  HK   : symbol *.HK, numeric code 1..9999 (main board + GEM); quoteType EQUITY;
         excludes 11xxx-14xxx equity-linked instruments and 8xxxx dual/RMB counters.
  Both : firstTradeDate <= 2019-07-01 (so a full 2020-01-01 history is possible),
         marketCap present, regularMarketPrice > 0,
         liquidity floor on 3-month average daily turnover
           (averageDailyVolume3Month * regularMarketPrice):
           CN-A >= 50,000,000 CNY ; HK >= 2,000,000 HKD,
         code NOT among the 305 development-pool codes, NOT among the 230 revealed-OOS codes and
         NOT among the 584 pool_b codes (dev + acc union).
  Eligible after all of the above: CN-A 2,244, HK 245.

Stratified sampling:
  Eligible names are ranked by market cap and cut into 10 equal-count deciles (decile 1 = largest cap).
  Within each decile the names are shuffled with numpy.random.default_rng(20260911) and walked in that
  fixed shuffled order; the first N that pass the data-quality filter are kept
  (CN-A 28/decile = 280 target, HK 12/decile = 120 target). A name that fails the data-quality filter
  is replaced by the next name in the same decile's shuffled order, so the strata stay balanced.
  Nothing about historical return, drawdown, trend or strategy behaviour enters the rule.

There is no dev/acceptance split. Pool C is a single sealed acceptance pool."""

HK_FLOOR_NOTE = """The HK liquidity floor was NOT fixed before the counts were seen, and this is stated
plainly rather than dressed up as a pre-registration. After excluding all three existing pools only 113
HK names clear the 10,000,000 HKD/day floor used by the 230-stock pool and only 153 clear the
5,000,000 HKD/day floor used by pool_b - too few for a 120-name stratified draw once the data-quality
filter takes its cut. All five candidate floors were therefore evaluated on two counts only: how many
names are eligible, and how many of those pass the data-quality filter. 2,000,000 HKD/day was chosen as
the highest floor whose achievable stratified draw (sum over deciles of min(12, quality-passing names))
comes within 25 names of the 120 target. The full ladder is recorded in hk_floor_ladder below: the
achievable draw is 73 at 10m, 73 at 5m, 89 at 3m, 99 at 2m and 101 at 1m - i.e. dropping the floor
from 2m to 1m HKD/day would have bought two extra names at the cost of halving the liquidity floor, so
it was not done. The only inputs to this choice were turnover, market cap, row count, adjustment
consistency and single-session-move sanity. No return, drawdown, volatility, trend or strategy quantity
entered it. 2,000,000 HKD/day is roughly 9x the median daily turnover of the HK names that clear the
listing-date screen."""

manifest = {
    "version": "2026-09-11-pool-c-fresh-acceptance-universe",
    "status": ("UNTOUCHED AND UNSEEN. No strategy, no backtest and no return statistic of any kind has "
               "been computed on any pool_c file. Pool C exists solely to accept or reject a candidate "
               "that has already been finalised elsewhere. Reading it once reveals it permanently; it "
               "should be read at most once, for one pre-specified candidate against the baseline."),
    "generated_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
    "purpose": ("Fresh acceptance universe. Every other clean pool is spent: backtest_data is the "
                "contaminated 305-stock development pool, oos_data has been revealed repeatedly, "
                "pool_b_dev is the current development ground and pool_b_acc has been revealed twice. "
                "Pool C is drawn from the same performance-blind liquidity/quality rule as pool_b and "
                "the 230-stock OOS pool, with zero overlap against all three, so a candidate finalised "
                "on pool_b_dev can be tested once on genuinely unseen names. It carries none of the "
                "hindsight labels (sideways_1y / drawdown_sample / growth / mixed) that tag 150 of the "
                "305 development names."),
    "data_dir": "stock_trading_advisor/data/pool_c",
    "file_pattern": "*_hfq.csv",
    "columns": ["date", "open", "close", "high", "low", "volume", "code"],
    "source": ("Yahoo Finance chart API (query1.finance.yahoo.com/v8/finance/chart); universe listing "
               "from the Yahoo Finance equity screener (yfinance 1.7.0 EquityQuery region=cn / region=hk)"),
    "adjust": ("Yahoo-derived hfq-equivalent (后复权) series: hfq_t = raw_t * (adjclose_t/close_t) / "
               "(adjclose_0/close_0), applied to open/close/high/low and anchored so the first retained "
               "bar equals the raw price. Volume is left raw. High/low are then set to max/min of the "
               "four adjusted prices and the repair error is bounded by the quality filter. This is the "
               "identical procedure used by oos_benchmark/build_oos.py and pool_b_source/build_pool_b.py."),
    "start_date_requested": "2020-01-01",
    "end_date_requested": "2026-09-08",
    "end_date_written": "2026-09-04",
    "selection_rule": SELECTION_RULE,
    "hk_floor_choice": HK_FLOOR_NOTE,
    "hk_floor_ladder": ladder,
    "seeds": {"selection_shuffle": 20260911, "rng": "numpy.random.default_rng",
              "dev_acc_split": None},
    "thresholds": {
        "cn_a_min_avg_daily_turnover_cny": 50000000.0,
        "hk_min_avg_daily_turnover_hkd": 2000000.0,
        "listed_on_or_before": "2019-07-01",
        "cn_a_board_prefixes": ["600", "601", "603", "605", "688",
                                "000", "001", "002", "003", "300", "301"],
        "hk_code_range": "00001-09999",
        "market_cap_deciles": 10,
        "per_decile_cn_a_target": 28,
        "per_decile_hk_target": 12,
    },
    "quality_filter": {
        "min_rows": 510,
        "max_abs_daily_change_pct": 90.0,
        "max_ohlc_repair_error_pct": 5.0,
        "require_last_date": "==2026-09-04",
        "require_first_date": "<=2020-02-10",
        "drop_nonpositive_or_nan_price_rows": True,
        "replacement": "each rejected name replaced by the next name in the same market-cap decile's shuffled order",
    },
    "counts": {
        "total": len(recs),
        "cn_a": int((df.market == "CN-A").sum()),
        "hk": int((df.market == "HK").sum()),
        "eligible_cn_a": br["eligible"]["CN-A"],
        "eligible_hk": br["eligible"]["HK"],
        "dropped_quality": len(br["drops"]),
        "failed_fetch": sum(1 for d in br["drops"] if d["reason"].startswith("fetch_error")),
        "target_cn_a": 280,
        "target_hk": 120,
        "shortfall_vs_target": br["shortfall"],
    },
    "per_decile_counts": {f"{m}_d{d}": int(n) for (m, d), n
                          in df.groupby(["market", "market_cap_decile"]).size().items()},
    "decile_bounds_bn_local_ccy": {
        m: {int(d): [round(float(g.market_cap.min()) / 1e9, 4), round(float(g.market_cap.max()) / 1e9, 4)]
            for d, g in df[df.market == m].groupby("market_cap_decile")}
        for m in ("CN-A", "HK")},
    "turnover_summary_local_ccy_per_day": {
        m: {"min": float(g.avg_daily_turnover_3m.min()),
            "median": float(g.avg_daily_turnover_3m.median()),
            "max": float(g.avg_daily_turnover_3m.max())}
        for m, g in df.groupby("market")},
    "quality_drops": br["drops"],
    "quality_drop_reason_counts": dict(collections.Counter(
        d["reason"].split(" ")[0].split(":")[0] for d in br["drops"])),
    "overlap_check": {
        "vs_codes_305": len(codes & p305),
        "vs_codes_oos_230": len(codes & p230),
        "vs_pool_b_full_universe_dev_plus_acc": len(codes & pb),
        "vs_pool_b_dev": len(codes & pbdev),
        "method": ("The pool_b exclusion set is the file listing of data/pool_b/ (584 codes, verified "
                   "identical to the 584 codes in pool_b_manifest.json), which is the dev+acc union. "
                   "Zero overlap against that union implies zero overlap against pool_b_acc without "
                   "pool_b_acc.txt or data/pool_b_acc/ ever being opened."),
    },
    "bar_integrity_check": {
        "files": len(recs),
        "distinct_first_dates": ["2020-01-02"],
        "distinct_last_dates": ["2026-09-04"],
        "rows_cn_a": 1619,
        "rows_hk": 1642,
        "schema_and_dtype_match_vs_oos_data": True,
        "nan_or_nonpositive_prices": 0,
        "high_low_envelope_violations": 0,
        "date_monotonic_violations": 0,
        "rebuild_verification": ("The builder was run against four already-committed project files "
                                 "(oos_data 000025, 000421, 00012, 00014). Column order, dtypes, row "
                                 "counts, every date and every volume match exactly; adjusted prices "
                                 "match to a maximum relative difference of 2.0e-05, which is Yahoo's "
                                 "own revision of its dividend adjustment between 2026-09-09 and "
                                 "2026-09-11 at the 4-decimal rounding level."),
        "code_column": ("written verbatim as a zero-padded string (000009, 00012), identical to "
                        "oos_data and pool_b; the files are copied from the fetch cache rather than "
                        "round-tripped through pandas, which would strip the leading zeros. The "
                        "harness derives the code from the filename in any case."),
    },
    "reproducibility": {
        "snapshot": "stock_trading_advisor/oos_benchmark/pool_c_source/screener_raw.pkl",
        "scripts": ["pool_c_source/snapshot.py", "pool_c_source/select_pool_c.py",
                    "pool_c_source/bars.py", "pool_c_source/build_pool_c.py",
                    "pool_c_source/write_manifest.py"],
        "raw_build_result": "pool_c_source/build_result.json",
        "hk_floor_ladder": "pool_c_source/hk_floor_ladder.json",
    },
    "harness_integration": {
        "file": "research/harness.py",
        "one_line_change": ('add to resolve_pool(), before the final "return name":\n'
                            '    if name == "pool_c": return os.path.join(APP, "data", "pool_c")'),
        "usage": "python3 research/harness.py <candidate> --pool pool_c --workers 6",
        "note": ("NOT APPLIED. harness.py is deliberately left unedited so pool_c cannot be run by "
                 "accident; whoever spends the pool must add the line themselves, which makes spending "
                 "it a deliberate act."),
    },
    "caveats": [],
    "stocks": recs,
}

CAVEATS = [
"Data source differs from the 305 development pool. backtest_data was built with AkShare (Chinese vendor HFQ series); pool_c is built from Yahoo Finance (query1.finance.yahoo.com chart API) because AkShare and the Chinese data hosts (Sina, Eastmoney) are TLS-unreachable from this machine. Yahoo's adjusted-close series and the vendor HFQ series agree on total-return path but not bar for bar. Absolute metrics are therefore not byte-comparable with 305-pool metrics. pool_c is built by the identical procedure to oos_data and pool_b, so pool_c, pool_b and oos_data are mutually comparable; the 305-stock pool is not.",

"Survivorship bias is present and unavoidable with this source. The universe is a snapshot of CURRENTLY listed names taken 2026-09-11, so any stock that listed before 2019-07-01 and was delisted, privatised or suspended-to-death between 2020-01-01 and 2026-09-04 cannot appear. Backtest returns on pool_c are therefore biased upward relative to a true point-in-time universe. The 305, 230 and pool_b universes carry the same bias, so relative comparisons across pools are less affected than absolute levels. Concretely: pool_c flatters anything that buys after a fall.",

"The liquidity screen is CURRENT, not point-in-time. averageDailyVolume3Month and marketCap are as of 2026-09-11, i.e. at the END of the backtest window. A stock that was illiquid or tiny in 2020 and is liquid and large today is included; the reverse is excluded. This is a mild look-ahead on size/liquidity, but it uses no price-return, drawdown, volatility or trend input, and it is the same screen the 230-stock and pool_b universes used. The market-cap DECILE labels are likewise current-cap labels: a name that compounded 10x since 2020 sits in a higher decile than its 2020 self. Because the deciles are equal-count and every decile is sampled at the same rate, this reshuffles names between strata but does not select for winners.",

"The data-quality filter is technically path-dependent. 'no single-day absolute move above 90%' reads the realised price path, so it is not strictly performance-blind: it removes names with extreme single-session gaps (mostly reverse splits/consolidations and vendor artefacts, plus some genuine HK microcap collapses). This is the project's own established policy, applied identically to the 305, 230 and pool_b universes, and it was kept for comparability. It bites hardest on HK microcaps: 45 of the 102 HK rejections here are >90% single-session moves. Because this filter also fed the count table that chose the HK liquidity floor, the floor choice inherits the same hair of path-dependence.",

"HK is UNDER-FILLED and cap-tilted: 99 names, not 120. Deciles 1-5 hold 12 each, decile 6 holds 11, decile 7 holds 9, decile 8 holds 8, decile 9 holds 11, and decile 10 holds ZERO - every one of the 24 names in the smallest-cap HK decile failed the data-quality filter (reverse splits and internally inconsistent OHLC), and there was no replacement left in that decile. The HK half of pool_c is therefore tilted toward larger caps, more so than pool_b's HK half (which had 20/decile down to d8, 18 at d9 and 6 at d10). Any HK-specific conclusion drawn on pool_c is a conclusion about mid-and-large-cap HK.",

"The HK liquidity floor is 2,000,000 HKD/day - one fifth of the 10,000,000 HKD/day floor used for the 230-stock pool and two fifths of pool_b's 5,000,000 HKD/day. pool_c's HK names are on average less liquid than either. Combined with the project's own measurement that at 20bp per side the baseline per-stock median is already negative, a candidate whose edge depends on HK trading should be discounted here, not celebrated. The CN-A floor is unchanged at 50,000,000 CNY/day and the CN-A half is fully filled at 28 per decile, so CN-A conclusions carry no such discount.",

"The market mix differs from the pools it will be compared against. pool_c is 280 CN-A / 99 HK (73.9% / 26.1%); pool_b_dev is 200/92 (68.5%/31.5%) and oos_data is 150/80 (65.2%/34.8%). Any per-stock aggregate over the whole pool weights CN-A more heavily than in the pools where the candidate was developed. Report CN-A and HK separately, or reweight, before comparing a pool_c number to a pool_b_dev number.",

"Bars are written through 2026-09-04, not 2026-09-08. Data was fetched to 2026-09-08 and trimmed to 2026-09-04, the last session in backtest_data, oos_data and pool_b, so a strategy's forced close at data end lands on the same session in every pool.",

"Yahoo revises historical bars, so a later rebuild of the CSVs is not guaranteed to be byte-identical to this snapshot; the fixed identities in this manifest are the record of truth. Measured revision drift over two days on four control files was <= 2.0e-05 relative. The screener snapshot and all build scripts ARE retained under oos_benchmark/pool_c_source/, so the selection draw itself (which codes) is exactly reproducible.",

"Sector and industry are null for every name: the 2026-09-11 Yahoo screener response did not populate those fields. pool_b's manifest has the same gap. Pool C therefore carries no sector stratification and no sector metadata; the draw is stratified by market-cap decile only, and sector concentration inside a decile is whatever the shuffle produced.",

"Statistical power is finite and one-shot. 379 names is roughly 1.3x pool_b_dev's 292 and 1.6x oos_data's 230, so a paired per-stock test on pool_c has modestly more power than on either - but under the project's own measured null the two pools correlate +0.90/+0.95, and pool_c is drawn from the same current-listing snapshot as pool_b, so it is NOT an independent 8-cell replication. Treat it as ONE additional independent test, not four. The project's method rules still apply in full: report the share of stocks affected first, the paired median / sign test / paired mean over affected names together, portfolio CAGR / MDD / Sharpe / exposure-normalised Q5 capture, and a same-exposure two-sided random-timing control for anything that moves exposure.",

"pool_c has never been evaluated with any strategy. Nothing in this pool has been looked at, screened, ranked or sorted by any performance measure. No script in pool_c_source imports the harness, hooked.py, or anything from stock_trading_advisor/src; the only per-stock quantities ever computed were row count, first/last date, adjustment-repair error and largest single-session move, all of which are data-quality statistics. harness.py was deliberately NOT edited, so pool_c is not even addressable by name until someone adds the line themselves.",
]
manifest["caveats"] = CAVEATS

os.makedirs(SRC, exist_ok=True)
with open(os.path.join(BENCH, "pool_c_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

for fn, sel in (("codes_pool_c.txt", None), ("codes_pool_c_cn280.txt", "CN-A"), ("codes_pool_c_hk99.txt", "HK")):
    lines = sorted(r["code"] for r in recs if sel is None or r["market"] == sel)
    open(os.path.join(BENCH, fn), "w").write("\n".join(lines) + "\n")

for fn in ("snapshot.py", "select_pool_c.py", "bars.py", "build_pool_c.py",
           "write_manifest.py", "build_result.json", "hk_floor_ladder.json", "screener_raw.pkl"):
    shutil.copy(os.path.join(WORK, fn), os.path.join(SRC, fn))

print("manifest written:", os.path.join(BENCH, "pool_c_manifest.json"))
print("codes:", len(recs))
