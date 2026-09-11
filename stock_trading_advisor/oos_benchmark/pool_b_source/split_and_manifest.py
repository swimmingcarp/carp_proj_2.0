import json, os, collections
import numpy as np, pandas as pd

APP = "/home/jaden/carp/carp_proj_2.0/stock_trading_advisor"
BENCH = os.path.join(APP, "oos_benchmark")
POOL = os.path.join(APP, "data", "pool_b")
SELECT_SEED, SPLIT_SEED = 20260909, 20260910

res = json.load(open("/tmp/cleanpool/build_result.json"))
recs = {r["code"]: r for r in res["records"]}

groups = collections.defaultdict(list)
for r in res["records"]:
    groups[(r["market"], r["market_cap_decile"])].append(r["code"])

rng = np.random.default_rng(SPLIT_SEED)
dev, acc = [], []
for key in sorted(groups):                      # deterministic group order
    codes = sorted(groups[key])                 # deterministic within-group order
    idx = np.arange(len(codes)); rng.shuffle(idx)
    shuffled = [codes[i] for i in idx]
    half = len(shuffled) // 2
    dev += shuffled[:half]
    acc += shuffled[half:]
    for c in shuffled[:half]:
        recs[c]["half"] = "dev"
    for c in shuffled[half:]:
        recs[c]["half"] = "acc"
dev, acc = sorted(dev), sorted(acc)

p305 = {l.strip() for l in open(APP + "/backtest_benchmark/codes_305.txt") if l.strip()}
p230 = {l.strip() for l in open(APP + "/oos_benchmark/codes_oos_230.txt") if l.strip()}
assert not (set(dev) & set(acc))
assert not ((set(dev) | set(acc)) & (p305 | p230))
assert set(dev) | set(acc) == set(recs)

open(os.path.join(BENCH, "pool_b_dev.txt"), "w").write("\n".join(dev) + "\n")
open(os.path.join(BENCH, "pool_b_acc.txt"), "w").write("\n".join(acc) + "\n")

# mirror directories of symlinks so the harness can point at either half
for half, codes in (("dev", dev), ("acc", acc)):
    d = os.path.join(APP, "data", "pool_b_" + half)
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(d):
        os.unlink(os.path.join(d, name))
    for c in codes:
        os.symlink(os.path.join("..", "pool_b", "%s_hfq.csv" % c), os.path.join(d, "%s_hfq.csv" % c))

SELECTION_RULE = """Pool B universe selection (performance-blind; liquidity and data quality only).

Source snapshot: Yahoo Finance equity screener, region=cn (7435 rows) and region=hk (9292 rows before
  de-duplication, 9269 unique symbols), pulled 2026-09-09, sorted by intraday market cap descending.
  The raw snapshot is retained at stock_trading_advisor/oos_benchmark/pool_b_source/screener_raw.pkl,
  together with the four scripts that produced this pool (snapshot.py, select_pool_b.py,
  build_pool_b.py, split_and_manifest.py), so the draw can be regenerated exactly.

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
           CN-A >= 50,000,000 CNY ; HK >= 5,000,000 HKD,
         code NOT among the 305 development-pool codes and NOT among the 230 revealed-OOS codes.
  Eligible after all of the above: CN-A 2648, HK 334.

Stratified sampling:
  Eligible names are ranked by market cap and cut into 10 equal-count deciles (decile 1 = largest cap).
  Within each decile the names are shuffled with numpy.random.default_rng(20260909) and walked in that
  fixed shuffled order; the first N that pass the data-quality filter are kept
  (CN-A 40/decile = 400 target, HK 20/decile = 200 target). A name that fails the data-quality filter
  is replaced by the next name in the same decile's shuffled order, so the strata stay balanced.
  Nothing about historical return, drawdown, trend or strategy behaviour enters the rule.

Dev / acceptance split:
  Within every (market, market-cap decile) cell the kept codes are sorted, shuffled with
  numpy.random.default_rng(20260910), and cut in half: first half -> pool_b_dev, second half ->
  pool_b_acc. Every cell has an even count, so the two halves have identical market and decile
  composition (292 codes each)."""

CAVEATS = [
 "Data source differs from the 305 development pool. backtest_data was built with AkShare (Chinese "
 "vendor HFQ series); pool_b is rebuilt from Yahoo Finance (query1.finance.yahoo.com chart API) "
 "because AkShare and the Chinese data hosts (Sina, Eastmoney) are unreachable from this machine "
 "(TLS failures through a transparent proxy). Yahoo's adjusted-close series and the vendor HFQ series "
 "agree on total-return path but not bar for bar: Yahoo rounds differently, occasionally back-fills "
 "halted sessions, and its dividend/split handling can differ by a session. Absolute metrics are "
 "therefore not byte-comparable with 305-pool metrics; compare candidates WITHIN pool_b. pool_b is "
 "built exactly like the 230-stock oos_data pool, so pool_b and oos_data are mutually comparable.",

 "Survivorship bias is present and unavoidable with this source. The universe is a snapshot of "
 "CURRENTLY listed names taken 2026-09-09, so any stock that listed before 2019-07-01 and was "
 "delisted, privatised or suspended-to-death between 2020-01-01 and 2026-09-08 cannot appear. "
 "Backtest returns on pool_b are therefore biased upward relative to a true point-in-time universe. "
 "The 305 and 230 pools carry the same bias, so relative comparisons between pools are less affected "
 "than absolute levels.",

 "The liquidity screen is CURRENT, not point-in-time. averageDailyVolume3Month and marketCap are as "
 "of 2026-09-09, i.e. at the END of the backtest window. A stock that was illiquid or tiny in 2020 "
 "and is liquid and large today is included; the reverse is excluded. This is a mild look-ahead on "
 "size/liquidity (it correlates weakly with realised growth), but it uses no price-return, drawdown, "
 "volatility or trend input, and it is the same screen the 230-stock pool used. Yahoo does not expose "
 "a point-in-time universe from this machine, so a truly point-in-time screen is not available here. "
 "The market-cap DECILE labels are likewise current-cap labels: a name that compounded 10x since 2020 "
 "sits in a higher decile than its 2020 self. Because the deciles are equal-count and every decile is "
 "sampled at the same rate, this reshuffles names between strata but does not select for winners - "
 "big losers that stayed above the liquidity floor are still in the pool, in lower deciles.",

 "The data-quality filter is technically path-dependent. 'no single-day absolute move above 90%' "
 "reads the realised price path, so it is not strictly performance-blind: it removes names with "
 "extreme single-session gaps (mostly reverse splits/consolidations and vendor artefacts, plus a few "
 "genuine HK microcap collapses). This is the project's own established policy, applied identically "
 "to the 305 and 230 pools, and it was kept for comparability. It bites hardest on HK microcaps.",

 "HK strata 9 and 10 are under-filled. After excluding both existing pools only 334 HK names clear "
 "the liquidity floor, ~33 per decile, and in the two smallest-cap deciles most of them fail the "
 "data-quality filter (26 for >90% single-day moves, 35 for raw OHLC that is internally inconsistent "
 "by more than 5% after adjustment). Deciles 9 and 10 were exhausted, so no replacement was possible: "
 "decile 9 holds 18 names and decile 10 holds 6, against 20 in deciles 1-8. HK total is 184, not 200; "
 "pool_b's HK half is tilted toward larger caps. Both dev and acc inherit the identical tilt.",

 "The HK liquidity floor is 5,000,000 HKD/day, half the 10,000,000 HKD/day floor used for the "
 "230-stock pool. At 10m HKD only 261 HK names remained after excluding both existing pools, which "
 "leaves no replacement headroom for a 200-name stratified draw. 5m HKD/day is still roughly 29x the "
 "HK median. pool_b's HK names are therefore on average somewhat less liquid than the 230 pool's; the "
 "CN-A floor is unchanged at 50,000,000 CNY/day.",

 "Bars are written through 2026-09-04, not 2026-09-08. Data was fetched to 2026-09-08 and then "
 "trimmed to 2026-09-04, the last session in the project's own backtest_data and oos_data, so a "
 "strategy's forced close at data end lands on the same session in every pool.",

 "Yahoo revises historical bars, so a later rebuild of the CSVs is not guaranteed to be "
 "byte-identical to this snapshot; the fixed identities in this manifest are the record of truth. "
 "The screener snapshot and all four build scripts ARE retained under oos_benchmark/pool_b_source/, "
 "so the selection draw itself (which codes) is exactly reproducible.",

 "pool_b has never been evaluated with any strategy. Nothing in this pool has been looked at, "
 "screened, or ranked by any performance measure. pool_b_acc must stay untouched until a candidate "
 "is finalised on pool_b_dev; once it is read it is revealed, exactly as oos_data now is.",
]

manifest = {
 "version": "2026-09-09-pool-b-clean-universe",
 "status": "UNTOUCHED. No strategy has been run on any pool_b file. pool_b_dev is for development; "
           "pool_b_acc is a sealed acceptance half - reading it once reveals it permanently.",
 "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
 "purpose": "Selection-clean redevelopment universe: liquidity- and data-quality-selected stocks "
            "outside both the 305 development pool and the 230 revealed-OOS pool, drawn so that the "
            "development half and the acceptance half come from one unbiased distribution. Carries "
            "none of the hindsight labels (sideways_1y / drawdown_sample / growth / mixed) that tag "
            "150 of the 305 development names.",
 "data_dir": "stock_trading_advisor/data/pool_b",
 "dev_dir": "stock_trading_advisor/data/pool_b_dev (symlinks into pool_b)",
 "acc_dir": "stock_trading_advisor/data/pool_b_acc (symlinks into pool_b)",
 "file_pattern": "*_hfq.csv",
 "columns": ["date", "open", "close", "high", "low", "volume", "code"],
 "source": "Yahoo Finance chart API (query1.finance.yahoo.com/v8/finance/chart); universe listing "
           "from the Yahoo Finance equity screener (yfinance 1.7.0 EquityQuery region=cn / region=hk)",
 "adjust": "Yahoo-derived hfq-equivalent (后复权) series: hfq_t = raw_t * (adjclose_t/close_t) / "
           "(adjclose_0/close_0), applied to open/close/high/low and anchored so the first retained "
           "bar equals the raw price. Volume is left raw. High/low are then set to max/min of the "
           "four adjusted prices and the repair error is bounded by the quality filter.",
 "start_date_requested": "2020-01-01",
 "end_date_requested": "2026-09-08",
 "end_date_written": "2026-09-04",
 "selection_rule": SELECTION_RULE,
 "seeds": {"selection_shuffle": SELECT_SEED, "dev_acc_split": SPLIT_SEED,
           "rng": "numpy.random.default_rng"},
 "thresholds": {"cn_a_min_avg_daily_turnover_cny": 5e7, "hk_min_avg_daily_turnover_hkd": 5e6,
                "listed_on_or_before": "2019-07-01",
                "cn_a_board_prefixes": ["600","601","603","605","688","000","001","002","003","300","301"],
                "hk_code_range": "00001-09999", "market_cap_deciles": 10,
                "per_decile_cn_a_target": 40, "per_decile_hk_target": 20},
 "quality_filter": {"min_rows": 510, "max_abs_daily_change_pct": 90.0,
                    "max_ohlc_repair_error_pct": 5.0, "require_last_date": "==2026-09-04",
                    "require_first_date": "<=2020-02-10",
                    "drop_nonpositive_or_nan_price_rows": True,
                    "replacement": "each rejected name replaced by the next name in the same "
                                   "market-cap decile's shuffled order"},
 "counts": {"total": len(recs), "cn_a": sum(1 for r in recs.values() if r["market"] == "CN-A"),
            "hk": sum(1 for r in recs.values() if r["market"] == "HK"),
            "dev": len(dev), "acc": len(acc),
            "eligible_cn_a": 2648, "eligible_hk": 334,
            "dropped_quality": len(res["drops"]), "failed_fetch": 0,
            "shortfall_vs_target": {"CN-A": res["shortfall"]["CN-A"], "HK": res["shortfall"]["HK"]}},
 "per_decile_counts": {"%s_d%d" % (k[0], k[1]): len(v) for k, v in sorted(groups.items())},
 "quality_drops": res["drops"],
 "overlap_check": {"vs_codes_305": 0, "vs_codes_oos_230": 0, "dev_vs_acc": 0},
 "reproducibility": {
    "snapshot": "stock_trading_advisor/oos_benchmark/pool_b_source/screener_raw.pkl",
    "scripts": ["pool_b_source/snapshot.py", "pool_b_source/select_pool_b.py",
                "pool_b_source/build_pool_b.py", "pool_b_source/split_and_manifest.py"],
    "build_log": "pool_b_source/build.log",
    "raw_build_result": "pool_b_source/build_result.json"},
 "harness_integration": {
    "file": "research/harness.py",
    "one_line_change": 'add to resolve_pool(), before the final "return name":\n'
                       '    if name in ("pool_b", "pool_b_dev", "pool_b_acc"): '
                       'return os.path.join(APP, "data", name)',
    "usage": "python3 research/harness.py <candidate> --pool pool_b_dev"},
 "caveats": CAVEATS,
 "stocks": [recs[c] for c in sorted(recs)],
}
json.dump(manifest, open(os.path.join(BENCH, "pool_b_manifest.json"), "w"), indent=1, ensure_ascii=False)
print("dev", len(dev), "acc", len(acc), "manifest stocks", len(manifest["stocks"]))
