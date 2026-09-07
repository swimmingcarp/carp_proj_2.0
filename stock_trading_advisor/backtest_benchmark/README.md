# Backtest Benchmark 305

- version: `2026-09-07-backtest-benchmark-305`
- data: `stock_trading_advisor/data/backtest_data/*_hfq.csv`
- count: `305` (`CN-A=206`, `HK=99`)
- historical 132 restored: `128`
- ordinary signal adjust: `qfq`
- offline benchmark adjust: `hfq`
- observed last trading date: `2026-09-04`

## Cohorts

- `codes_250.txt`: strategy-development set (`OLD128 + NEW122`).
- `holdout_hk_35.txt`: first HK validation set, revealed during intermediate ablation; it is no longer treated as sealed evidence.
- `holdout_hk_20_round2.txt`: second HK validation set, selected before the structural-removal candidate. It has now also been inspected during parameter-neighborhood checks, so it is no longer sealed evidence.
- `codes_305.txt`: complete current benchmark in stable cohort order.

The round-2 holdout was selected only on business continuity, date coverage, OHLC quality, and style coverage. Strategy returns were not used for inclusion. `02252` was rejected because its latest row stopped at `2026-08-31`; `01877` replaced it before any strategy result was calculated. Because both holdout files have now informed research, future claims of out-of-sample improvement require a later time window or a newly frozen cohort.

## Selection Policy

- Prefer technology, growth, biotech/healthcare, robotics, software/internet, EV, and new-energy names.
- Retain financial, telecom, utility, industrial, and legacy bluechips for style coverage.
- Include sideways and severe-drawdown pressure samples instead of selecting only past winners.
- Exclude ST/delisting-style names, stale histories, non-positive prices, extreme daily jumps, and near-zero collapses beyond `-99%`.
- Include the user-required names `09868`, `01797`, `06682`, and `02556` when they pass data quality checks.
- Keep the official offline history separate from the ordinary download cache.

## Quality Snapshot

- files checked: `305`
- duplicate codes: `0`
- min rows: `510`
- oldest/latest end date: `2026-09-04`
- max absolute daily change: `88.13%`
- near-zero drawdown threshold: `-99.0%`
- OHLC-repaired files: `46`
- repaired high/low rows: `57 / 34`
- dropped non-positive rows: `2`

Repairs only clamp a rounded high or low to the same row's open/close envelope. Per-stock details and repair counts are recorded in `manifest.json`.

## Protected Baseline

- commit: `a8dd63a`
- report: `stock_trading_advisor/reports/offline_backtest_report_20260906_232005.txt`
- reproduction: `stock_trading_advisor/reports/offline_backtest_report_20260907_031037.txt`
- universe: `codes_250.txt`
- data/execution: fixed `hfq` CSV files, same-day `close`, forced close at data end
- success/failure: `250 / 0`
- average drawdown: `-37.13%` (legacy non-mark-to-market formula)
- profitable stocks: `195 / 250` (`78.0%`)
- median return: `59.20%`
- average per-stock win rate: `35.99%`
- return-percentage profit factor: `2.0714`
- average return/trades: `167.67% / 28.84`

The `2.0714` profit factor is the sum of positive per-trade returns divided by the absolute sum of negative per-trade returns. It is not an amount profit factor. The legacy drawdown understates holding-period risk; the same baseline signals measured with daily mark-to-market equity produce about `-46.72%` average drawdown on DEV250.

## Historical Stage-1 Benchmark

The failed RSI25/RSI100 + ATR + MA50 replacement and the later broad structural-removal candidate were both rejected. Git tag `strategy-benchmark-20260907-stage1` marks the accepted first-stage surgery benchmark and remains the reproducible historical comparator. It removes two active narrow-state blockers (`slow_bull_mature_switch_*` and `golden_cross_slow_switch_*`) plus the disabled online-EV, global dynamic, and runner routing implementations and all of their switches.

Formal benchmark reports:

- DEV250: `stock_trading_advisor/reports/offline_backtest_report_20260907_053200.txt` (`250 / 0`).
- ALL305: `stock_trading_advisor/reports/offline_backtest_report_20260907_053534.txt` (`305 / 0`).

On DEV250, the exact audit changes are: average return `167.673618% -> 167.771937%`, average drawdown `-37.134048% -> -37.131628%`, profitable stocks `78.0% -> 78.0%`, median return `59.201610% -> 59.201610%`, average per-stock win rate `35.985389% -> 36.002302%`, and same-formula profit factor `2.071365 -> 2.072099`. The improvement is small, not a regime-changing gain.

Across ALL305, average return improves by `0.119024` points, 10% trimmed mean by `0.158720`, profitable-stock share by `0.327869` points, average drawdown by `0.013032` points, average win rate by `0.049058` points, same-formula profit factor by `0.001069`, and amount profit factor by `0.000907`; the median is unchanged. OLD128 and NEW122 move in the same non-negative direction. The 55-stock sealed HK set also improves on all of those measures, including median `-0.568751% -> -0.477392%`.

The one disclosed regression is the broader 99-stock HK subgroup median, `7.261542% -> 6.247166%`; that subgroup's profitable-stock share, mean, trimmed mean, drawdown, win rate, and both profit factors improve. The changed trade paths are confined to 18 stocks; 13 improve and 5 worsen. This was a defensible but modest first-stage simplification, not proof that the remaining legacy strategy was sufficiently simple.

## Active Single-Strategy Baseline

Git tag `strategy-benchmark-20260907-single-strategy` marks the active baseline. It replaces the remaining strategy-family graph with one RSI/ATR strategy. `new_strategy.py` is reduced from `20,322` lines at the protected baseline to `740` lines. Strategy switches, per-family routers, online regime downloads, repair/re-entry branches, and orphan strategy modules are removed rather than left disabled.

The retained entry model has four auditable routes: RSI cross, trend continuation, controlled discount, and momentum recovery. They share RSI, ATR direction, Heikin-Ashi, MA/volume risk gates, a 180-day log-slope downtrend block, and a 100-day Hurst gate that applies only to discount entries. Exits are hard stop, confirmed trailing stop, MA trend exit, impulse protection, MA13 distribution exit, and a one-day delayed trend exit. Three coarse risk tiers remain because replacing them with one or two tiers degraded broad cohorts; they remain under audit and are not claimed to be optimal.

Formal reports:

- Strategy-freeze DEV250: `stock_trading_advisor/reports/offline_backtest_report_20260907_214359.txt` (`250 / 0`).
- Strategy-freeze ALL305: `stock_trading_advisor/reports/offline_backtest_report_20260907_214817.txt` (`305 / 0`).
- Corrected daily mark-to-market ALL305: `stock_trading_advisor/reports/offline_backtest_report_20260907_232138.txt` (`305 / 0`).

| Cohort | Avg return | Legacy avg drawdown | Profitable | Median return | Avg win rate | Return PF | Amount PF |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV250 | 145.28% | -41.18% | 72.80% | 44.67% | 36.84% | 1.9060 | 1.8317 |
| ALL305 | 130.79% | -40.89% | 71.48% | 38.04% | 36.40% | 1.8698 | 1.7848 |
| OLD128 | 167.57% | -38.34% | 75.00% | 44.68% | 37.20% | 2.0696 | 1.9269 |
| NEW122 | 121.90% | -44.17% | 70.49% | 44.30% | 36.46% | 1.7529 | 1.7243 |
| HK99 | 65.04% | -43.39% | 57.58% | 6.01% | 33.66% | 1.5862 | 1.4750 |
| Previously held-out 55 | 64.89% | -39.55% | 65.45% | 10.30% | 34.42% | 1.6736 | 1.4987 |

The strategy-freeze table preserves the historical non-mark-to-market drawdown for exact commit comparison. With the corrected engine, ALL305 average per-stock daily mark-to-market drawdown is `-50.04%` and median drawdown is `-49.18%`; return, profitable-stock share, median return, average win rate, both profit factors, trade counts, and final capital are unchanged across all 305 stocks.

Daily marked-to-market equal-weight portfolio results:

| Cohort | CAGR | Max drawdown | Sharpe | Median stock CAGR |
|---|---:|---:|---:|---:|
| DEV250 | 14.39% | -14.32% | 1.04 | 6.32% |
| ALL305 | 13.35% | -13.73% | 1.04 | 5.13% |
| OLD128 | 15.89% | -12.85% | 1.14 | 6.01% |
| NEW122 | 12.69% | -21.33% | 0.86 | 6.48% |
| HK99 | 7.80% | -25.66% | 0.65 | 0.88% |
| Previously held-out 55 | 7.78% | -22.12% | 0.63 | 1.48% |

This baseline is not a performance win over the protected DEV250 original baseline: average return, drawdown, profitable-stock share, median return, and return profit factor regress; only average win rate improves. It also trails stage 1 on most ALL305 metrics. It is established as the active simplification baseline for structural and cross-cohort reasons: OLD128 portfolio CAGR falls from stage 1's `19.82%` to `15.89%`, while NEW122 rises from `10.79%` to `12.69%` and the previously held-out 55 rise from `3.49%` to `7.78%`. The old/new CAGR gap therefore narrows from `9.03` to `3.20` points. That is evidence of reduced development-set dependence, but not independent proof of future performance.

Parameter-neighborhood checks did not select isolated peaks: long-slope thresholds `-0.002/-0.003/-0.004`, Hurst thresholds `0.60/0.65/0.70`, and RSI pairs `25/60`, `30/65`, `35/70` produced smooth risk/return trade-offs. The middle settings were retained without claiming they are globally optimal. A custom volume-quality score, a 250-day path filter, and simpler one/two-tier exits were rejected and removed after broad-cohort degradation.

## Lookahead Test

- baseline command: `python3 stock_trading_advisor/tests/test_lookahead_bias_smart.py --workers 6`
- baseline result: passed `6 / 6`, `1,628` tested prefixes, `0` failures
- candidate-specific result: all `18` affected stocks, `1,040` tested prefixes, `0` failures
- candidate scope: signal/intermediate columns, completed trade ledger, and open-position entry date/price at each tested cutoff

For the active single-strategy baseline, route attribution selected `600775` and `00512`. Together they actually trigger all four retained entry routes and all six retained exit routes. The current test passed `2 / 2`, `1,448` tested prefixes, with `0` signal, intermediate-state, open-position, or completed-trade-ledger differences:

```bash
python3 stock_trading_advisor/tests/test_lookahead_bias_smart.py --workers 2
```

See `manifest.json` for the exact stock list, themes, quality metrics, baseline fields, and development/holdout comparison.
