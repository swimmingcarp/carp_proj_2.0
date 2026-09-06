# Backtest Benchmark 250

- version: `2026-09-06-backtest-benchmark-250`
- data: `stock_trading_advisor/data/backtest_data/*_hfq.csv`
- count: `250` (`CN-A=206`, `HK=44`)
- historical 132 restored: `128`
- user-required names included: `09868`, `01797`, `06682`, `02556`
- ordinary signal adjust: `qfq`
- offline benchmark adjust: `hfq`
- observed last trading date: `2026-09-04`

## Selection Policy

- Restore usable names from the historical 132-stock benchmark where data remains usable.
- Force-include user-required names when they pass basic data quality checks: 09868, 01797, 06682, 02556.
- Prefer technology, growth, biotech/healthcare, robotics/automation, software/internet, EV/new-energy.
- Keep a minority of financial bluechips and legacy bluechips for market-style coverage.
- Exclude ST/delist-style names, missing recent data, non-positive prices, extreme daily jumps, and very short histories.
- Allow drawdown pressure samples, but exclude near-zero collapses beyond -99.0%.
- Use hfq data for offline benchmark; ordinary live/signal path uses qfq.
- Include sideways/drawdown samples so the benchmark is not only rising stocks.

## Quality Snapshot

- files checked: `250`
- min rows: `510`
- oldest end date: `2026-09-04`
- latest end date: `2026-09-04`
- max abs daily change: `88.13%`
- near-zero drawdown threshold: `-99.0%`
- OHLC-repaired files: `26`
- repaired high rows: `31`
- repaired low rows: `22`
- dropped non-positive rows: `1`

## Baseline Report

- report file: `stock_trading_advisor/reports/offline_backtest_report_20260906_232005.txt`
- success/failure: `250/0`
- avg return: `167.67%`
- median return: `59.20%`
- avg max drawdown: `-37.13%`
- avg win rate: `35.99%`
- total profit factor: `2.07`
- avg trades: `28.84`
- avg final capital: `26767.36`

## Lookahead Test

- command: `python3 stock_trading_advisor/tests/test_lookahead_bias_smart.py`
- result: `passed` (`6/6`, failures `0`)

## Theme Counts

- `advanced_manufacturing`: `17`
- `biotech_healthcare`: `18`
- `ev_new_energy`: `18`
- `financial_bluechip`: `16`
- `legacy_bluechip`: `9`
- `mixed_market`: `97`
- `robotics_ai_auto`: `9`
- `semiconductor_hardware`: `30`
- `software_internet`: `36`

## Path Counts

- `drawdown_sample`: `26`
- `growth`: `54`
- `mixed`: `46`
- `sideways_1y`: `124`

See `manifest.json` for the exact stock list and per-stock quality metrics.
