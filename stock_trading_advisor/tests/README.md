# 回测时点与未来函数测试

## 运行

```bash
python3 stock_trading_advisor/tests/test_lookahead_bias_smart.py
```

## 覆盖范围

`test_lookahead_bias_smart.py` 使用 `data/backtest_data/*_hfq.csv` 中的代表性 A/H 股：

- 先用完整历史计算参照结果。
- 完整覆盖历史中的买卖成交点；连续中间状态覆盖每个状态切换边界，并抽取等量、可重复的静默期样本。
- 每个测试点只向策略提供截至当日的前缀数据，并比较买点、卖点、持仓、原因、关键中间状态和截至该日的完整交易账本。
- 任意差异都视为可能使用未来数据，要求 0 差异。
- 展示用 `personality/pit_stage.py` 流水线单独测试，不与交易策略混为一体。

策略改动时先找出实际受影响股票，再选择能覆盖本次每条入场、退出和状态转换路径的最小 A/H 股集合。现有固定用例只能在确实受影响且覆盖路径时复用；新增路径必须先扩展 `signal_cols`、`comparison_cols` 和真实触发股票，不能直接沿用旧结果验收。

当前单一策略候选默认使用 `600775` 与 `00512`：二者都受完整策略替换影响，合并后实际触发 RSI cross、trend continuation、controlled discount、momentum recovery 四条入场路径，以及 hard stop、confirmed trailing、MA trend、impulse protection、MA13 distribution、delayed trend 六条退出路径。这个最小集合只对当前路径集合成立。

## 手工验收

1. 运行少量 A/H 股离线报告 smoke test。
2. 运行逐日前缀测试并确保 0 失败。
3. 运行全部 `backtest_data` 正式报告。
4. 对比 `median / trimmed mean / P25/P75 / 盈利股票占比 / 同公式tPF / 金额PF / equal-weight CAGR/DD/Sharpe / trades`，并分别检查 OLD128、NEW122 和验证集；算术均值只作右尾辅助观察。
5. 只有绩效满足保留标准且未来函数 0 失败，才允许建立新 baseline。

正式回测命令：

```bash
python3 stock_trading_advisor/main.py --report
```
