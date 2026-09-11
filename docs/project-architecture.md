# Carp Project Architecture

本文只描述 Carp 的稳定架构、运行路径和代码职责。具体 benchmark、报告路径和绩效数字以 `stock_trading_advisor/backtest_benchmark/` 为准；研发纪律以根目录 `AGENTS.md` 为准。

## 项目边界

Carp 是面向 A 股和港股的日线信号与离线回测项目，包含两条相互隔离的数据路径：

- 普通分析与实际信号：使用 `DataFetcher` 和 `data/cache/`，默认前复权 `qfq`，允许按配置更新行情。
- 正式离线回测：只读取 `data/backtest_data/*_hfq.csv`，使用固定后复权数据，不构造 `DataFetcher`，不访问网络。

策略只接收已经传入的个股日线，不自行加载指数、实时行情或其他外部状态。

## 核心模块

- `stock_trading_advisor/main.py`：CLI、单股分析、批量回测、汇总指标和报告编排。
- `stock_trading_advisor/src/new_strategy.py`：正式策略 `RSITrendStrategy` 及其买入、持有和退出状态。
- `stock_trading_advisor/src/strategy.py`：成交、手续费、逐日权益曲线和单股回测统计。
- `stock_trading_advisor/src/indicators.py`：通用技术指标。
- `stock_trading_advisor/src/analyzer.py`：信号、回测和交易明细的文本输出。
- `stock_trading_advisor/src/data_fetcher.py`：普通分析使用的数据源和缓存访问。
- `stock_trading_advisor/scheduler.py`：定时分析和模拟持仓调度。
- `stock_trading_advisor/tests/test_strategy_backtest.py`：回测资金曲线和交易核算回归测试。
- `stock_trading_advisor/tests/test_lookahead_bias_smart.py`：策略信号的逐日前缀未来数据检测。
- `research/`：不进入正式运行路径的候选隔离、诊断、完整前缀回放和组合研究工具。

## 普通分析链路

入口示例：

```bash
python3 stock_trading_advisor/main.py -s 000001
```

调用顺序：

1. `main.py` 读取运行配置。
2. `DataFetcher.get_k_data()` 从普通缓存或数据源取得日线。
3. `RSITrendStrategy.analyze()` 基于传入数据计算指标和持仓状态。
4. `get_latest_signal()` 生成最新信号。
5. 需要展示历史表现时，调用共享回测基座。
6. `SignalAnalyzer` 输出信号和交易信息。

普通分析允许更新 `data/cache/`；不得把更新结果自动写入正式回测数据目录。

## 离线回测链路

入口示例：

```bash
python3 stock_trading_advisor/main.py --report
python3 stock_trading_advisor/main.py --report -b 000001 300750 00700
```

调用顺序：

1. `generate_offline_backtest_report()` 从 `data/backtest_data/` 选择后复权 CSV。
2. worker 直接读取 CSV，并通过 `df_override` 传给 `analyze_stock()`。
3. 每只股票运行同一策略和同一回测基座。
4. worker 返回单股交易结果和逐日权益曲线。
5. 主进程汇总分布、风险、交易和等权组合指标。
6. 报告写入 `stock_trading_advisor/reports/`。

并发执行器和 worker 数只影响调度，不应改变策略结果。

## 数据目录

- `stock_trading_advisor/data/backtest_data/`：固定后复权回测数据，不受普通下载流程刷新。
- `stock_trading_advisor/data/oos_data/`：已揭示的外部回归数据，只供研究工具使用，不属于正式 benchmark 输入。
- `stock_trading_advisor/data/cache/`：普通行情缓存，可随在线取数更新。
- `stock_trading_advisor/oos_benchmark/`：已揭示 OOS230 的固定代码、数据来源、限制和重建工具。
- `stock_trading_advisor/backtest_benchmark/`：股票池、数据质量、口径和正式 benchmark 记录。
- `stock_trading_advisor/reports/`：运行产物，不是策略输入。
- `stock_trading_advisor/data/realtime_positions.json`：调度器使用的模拟持仓状态，不得影响离线回测。

## 修改定位

- 买点、卖点或持仓状态：`src/new_strategy.py`。
- 成交、手续费、权益曲线或单股统计：`src/strategy.py`。
- 离线股票选择、并发、汇总或报告：`main.py`。
- 输出文案：`src/analyzer.py` 或 `main.py`。
- 数据来源与普通缓存：`src/data_fetcher.py`。
- benchmark 身份和数字：`backtest_benchmark/README.md` 与 `manifest.json`。
- 策略禁用记录：`docs/strategy-blacklist.md`。

配置项、费率、策略阈值和绩效数字均以当前源码或对应 benchmark 记录为准，不在本文复制。代码事实与本文冲突时，以代码为准，并在同一改动中修正文档。
