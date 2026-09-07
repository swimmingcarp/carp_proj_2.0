# Carp Project Description

本文描述 Carp 当前代码事实。修改项目结构、主调用链、数据目录或报告输出前必须阅读。

## 项目定位

Carp 是面向 A 股和港股的日线信号与离线回测项目。commit `a8dd63a` 及 `reports/offline_backtest_report_20260906_232005.txt` 是不可替换的原始对照；Git tag `strategy-benchmark-20260907-stage1` 保留第一阶段删减基准，`strategy-benchmark-20260907-single-strategy` 标记当前单策略 baseline。

两种数据口径严格分开：

- 普通单股分析和实际信号：通过 `DataFetcher` 获取或读取 `data/cache`，默认前复权 `qfq`。
- 正式回测：只读取 `data/backtest_data/*_hfq.csv`，使用固定后复权历史，不访问网络。

## 核心文件

- `stock_trading_advisor/main.py`：CLI、单股分析、批量分析、离线报告和输出编排。
- `stock_trading_advisor/src/new_strategy.py`：唯一正式策略 `RSITrendStrategy`。
- `stock_trading_advisor/src/strategy.py`：手续费、当日收盘价成交和遗留回测统计。
- `stock_trading_advisor/src/indicators.py`：通用技术指标实现。
- `stock_trading_advisor/src/analyzer.py`：信号、回测和交易明细文本输出。
- `stock_trading_advisor/src/data_fetcher.py`：普通分析的在线取数与普通缓存。
- `stock_trading_advisor/src/personality/`：报告中的趋势时间线分析；不参与策略买卖决策。
- `stock_trading_advisor/tests/test_lookahead_bias_smart.py`：信号点与抽样静默期的前缀未来函数测试，以及展示用 PIT 流水线测试。
- `docs/strategy-blacklist.md`：禁止重新引入的策略和优化方式。

## 主调用链

### 普通单股分析

命令：

```bash
python3 stock_trading_advisor/main.py -s 000001
```

执行顺序：

1. `main.py` 读取 `config/config.yaml`。
2. `DataFetcher.get_k_data()` 使用普通缓存或数据源取得默认 `qfq` 日线。
3. `RSITrendStrategy.analyze()` 只基于传入的个股日线生成指标和完整持仓状态。
4. `get_latest_signal()` 输出最新信号。
5. 默认运行 `StrategyBase.backtest()`；`--no-backtest` 可跳过展示性回测。
6. `SignalAnalyzer` 输出信号、绩效和尾盘近似成交明细。

普通分析允许 `DataFetcher` 按既有配置联网取得个股数据；策略本身不联网，也不加载指数或其他外部状态。

### 正式离线报告

命令：

```bash
python3 stock_trading_advisor/main.py --report
python3 stock_trading_advisor/main.py --report -b 000001 300750 00700
```

执行顺序：

1. `generate_offline_backtest_report()` 只扫描 `data/backtest_data/*_hfq.csv`。
2. 文件名决定股票代码；指定 `-s/-b` 时仍必须在该目录命中已有文件。
3. worker 通过 `df_override` 把 CSV 直接交给 `analyze_stock()`，不会构造 `DataFetcher`。
4. 每只股票运行同一策略和同一回测基座。
5. 正式报告汇总五项主指标，并同时输出收益分布、金额盈亏比、总交易胜率、赢家集中度，以及每日盯市等权组合的 CAGR、最大回撤和 Sharpe。
6. 报告写入 `stock_trading_advisor/reports/offline_backtest_report_YYYYMMDD_HHMMSS.txt`。

进程数可由 `STOCK_ADVISOR_REPORT_WORKERS` 指定，执行器可由 `STOCK_ADVISOR_REPORT_EXECUTOR=process|thread` 指定。这些只影响调度，不影响策略结果。

## 当前策略

`RSITrendStrategy` 是唯一正式策略。它用 RSI `30/65`、ATR20 x 3 趋势方向和 Heikin-Ashi 构造趋势延续、折价和动量入场，再用成交量衰减、ATR 扩张、M 顶近似、MA60 距离、上涨连击、长期对数斜率和 Hurst 持续性做有限质量过滤。Hurst 只阻止高持续性价格路径中的折价抄底，不参与其他买点路由。

持仓状态机只保留三类风险档位、硬止损、确认式 trailing、MA 趋势退出、脉冲趋势保护、单日大跌时延迟一日确认，以及放量阴线且偏离 MA13 的获利退出。divergence、W 底、ZigZag/Elliott、双通道、慢牛/MA60/banklike 回踩、extended hold、gap、波浪、家族冷却/修复/再入场和在线 regime/adaptive stop 均已从源码删除，不保留关闭代码或配置开关。

该重构不是在 DEV250 上全面提高历史指标。它接受开发集指标下降，以换取大幅减少规则、状态、参数和旁路，并在此前留出的 55 股集合上观察到收益分布与盈利覆盖改善。该集合已经参与研究判断，不再是密封样本，因此改善只能作为泛化线索，不能作为独立样本外证明。具体取舍和禁止恢复的方法记录在 `backtest_benchmark/README.md` 与 `docs/strategy-blacklist.md`。

## 回测语义

- 第 `t` 日信号以第 `t` 日 `close` 成交，作为用户尾盘最后一分钟成交的日线近似。
- 数据结束时遗留引擎强制按最后收盘价平仓。
- 基线报告的资金曲线在持仓期间没有逐日按市值更新，因此其 `-37.13%` 平均回撤是历史兼容口径，不是真实每日盯市回撤。
- 正式报告保留历史回撤用于和原始报告同口径比较，同时另列每日盯市等权组合风险；两种口径不得混称。
- 基线报告的总盈亏比是“全部交易正收益率之和 / 全部交易负收益率绝对值之和”，不是金额盈亏比。
- 策略比较同时检查中位数、截尾均值、盈利股票占比、赢家集中度、金额盈亏比和日等权组合，算术平均收益不能单独验收。

## 数据与状态目录

- `data/backtest_data/`：正式、固定、后复权回测数据。不得被普通下载流程刷新。
- `data/cache/`：普通行情缓存，可随在线取数更新，不能作为正式回测输入。
- `backtest_benchmark/`：股票池清单、质量快照和已建立的基准记录。
- `reports/`：运行产物，不是策略输入。
- `data/realtime_positions.json`：实时持仓状态，不得影响离线回测。

## 配置边界

`config/config.yaml` 只保留数据源、分析、回测和报告运行配置，`scheduler_config.yaml` 只保留调度、通知、分析展示和数据源配置。策略没有外部参数、家族开关或离线运行标记；研究候选必须改独立副本并完成验证，不能把配置文件当扫参入口。

## 修改定位

- 改买点、卖点、持有状态：`src/new_strategy.py`。
- 改成交时点、手续费、权益曲线或指标统计：`src/strategy.py`。
- 改离线股票选择、并发、汇总或报告文件：`main.py`。
- 改显示文案：`src/analyzer.py` 和 `main.py`。
- 改数据来源和缓存：`src/data_fetcher.py`，并检查普通与正式回测边界。
- 改策略前先读 `docs/strategy-blacklist.md`。仍在当前实现中的黑名单分支不得扩展或换名重加，只能通过独立候选做完整路径删除或替换验证。

代码事实和本文冲突时，以代码为准，并在同一改动中修正文档。
