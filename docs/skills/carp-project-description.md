# Carp Project Description

本文件只介绍项目本身，不定义执行纪律。

如果你第一次进入这个仓库，读完本文件后，应该能回答：

- 这个项目现在到底是什么
- 主线策略在哪个文件
- 主要文件分别做什么
- 普通单股分析和正式离线报告分别走哪条调用链
- 正式结果看哪里
- cache / backtest_data / realtime 状态文件分别是什么

执行流程、验证纪律、commit 规范，交给 `docs/skills/carp-strategy-execution.md`。

## 项目定位

这个项目当前不是一个“技术指标演示工具”，也不是一个“单一 RSI 策略”。

它更准确地说是一个：

- 以 `stock_trading_advisor/src/new_strategy.py` 为核心的主线策略引擎
- 带多家族入场、多出口链和动态路由修复
- 通过正式离线报告评估保留版本的量化交易系统

## 正确认知

后续 session 默认应假设：

- 主线策略都在 `stock_trading_advisor/src/new_strategy.py`
- 真正的保留基线以正式 `offline_backtest_report_*.txt` 为准
- 许多旧文档、旧术语、旧 README 片段可能落后于当前实现
- 策略不是靠一两个指标决定，而是靠“家族 + 分支 + 退出簇”共同作用

## 核心文件与职责

- `stock_trading_advisor/src/new_strategy.py`
  - 当前主线策略
  - 负责指标计算、家族信号构造、持仓推进、真实 `entry_reason / exit_reason`、回测交易构造
- `stock_trading_advisor/src/strategy.py`
  - 主线策略复用的回测与手续费基座
  - `new_strategy.py` 会在这个基座上叠加做 T 交易合并等项目特有逻辑
- `stock_trading_advisor/main.py`
  - CLI 入口
  - 单股分析、批量回测、正式离线报告入口
- `stock_trading_advisor/src/factor_library.py`
  - 因子兵工厂
  - 适合沉淀可复用因子，不要把长期有效因子散落在临时代码里
- `stock_trading_advisor/src/data_validator.py`
  - 原始行情数据质量验证
  - 检查完整性、重复数据、异常涨跌幅、价格/成交量合理性
- `stock_trading_advisor/src/indicator_validator.py`
  - 技术指标结果验证
  - 检查 KDJ / MACD / MA 等指标是否异常
- `stock_trading_advisor/src/analyzer.py`
  - 信号展示与分析输出
  - 不是交易逻辑核心
- `stock_trading_advisor/src/plotter.py`
  - K 线图与买卖点标注输出
- `stock_trading_advisor/src/data_fetcher.py`
  - 数据获取与 cache 读写逻辑
- `stock_trading_advisor/src/market_hours.py`
  - 交易时间与最近交易日判断
  - cache 新鲜度判断依赖它
- `stock_trading_advisor/src/wechat_notifier.py`
  - 微信通知
- `stock_trading_advisor/scheduler.py`
  - 定时调度逻辑
- `stock_trading_advisor/reports/offline_backtest_report_*.txt`
  - 正式离线回测报告
  - 这是讨论正式结果的统一来源

## 主调用链

理解项目时，优先区分两条主路径，而不是只记一条：

### 1. 单股 / 普通分析路径

1. `stock_trading_advisor/main.py`
2. `analyze_stock(...)`
3. `DataFetcher.get_k_data(...)` 或 `df_override`
4. `RSITrendStrategy.analyze(df)`
5. `RSITrendStrategy.get_latest_signal(df_analyzed)`
6. `SignalAnalyzer`
7. 如需回测，再走 `RSITrendStrategy.backtest(df)` 和 `RSITrendStrategy.get_trading_signals(df)`

### 2. 正式离线报告路径

1. `stock_trading_advisor/main.py --report`
2. `generate_offline_backtest_report(...)`
3. `_run_offline_backtest_task(...)`
4. 直接读取 `data/backtest_data/*_hfq.csv`
5. 给策略注入 `offline_report_mode=True` / `allow_external_data=False`
6. `RSITrendStrategy.analyze(df)`
   - 策略内指数 regime / adaptive stop 所需的外部指数网络加载会被禁用
   - 若这类功能依赖外部指数数据，本次离线报告会跳过该指数信号
7. `RSITrendStrategy.backtest(df)`
8. 汇总并写入 `reports/offline_backtest_report_*.txt`

## CLI 现实口径

- `--report` 是离线正式回测入口
- `--report` 可以和 `-s / -b` 组合，用来限制报告股票范围
- `--chart-generation` 会在回测后输出买卖点标注图
- `--new-strategy` 现在只是兼容保留参数
- CLI 实际始终走 `RSITrendStrategy`
- `-b` 但不带 `--report` 时，走的是普通批量分析路径，不等于正式离线报告

不要被 `--new-strategy` 这个名字误导，以为还有另一份独立主线代码。

## `new_strategy.py` 的结构

后续阅读 `new_strategy.py` 时，建议按层次理解，而不是从上到下硬读全文。

### 1. 配置层

- `RSITrendStrategy.__init__`
- 默认参数、实验开关、专属止损、extended hold、runner/reclaim 开关基本都在这里

### 2. 预处理与指标层

- `_prepare_dataframe`
- `analyze`
- RSI / ATR / MA / MACD / Aroon / 双通道特征 / 线性回归斜率 / 近似高周期特征 / regime 信号都在这里生成

### 3. 入场信号层

- `analyze` 内部会构造大量家族信号，再合并为综合入场条件
- 不只包含早期的 RSI / 背离 / W 底 / 双通道
- 也已经接入 `main_wave`、`ZigZag`、`Elliott`、`wave_cycle`、`squeeze_breakout` 等结构信号

### 4. 核心执行层

- `_build_position_series_with_divergence`
- 这是当前主线最重要的执行引擎
- 负责逐 bar 持仓推进、真实入场退出原因、专属止损、bounce exit、cooldown、做 T 接回等

### 5. 回测与交易合并层

- `backtest`
- 在父类回测基础上补了“做 T 链合并”

## 当前主线的策略家族与结构信号

- 趋势追随类
  - `RSI金叉`
  - `RSI多头延续`
  - `RSI趋势买入`
  - `RSI动量加速`
  - `趋势再突破`
  - `趋势跑者突破`
- 趋势回踩 / 慢趋势类
  - `MA回踩因子`
  - `慢牛回踩因子`
- 反转 / 低吸类
  - `底背离信号`
  - `W底形态`
  - `折价区补仓`
  - `跳空回补`
- 震荡 / 通道类
  - `双通道信号`
  - `Aroon震荡入场`
- 结构识别类
  - `ZigZag fixed / ddb / dc / prob`
  - `艾略特波浪近似`
  - `main_wave_signal`

代码中还已经接入、但是否生效取决于默认配置的实验结构：

- `压缩突破（squeeze_breakout_*）`
- `波浪周期（wave_cycle_*）`

很多优化本质上是在修某个家族下的坏簇，而不是在改整个系统。

## 动态路由与画像的现实状态

当前主线已经不再保留“按股票一次分型后整段静态路由”的旧子系统。

现在真正还在代码里承担作用的是：

- `rolling_state_profile_*`
  - 用滚动窗口持续识别 `slow-trend / leader` 状态
- `continuation_entry_context_*`
  - 对 `standard / GC / momentum / dual` 做买点级别的独立自适应过滤
- `_route_profile_mode`
  - 仅作为 `profile_bar_router_enabled` 打开时的 bar 级模式映射 helper
- `online_family_router_*`
  - 用于 continuation 家族内的在线校准

因此现在讨论“自适应”时，不要再把旧的静态 `adaptive_profile_router` 当成现役模块；主线的真实自适应，已经转到滚动状态画像和买点上下文路由上。

## 状态文件与正式资产

- `stock_trading_advisor/data/cache/`
  - 普通本地行情缓存
  - 可能被取数路径刷新
  - 不作为正式离线回测数据源
- `stock_trading_advisor/data/backtest_data/`
  - 离线回测数据
  - 文件名固定为 `{股票代码}_hfq.csv`
  - 很关键
  - 正式报告只读这里
  - 默认不要动，除非用户明确要求刷新回测基准数据
- `stock_trading_advisor/backtest_benchmark/`
  - 离线回测基准股票池、构建说明和 manifest
  - 用来说明当前 `backtest_data` 的股票范围、复权口径和数据日期
- `stock_trading_advisor/data/realtime_positions.json`
  - 调度器的实盘持仓追踪状态
  - 只影响 scheduler 提醒逻辑
  - 不应和离线正式回测混为一谈
- `stock_trading_advisor/reports/offline_backtest_report_*.txt`
  - 正式报告归档
  - 是讨论正式结果和保留基线的统一来源

## Cache 漂移的项目事实

这个项目里，`data/cache` 被意外改写，最常见的原因不是手工编辑 csv，而是误走了会自动刷新的取数路径。正式离线回测应使用独立的 `data/backtest_data`，避免下载缓存的新数据混入基准数据。

根因在代码行为本身：

- `DataFetcher.get_k_data(...)` 在非纯回测路径下，会检查 cache 新鲜度
- 如果判定缓存不是最近交易日，就会抓网络数据
- 然后调用 `_save_to_cache(...)` 写回原缓存文件
- `--report` 这条正式报告路径会直接读取 `data/backtest_data/*_hfq.csv`，不会经过 `DataFetcher.get_k_data(...)`
- `--report` 会强制禁用策略内部外部指数下载，避免 market regime / adaptive stop 在离线报告里偷偷触发网络请求

复权口径必须区分清楚：

- 普通分析、调度器和实际买卖信号默认使用前复权（`qfq`）
- 正式离线回测基准数据使用后复权（`hfq`），降低长历史前复权价格失真对收益验证的影响
- 不要用普通 `data/cache` 中的 `qfq` 文件替代 `backtest_data` 的 `hfq` 基准文件

因此如果任务要求 cache / 回测数据不可变：

- 不要走任何 `DataFetcher.get_k_data(...)` 路径
- 不要默认相信普通单股 / 批量分析命令是只读的
- 要显式走 `--report` 或 `df_override` 的纯离线分析口径

具体执行纪律，交给 `docs/skills/carp-strategy-execution.md`。

## 常见坑点

- `weekly` / `monthly` 命名的变量，不一定就是真实高周期数据；必须回到代码里确认它到底是：
  - 真实高周期序列
  - 还是日线近似值
- 代码里“已经有某个家族 / 某组参数”不等于“默认已经启用”
  - 例如 `wave_cycle_enabled`、`squeeze_breakout_enabled` 都要回到默认配置确认
- 本项目回测执行价口径固定为 `close`
- 盘中高低价只能用于信号参考，不能直接当成交价
- 不要只迁移一个完整子系统的一小段逻辑，而把它依赖的路由、放行、持有保护丢在外面

## 命名坑点

- `RSITrendStrategy`
  - 名字像“单一 RSI 策略”
  - 实际上已经承载整个主线多家族系统
- `--new-strategy`
  - 名字像“新旧策略切换”
  - 实际只是兼容参数
- `lt_elder_weekly_macd`
  - 名字像真实周线 MACD
  - 但当前实现是基于日线 EMA65/EMA130 算出来的周线近似值

## 正式结果怎么看

后续讨论“这版到底是不是更好”，应优先看：

- `avg return`
- `median`
- `tPF`
- losers 数量
- 被新系统打坏的高收益股票是否修回

不要只看单只股票，也不要只看 `avg` 一项。

具体 baseline 保留流程、未来函数要求和 commit 纪律，交给 `docs/skills/carp-strategy-execution.md`。

## 修改代码时，先去哪里找

- 改入场家族 / 过滤器 / 路由
  - 先看 `analyze`
  - 再看 `entry_reasons` 的设置位置
- 改 `ZigZag / Elliott / wave_cycle / squeeze_breakout`
  - 先看 `analyze`
  - 再看 `_build_position_series_with_divergence`
- 改某类交易为什么被卖掉
  - 先看 `_build_position_series_with_divergence`
  - 再看 `exit_reasons`
- 改做 T、extended hold、runner 持有保护
  - 先看 `_build_position_series_with_divergence`
  - 再看 `backtest`
- 改数据质量 / 技术指标验证
  - 先看 `data_validator.py`
  - 再看 `indicator_validator.py`
- 改 CLI、报告、批量并发
  - 先看 `main.py`
- 改定时提醒
  - 先看 `scheduler.py`
  - 再看 `wechat_notifier.py`
- 改研究用因子
  - 先看 `factor_library.py`

## 边界

本文件主要回答的是：

- 这个项目现在是什么
- 哪些文件最关键
- 某段逻辑属于哪个家族或哪一层
- 当前项目有哪些硬约束和常见坑

执行节奏、实验循环、验证纪律、回退与 commit 规范，交给 `docs/skills/carp-strategy-execution.md`。
