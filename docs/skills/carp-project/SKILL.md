---
name: carp-project
description: 当需要了解本项目的专属上下文时使用，包括架构、文件布局、策略家族、报告解读、缓存规则、调度器位置，以及本仓库特有的实现约束和常见坑点。
---

# 项目上下文手册

本 skill 用于提供本项目的统一上下文，帮助后续 session 更快理解代码，不必每次都从零开始重新摸索。

当需要修改策略逻辑、README、调度脚本、报告分析逻辑，或者判断某个改动到底影响哪一层系统时，应优先使用本 skill，先建立一致的项目认知，再进入具体执行。

## 先建立正确心智模型

这个项目当前不是一个“技术指标演示工具”，也不是一个“单一 RSI 策略”。

它现在更准确地说是一个：
- 以 `new_strategy.py` 为核心的主线策略引擎
- 带多家族入场与多出口链
- 叠加动态放行、坏簇修复、extended hold、runner/reclaim 逻辑
- 通过正式离线报告评估保留版本的量化交易系统

后续 session 进入这个项目时，默认应假设：
- 主线策略都在 `stock_trading_advisor/src/new_strategy.py`
- 真正的保留基线以正式 `cache_backtest_report_*.txt` 为准
- 许多旧文档、旧术语、旧 README 片段可能落后于当前实现
- 策略不是靠一两个指标决定，而是靠“家族 + 分支 + 退出簇”共同作用

## 核心文件

- 当前主线策略：`stock_trading_advisor/src/new_strategy.py`
- 命令行入口：`stock_trading_advisor/main.py`
- 因子库：`stock_trading_advisor/src/factor_library.py`
- 数据获取：`stock_trading_advisor/src/data_fetcher.py`
- 信号格式化：`stock_trading_advisor/src/analyzer.py`
- K线图输出：`stock_trading_advisor/src/plotter.py`
- 微信通知：`stock_trading_advisor/src/wechat_notifier.py`
- 正式回测报告：`stock_trading_advisor/reports/cache_backtest_report_*.txt`
- 本地行情缓存：`stock_trading_advisor/data/cache/`
- 调度器：`stock_trading_advisor/scheduler.py`

其中 `stock_trading_advisor/src/factor_library.py` 应被视为本项目的因子兵工厂：
- 它不是一次性的研究脚本
- 而是应持续沉淀、持续扩充的复用因子库
- 如果从网络搜索、GitHub、论文或研究过程中确认了有效因子，优先更新到这里，而不是长期散落在临时脚本或局部试验代码里

## 主调用链

理解项目时，优先记住这条主调用链：

1. `stock_trading_advisor/main.py`
2. `analyze_stock(...)`
3. `RSITrendStrategy.analyze(df)`
4. `RSITrendStrategy.backtest(df)`
5. `RSITrendStrategy.get_trading_signals(df)`
6. `SignalAnalyzer` / 报告输出 / K线图输出

也就是说：
- `main.py` 负责 CLI、批量并发、离线报告入口
- `new_strategy.py` 负责真正的策略计算和交易序列构造
- `analyzer.py` 主要负责展示，不是交易逻辑核心
- `reports/cache_backtest_report_*.txt` 是结果留档，不是中间调试产物

## CLI 现实口径

- CLI 入口在 `stock_trading_advisor/main.py`
- `--report` 是离线正式回测入口
- `--chart-generation` 会在回测后输出买卖点标注图
- `--new-strategy` 现在是兼容保留参数，CLI 实际始终走 `RSITrendStrategy`，而当前主线策略已经内置在这个类里

不要被 `--new-strategy` 这个参数名误导，以为存在另一份独立主线代码。

## new_strategy.py 的真正结构

后续 session 读 `new_strategy.py` 时，建议按下面的层次理解，而不是从上到下死读几千行：

### 1. 配置层

- `RSITrendStrategy.__init__`
- 包含大量默认参数
- 很多实验性分支、坏簇修复、专属止损、extended hold、runner/reclaim 开关都在这里定义

如果需要判断某个策略行为是否“当前真的启用”，先看这里的默认配置值。

### 2. 预处理与指标层

- `_prepare_dataframe`
- `analyze`
- 这里会计算：
  - RSI
  - ATR
  - MA 系列
  - MACD
  - Aroon
  - 双通道相关特征
  - 线性回归斜率
  - 近似高周期特征
  - 各类过滤器与 regime 信号

### 3. 入场信号层

`analyze` 内部会构造大量家族信号，再合并为综合入场条件。

### 4. 核心执行层

真正最重要的不是 `analyze`，而是：
- `_build_position_series_with_divergence`

这是当前主线的核心执行引擎。  
它负责：
- 逐 bar 持仓状态推进
- 记录真实 `entry_reason`
- 记录真实 `exit_reason`
- 处理各家族专属止损、trailing、hard cap、extended hold、bounce exit、cooldown、做T接回
- 管理慢牛、W底、底背离、双通道、RSI continuation、golden cross 等专属分支

如果要定位“为什么这笔交易被这样卖掉”，最终大多都要回到这个函数。

### 5. 回测与交易合并层

- `backtest`
- 在父类回测基础上，补了“做T链合并”的逻辑
- 所以单笔交易统计并不总是简单的一进一出，而可能把多次做T合并成一笔

后续 session 分析交易明细时，要记住这一点。

## 当前工作模型

当前主线更准确地说是一个：
- 多家族交易系统
- 带动态路由和针对性修复
- 买卖逻辑逐步按家族分流

也就是说，系统中同时存在多种入场范式、持有范式和退出范式，不能再用“一个统一 RSI 逻辑”去理解全部实现。

## 主要策略家族

- `RSI金叉`
- `RSI多头延续`
- `RSI趋势买入`
- `RSI动量加速`
- `MA回踩因子`
- `慢牛回踩因子`
- `趋势再突破`
- `趋势跑者突破`
- `底背离信号`
- `W底形态`
- `折价区补仓`
- `双通道信号`
- `Aroon震荡入场`

这些家族不是孤立的研究脚本，而是当前主线策略中的真实组成部分。后续分析和修改时，应明确当前改动影响的是：
- 哪一个家族
- 哪一个坏簇
- 哪一条退出链
- 哪一批受损股票

## 家族理解方式

推荐按以下几类理解，而不是死记所有标签：

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

很多具体优化，本质上是在修这几类中的某一个坏簇，而不是在改整个系统。

## 动态路由与 profile 的现实状态

代码里存在一些看起来像“按画像路由”的结构，例如：
- `_compute_profile_features`
- `_route_profile_mode`
- `_profile_mode_overrides`
- `_apply_adaptive_profile_router`

但当前实现里，`_apply_adaptive_profile_router` 明确关闭了“按股票/按画像静态路由”，只保留买点级别的动态机制。

所以后续 session 不要默认以为当前系统已经完整启用了按股 profile router。  
真正活跃的，更多是：
- 家族级别的动态放行
- 坏簇级别的修补
- 针对持有和退出分支的条件化处理

## 项目约束

- 正式回测报告是保留基线的唯一可信记录。
- 除非任务明确要求，不要随意修改 `data/cache`。
- 临时研究产物不要当成正式项目资产。
- README 应描述当前主线，不应继续沿用早期 `Mixed Strategy / KDJ / 16日均线` 的旧说法。

## Cache 漂移坑点

这个项目里，`data/cache` 被“意外改写”最常见的原因，不是手工编辑 csv，而是误走了会自动刷新缓存的取数路径。

后续 session 必须明确区分：

- 安全路径：
  - `generate_cache_backtest_report(...)`
  - `python3 stock_trading_advisor/main.py --report ...`
  - 手工 `pd.read_csv(cache_file)` 后，通过 `analyze_stock(..., df_override=df_raw)` 做分析

这些路径的共同点是：
- 直接读取现有 `cache/*.csv`
- 不调用 `DataFetcher.get_k_data(...)`
- 因而不会触发缓存新鲜度检查和写回

- 危险路径：
  - 任何直接或间接调用 `DataFetcher.get_k_data(...)` 的研究脚本
  - `analyze_stock(..., df_override=None)` 这条默认取数链
  - `python3 stock_trading_advisor/main.py -b ...` 这种普通批量分析路径

根因在代码里非常明确：
- `DataFetcher.get_k_data(...)` 在非纯回测路径下，会调用 `_load_from_cache(..., check_freshness=True)`
- 如果缓存最后日期小于“最近交易日”，就把缓存判定为过期
- 然后走网络抓数，并调用 `_save_to_cache(...)` 覆盖原缓存

所以如果任务要求“cache 绝对不能动”，不要只理解成“不要手工改文件”；更要理解成：
- 不要走任何 `DataFetcher.get_k_data(...)` 路径
- 不要默认相信普通单股/批量分析命令是只读的
- 要显式走 `--report` 或 `df_override` 的纯缓存分析口径

## 项目中的“状态文件”

后续 session 容易误判哪些文件是正式资产，哪些只是运行状态。这里明确一下：

- `stock_trading_advisor/data/cache/`
  - 本地行情缓存
  - 很关键
  - 默认不要动

- `stock_trading_advisor/data/realtime_positions.json`
  - 调度器的实盘持仓追踪状态
  - 只影响 scheduler 提醒逻辑
  - 不应和离线正式回测混为一谈

## 常见坑点

- 这个项目里带有 `weekly` / `monthly` 命名的变量，不一定就是真实高周期数据；必须回到代码里确认它到底是：
  - 真实高周期序列
  - 还是日线近似值
- 当任务要求 cache 不可变时，`--report` 和 `df_override` 才是默认安全口径；不要在研究脚本里顺手调用 `DataFetcher().get_k_data(...)`
- 回测执行价口径固定为收盘价（close）；盘中高低价只能用于信号参考，不能作为实际成交价
- 不要只迁移一个完整子系统的一小段逻辑，而把它原本依赖的：
  - 路由
  - 放行
  - 持有保护
  丢在外面
- 未来函数检测里，“测试通过”不等于“覆盖有效”：
  - 如果本轮新增了买点、卖点或路由分支，但测试股票没有触发这些分支，结论是无效的
  - 必须把新增分支对应信号加入 `test_lookahead_bias_smart.py` 的对比列
  - 必须确认样本内至少有股票真实触发这些新增分支，否则要补样本再测

## 命名坑点

后续 session 特别容易被下面这些名字误导：

- `RSITrendStrategy`
  - 名字像“单一 RSI 策略”
  - 但实际上已经承载整个主线多家族系统

- `--new-strategy`
  - 名字像“新旧策略切换”
  - 实际只是兼容参数

- `lt_elder_weekly_macd`
  - 名字像真实周线 MACD
  - 但当前实现是基于日线 EMA65/EMA130 算出来的“周线近似值”
  - 不是真实 resample 周线

## 正式结果怎么看

后续 session 讨论“这版到底是不是更好”，应优先看：

- `avg return`
- `median`
- `tPF`
- losers 数量
- 被新系统打坏的高收益股票是否修回

不要只看单只股票，也不要只看 `avg` 一项。

正式报告文件：
- `stock_trading_advisor/reports/cache_backtest_report_*.txt`

这是保留基线的依据，也是 commit 摘要应该引用的来源。

## 修改代码时，先去哪里找

不同任务，优先定位点不同：

- 改入场家族 / 过滤器 / 路由
  - 先看 `analyze`
  - 再看 `entry_reasons` 的设置位置

- 改某一类交易为什么被卖掉
  - 先看 `_build_position_series_with_divergence`
  - 再看 `exit_reasons`

- 改做T、extended hold、runner 持有保护
  - 先看 `_build_position_series_with_divergence`
  - 再看 `backtest`

- 改 CLI、报告、批量并发
  - 先看 `main.py`

- 改定时提醒
  - 先看 `scheduler.py`
  - 再看 `wechat_notifier.py`

- 改研究用因子
  - 先看 `factor_library.py`

## 本 skill 的作用边界

本 skill 主要回答的是：
- 这个项目现在的主线是什么
- 哪些文件最关键
- 某段逻辑属于哪个家族或哪一层
- 当前项目有哪些硬约束和常见坑

本 skill 不负责执行流程控制。  
执行节奏、实验循环、回退纪律、持续推进规则，应交给 `carp-strategy-dev`。
