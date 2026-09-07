# Stock Trading Advisor

基于技术指标的股票交易策略分析系统，提供买卖信号提示和策略回测功能。

## AI / Agent Usage

如果使用 Codex 或其他 agent 处理本仓库任务，先读根目录的 `AGENT.md`。

## 🔔 定时提醒服务（推荐使用）

系统已配置 **systemd 服务**，支持开机自启动和崩溃自动重启：

```bash
cd stock_trading_advisor

# 查看服务状态
./manage_scheduler.sh status

# 查看日志
./manage_scheduler.sh logs

# 重启服务（修改配置后）
./manage_scheduler.sh restart

# 停止/启动服务
./manage_scheduler.sh stop
./manage_scheduler.sh start
```

**主要特性：**
- ✅ **开机自动启动** - 系统重启后无需手动启动
- ✅ **崩溃自动重启** - 进程异常退出后10秒自动重启
- ✅ **定时自动执行** - 默认每天14:57执行分析
- ✅ **微信消息通知** - 支持企业微信机器人推送

**说明：** systemd 的使用方式已经合并到本 README，直接按下面的 `manage_scheduler.sh` 示例操作即可。

## 功能特性

✨ **核心功能**
- 📈 **固定策略对照**：commit `a8dd63a` 及指定报告永久保留为原始 baseline
- 🔍 **候选隔离验证**：策略替换先在独立环境验证，未通过时不得覆盖正式运行路径
- 💹 **单股分析 + 全量离线回测**：既支持单只股票分析，也支持对离线回测股票池生成完整批量报告
- 🧾 **固定离线基准**：以 `codes_250.txt`、后复权数据和指定基线报告做同口径比较
- 🖼️ **K 线图与买卖点标注**：支持生成带买卖点的图表用于人工复盘
- 📱 **批量分析与定时提醒**：支持固定时间自动运行，并通过企业微信推送结果
- ⚙️ **稳定运行配置**：`config.yaml` 管理数据源、回测、报告和运行方式；策略不再暴露扫参开关

⚡ **性能优势**
- 🚀 **离线回测数据优先**：批量回测直接复用 `data/backtest_data`，适合高频研究和反复验证
- ⚙️ **高并发离线报告**：`--report` 模式默认支持多进程并发，worker 数按“股票数”和“CPU 核心数”自动取较小值
- 📐 **向量化指标计算**：核心指标计算采用向量化实现，减少逐 bar 回测的额外开销
- 🔌 **多数据源接入**：支持 `AKShare`、`Tushare`、`yfinance`

## 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/swimmingcarp/carp_proj_2.0.git
cd carp_proj_2.0
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r stock_trading_advisor/requirements.txt
```

**提示**：如果安装较慢，可使用国内镜像：
```bash
pip install -r stock_trading_advisor/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 配置参数（可选）
编辑 `stock_trading_advisor/config/config.yaml` 修改数据源、回测资金和报告运行参数

### 5. 运行程序
```bash
# 分析单只股票
python3 stock_trading_advisor/main.py -s 000001

# 或者使用完整命令
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 001279
```

## 使用方法

### 单只股票分析

```bash
# 激活虚拟环境并分析平安银行（000001）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001

# 分析贵州茅台（600519）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 600519

# 不显示回测结果
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001 --no-backtest

# 使用当前主线策略分析单只股票
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 300293

# 对离线回测数据中的所有股票进行回测并输出报告
# 默认就是高并发静默模式：多进程 + 自动按 min(股票数, CPU核心数) 分配 worker + 不逐只刷屏
source venv/bin/activate && python3 stock_trading_advisor/main.py --report

# 显式指定并发（示例：使用 16 个进程）
source venv/bin/activate && STOCK_ADVISOR_REPORT_EXECUTOR=process STOCK_ADVISOR_REPORT_WORKERS=16 python3 stock_trading_advisor/main.py --report

# 如果需要，也可以切到多线程执行器
source venv/bin/activate && STOCK_ADVISOR_REPORT_EXECUTOR=thread STOCK_ADVISOR_REPORT_WORKERS=16 python3 stock_trading_advisor/main.py --report

# 只对指定股票生成离线报告（需已有离线回测数据）
source venv/bin/activate && python3 stock_trading_advisor/main.py --report -b 300293 300274 300750 605117

# 生成K线图并标注买卖点
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001 --chart-generation
# 图片将保存到 reports/kline_000001.png
```

使用 `--report` 时，系统会将完整的批量回测明细保存到 `stock_trading_advisor/reports/offline_backtest_report_YYYYMMDD_HHMMSS.txt`（按时间戳命名），每只股票都会包含最新价格、历史交易对收益表以及交易统计，方便留档和复盘。

`--report` 为纯离线回测口径：个股数据只读取 `stock_trading_advisor/data/backtest_data/*_hfq.csv`。当前策略只使用传入的个股日线，不存在指数或其他外部行情加载路径。普通单股分析和实际买卖信号默认使用前复权（`qfq`），正式离线回测基准使用后复权（`hfq`）。`stock_trading_advisor/data/cache/` 只作为普通下载缓存使用，避免最新下载数据混入正式回测数据。

当前固定基线是 `codes_250.txt`；额外 55 只港股仅作为候选验证集，`codes_305.txt` 是完整审计集合。股票池划分、质量审计和基准报告指标见 `stock_trading_advisor/backtest_benchmark/`。

`--report` 现在默认使用高并发静默模式：默认执行器为多进程，默认 worker 数为 `min(目标股票数量, CPU 核心数)`。系统会将 `OMP/OPENBLAS/MKL/NUMEXPR` 线程压到 `1`，避免每个子进程再额外开线程导致过度并行。可通过环境变量 `STOCK_ADVISOR_REPORT_EXECUTOR`（`process`/`thread`）和 `STOCK_ADVISOR_REPORT_WORKERS` 覆写执行器与并发数；如需恢复逐只刷屏，可把 `report.verbose` 改回 `true`。

**输出示例**：
```
============================================================
股票代码: 000001
股票名称: 平安银行
当前价格: 12.35
============================================================

📈 【买入信号】 [████░]
信号强度: 4/5

理由: RSI与ATR多头共振

--- 技术指标 ---
RSI快线: 48.50, RSI慢线: 42.30, 差值: 6.20
ATR趋势: 多头
退出参考价: 由当前策略状态计算
============================================================

📊 回测结果
============================================================
初始资金: ¥10,000.00
最终权益: ¥12,456.78
总收益率: 24.57%
最大回撤: -8.23%
夏普比率: 1.45

--- 交易统计 ---
交易次数: 15
胜率: 60.00%
交易天数: 365
============================================================

💡 投资建议
------------------------------------------------------------
⚠️ 谨慎买入
  - 有买入信号，但强度一般
  - 建议: 小仓位试探，观察后续走势

基于历史回测:
  ✅ 该策略历史表现良好

⚠️ 风险提示:
  - 本分析仅供参考，不构成投资建议
  - 股市有风险，投资需谨慎
  - 请根据自身风险承受能力做出决策
------------------------------------------------------------
```

### 批量分析

```bash
# 分析多只股票
source venv/bin/activate && python3 stock_trading_advisor/main.py -b 000001 000002 600519 601318

# 使用自定义配置
source venv/bin/activate && python3 stock_trading_advisor/main.py -b 000001 000002 -c stock_trading_advisor/config/your_config.yaml
```


### 📅 定时提醒

**系统已配置 systemd 服务，支持自动化运行和管理**

#### 🚀 快速管理

```bash
cd stock_trading_advisor

# 查看服务状态和进程信息
./manage_scheduler.sh status

# 查看最近日志
./manage_scheduler.sh logs

# 重启服务（修改配置后必须执行）
./manage_scheduler.sh restart
```

#### ⚙️ 配置说明

1. **监控股票列表** - 编辑 `config/watch_list.txt`
```
300293  # 蓝英装备
002920  # 德赛西威
300757  # 罗博特科
```

2. **定时时间设置** - 编辑 `config/scheduler_config.yaml`
```yaml
schedule:
  run_times:
    - "14:57"  # 下午2:57（收盘前3分钟）
    # - "09:35"  # 可添加多个时间点
```

3. **微信通知配置** - 编辑 `config/scheduler_config.yaml`
```yaml
wechat:
  work_wechat:
    enabled: true
    webhook_url: 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY'
```

**修改配置后必须重启服务：**
```bash
./manage_scheduler.sh restart
```

#### 📊 输出示例

```
======================================================================
📊 股票交易提醒 - 2025-10-21 14:57:00
======================================================================

📈 监控股票总数: 4 只
   🟢 买入: 2 只  |  🔴 卖出: 0 只  |  ⚪ 持有: 2 只

======================================================================
🟢 买入信号 (2只)
======================================================================
  【300293】 蓝英装备
     操作: 买入  |  当前价: ¥23.37
     信号: MACD金叉
     RSI: 35.23  |  MACD: -0.4177
======================================================================
```

所有提醒保存在 `reports/trading_alerts.txt`

#### ✨ 服务特性

- **开机自启动** - 系统重启后自动运行，无需手动启动
- **崩溃自动恢复** - 进程异常退出后10秒内自动重启
- **完整日志记录** - 所有运行日志保存在 `logs/` 目录
- **状态监控** - 使用 `./manage_scheduler.sh status` 随时查看运行状态

#### 🔧 手动测试（可选）

如果需要立即执行一次分析（不等待定时）：

```bash
cd stock_trading_advisor

# 1. 临时停止服务
./manage_scheduler.sh stop

# 2. 手动运行一次
python3 scheduler.py --run-now

# 3. 重新启动服务
./manage_scheduler.sh start
```

#### 📖 详细文档

- [快速上手指南](stock_trading_advisor/WechatQuickStart.txt)

## 项目结构

```bash
stock_trading_advisor/
├── src/
│   ├── new_strategy.py        # 唯一的 RSI/ATR 趋势策略
│   ├── strategy.py            # 基础策略与回测接口
│   ├── indicators.py          # 指标计算
│   ├── analyzer.py            # 单股分析与结果整合
│   ├── plotter.py             # K线图与买卖点标注
│   ├── data_fetcher.py        # 数据获取
│   ├── data_validator.py      # 数据校验
│   ├── market_hours.py        # 交易时间检测
│   ├── wechat_notifier.py     # 微信通知
├── config/
│   ├── config.yaml            # 主配置
│   ├── scheduler_config.yaml  # 定时任务配置
│   ├── watch_list.txt         # 监控股票列表
│   ├── cn_stock_names.txt     # A股名称映射
│   └── hk_stock_names.txt     # 港股名称映射
├── data/
│   ├── cache/                 # 普通本地行情缓存，可被取数路径刷新
│   ├── backtest_data/         # 正式离线回测数据（*_hfq.csv）
│   ├── market_breadth.csv     # 市场宽度数据
│   ├── realtime_positions.json
├── backtest_benchmark/        # 离线回测基准股票池与说明
├── reports/                   # 回测报告、研究输出、图表
├── logs/                      # 运行日志
├── tests/
│   ├── README.md
│   ├── test_strategy_backtest.py      # 回测资金曲线与交易核算回归测试
│   └── test_lookahead_bias_smart.py  # 逐日前缀未来函数检测
├── main.py                    # CLI 入口
├── scheduler.py               # 定时调度器
├── manage_scheduler.sh        # systemd 管理脚本
├── run_scheduler.sh           # 调度启动脚本
├── WechatQuickStart.txt       # 微信推送快速说明
├── requirements.txt           # 依赖列表
└── README.md                  # 项目文档
```

## 策略说明

### Current Main Strategy（当前主线策略）

commit `a8dd63a` 和 `offline_backtest_report_20260906_232005.txt` 永久保留为原始对照；Git tag `strategy-benchmark-20260907-stage1` 保留第一阶段手术结果。当前主线是在二者之上完成的单策略重构，不能用它覆盖原始 baseline 的身份。

#### 当前状态

- 只保留一个 RSI/ATR 趋势策略，不再并联 divergence、W 底、双通道、慢牛、MA60/banklike、ZigZag/wave、extended hold 或再入场家族。
- 买点由 RSI `30/65` 趋势、ATR 方向、Heikin-Ashi、成交量和追高约束共同形成；折价买点在高持续性趋势中被阻止。
- 卖点只保留统一持仓状态机中的硬止损、确认式 trailing、趋势转空、趋势保护和一个放量偏离退出。
- 历史策略开关、家族专用配置、在线 regime/adaptive stop 和旧 CLI 兼容入口均已删除。
- 当前单策略 baseline tag 为 `strategy-benchmark-20260907-single-strategy`；策略冻结报告为 `offline_backtest_report_20260907_214359.txt`（DEV250）和 `offline_backtest_report_20260907_214817.txt`（ALL305），旧费用口径下的逐日盯市 ALL305 校正报告为 `offline_backtest_report_20260907_232138.txt`。
- 2026-09-08 起回测费用模型改为纯比例、与账户规模无关的当前费率：A 股佣金 `0.015%`（不设最低 5 元）、卖出印花税 `0.05%`；港股每边印花税 `0.1%` 加佣金与交易征费、交易费、会财局征费、交收费合计 `0.045%`。买卖点、交易次数与旧报告 305/305 完全一致，只有费用相关字段变化。当前费用口径的 ALL305 报告为 `offline_backtest_report_20260908_012933.txt`。
- 当前 ALL305 五项主指标为平均逐日盯市回撤 `-48.89%`、盈利股票占比 `73.77%`、收益中位数 `43.77%`、平均胜率 `37.67%`、同公式盈亏比 `1.94`；平均收益 `138.76%` 只作右尾辅助指标。旧费用口径下同一信号为 `-50.04% / 71.48% / 38.04% / 36.40% / 1.87 / 130.79%`，差异全部来自费用口径修正，不是策略改善。
- 该 baseline 除平均胜率外弱于受保护 DEV250 原始基线，建立理由是源码由 `20,322` 行降至 `740` 行，以及 OLD128/NEW122 的等权组合 CAGR 差距由 `9.03` 收窄至 `3.20` 个百分点（旧费用口径；当前费用口径下为 `16.22% / 13.13%`，差距 `3.09`）；这不是“历史收益全面提升”。

#### 核心思想

- 正式结果先与固定基线同口径比较，再检查 OLD128、NEW122、A/H 股和新增 55 股集合。
- 实验候选不得直接覆盖正式策略；失败候选必须完整撤回。
- 假设尾盘最后一分钟形成信号并成交，日线回测统一用当日收盘价近似该时点价格。
- 该口径不表示可以在收盘后使用已知完整日线结果回到收盘价成交；分钟内误差需要分钟数据另行验证。

#### 退出与风控

- 风控参数保留三类入场风险档位；统一风控和粗两级风控在最终组合的 DEV250、ALL305 与新增 55 股集合均出现一致退步，因此本轮没有为了少几个参数强行替换。
- 这些档位是待持续验证的有限复杂度，不是已证明的最优参数；新研究不得继续增加家族、路由、修复或强制再入场补丁。

## 配置参数

主配置在 [stock_trading_advisor/config/config.yaml](stock_trading_advisor/config/config.yaml)。

- `report`：离线批量回测配置
- `backtest`：初始资金和展示性日期配置
- `data_source`：数据源配置
- `analysis`：图表和回测展示配置

策略阈值固定在源码中，配置文件不提供策略开关或参数扫描入口。

一个更贴近当前使用方式的示例：

```yaml
data_source:
  adjust: qfq
  provider: akshare
  cache_enabled: true

report:
  verbose: false
```

## 注意事项

⚠️ **数据源限制**
- AKShare：免费，但有访问频率限制
- Tushare：需要积分，不同接口积分要求不同
- 建议合理控制请求频率，避免被限制

## 常见问题

### 1. 安装依赖失败
```bash
# 如果 pip 安装较慢，可使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Numba 编译慢
首次运行时 Numba 会编译函数，可能需要几秒钟。后续运行会使用缓存，速度很快。

### 3. 股票代码格式
- A 股：直接使用 6 位代码（如 `000001`）
- 自动识别：6 开头为上海，其他为深圳

### 4. 日志查看
- 主程序日志：`logs/trading.log`
- 调度器日志：`logs/scheduler.log`

### 5. 定时提醒相关

#### 如何查看服务状态？
```bash
cd stock_trading_advisor
./manage_scheduler.sh status
```

#### 修改配置后没有生效？
**原因**：修改配置后没有重启服务。

**解决方法**：
```bash
cd stock_trading_advisor
./manage_scheduler.sh restart
```

#### 如何查看执行日志？
```bash
cd stock_trading_advisor
./manage_scheduler.sh logs
```

#### 如何临时禁用调度器？
```bash
cd stock_trading_advisor
./manage_scheduler.sh stop
```

#### 如何手动执行一次分析？
```bash
cd stock_trading_advisor
# 1. 停止服务
./manage_scheduler.sh stop

# 2. 手动执行
python3 scheduler.py --run-now

# 3. 重新启动服务
./manage_scheduler.sh start
```

#### 服务崩溃了怎么办？
**不用担心！** systemd 会在10秒后自动重启服务。

可以通过以下命令查看重启历史：
```bash
sudo journalctl -u stock-scheduler.service | grep restart
```
