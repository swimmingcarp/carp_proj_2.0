# Stock Trading Advisor

基于技术指标的股票交易策略分析系统，提供买卖信号提示和策略回测功能。

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
- 📈 **多家族交易策略**：覆盖 `RSI` 趋势跟随、`MA` 回踩、慢牛回踩、`runner breakout`、`trend reclaim` 等不同买点家族
- 🧠 **自适应策略路由**：基于趋势强度、风险状态、runner 画像，在不同市场结构下切换更合适的入场与持有逻辑
- 💹 **单股分析 + 全量离线回测**：既支持单只股票分析，也支持对缓存股票池生成完整批量报告
- 🧾 **交易级复盘能力**：输出买卖点、交易统计、收益/回撤/胜率等核心指标，方便定位坏簇和修策略
- 🖼️ **K 线图与买卖点标注**：支持生成带买卖点的图表用于人工复盘
- 📱 **批量分析与定时提醒**：支持固定时间自动运行，并通过企业微信推送结果
- ⚙️ **配置驱动**：核心参数均可通过 `config.yaml` 调整，便于研究和迭代

⚡ **性能优势**
- 🚀 **本地缓存优先**：批量回测直接复用 `data/cache`，适合高频研究和反复验证
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
编辑 `stock_trading_advisor/config/config.yaml` 修改策略参数

### 5. 运行程序
```bash
# 分析单只股票
python3 stock_trading_advisor/main.py -s 000001 --new-strategy

# 或者使用完整命令
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 001279 --new-strategy
```

**备注：** `--new-strategy` 目前是兼容保留参数，当前主线命令即使不显式传入，也会走我们现在使用的这套策略。文档里保留它，是为了和历史命令、回测记录保持一致。

## 使用方法

### 单只股票分析

```bash
# 激活虚拟环境并分析平安银行（000001）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001 --new-strategy

# 分析贵州茅台（600519）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 600519 --new-strategy

# 不显示回测结果
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001 --new-strategy --no-backtest

# 使用当前主线策略分析单只股票
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 300293 --new-strategy

# 对缓存中的所有股票进行离线回测并输出报告
# 默认就是高并发静默模式：多进程 + 自动按 min(股票数, CPU核心数) 分配 worker + 不逐只刷屏
source venv/bin/activate && python3 stock_trading_advisor/main.py --report --new-strategy

# 显式指定并发（示例：使用 132 个进程）
source venv/bin/activate && STOCK_ADVISOR_REPORT_EXECUTOR=process STOCK_ADVISOR_REPORT_WORKERS=132 python3 stock_trading_advisor/main.py --report --new-strategy

# 如果需要，也可以切到多线程执行器
source venv/bin/activate && STOCK_ADVISOR_REPORT_EXECUTOR=thread STOCK_ADVISOR_REPORT_WORKERS=132 python3 stock_trading_advisor/main.py --report --new-strategy

# 只对指定股票生成离线报告（需已有缓存）
source venv/bin/activate && python3 stock_trading_advisor/main.py --report --new-strategy -b 300293 300274 300750 605117

# 生成K线图并标注买卖点
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001 --new-strategy --chart-generation
# 图片将保存到 reports/kline_000001.png
```

使用 `--report` 时，系统会将完整的批量回测明细保存到 `stock_trading_advisor/reports/cache_backtest_report_YYYYMMDD_HHMMSS.txt`（按时间戳命名），每只股票都会包含最新价格、历史交易对收益表以及交易统计，方便留档和复盘。

`--report` 现在默认使用高并发静默模式：默认执行器为多进程，默认 worker 数为 `min(目标股票数量, CPU 核心数)`。系统会将 `OMP/OPENBLAS/MKL/NUMEXPR` 线程压到 `1`，避免每个子进程再额外开线程导致过度并行。可通过环境变量 `STOCK_ADVISOR_REPORT_EXECUTOR`（`process`/`thread`）和 `STOCK_ADVISOR_REPORT_WORKERS` 覆写执行器与并发数；如需恢复逐只刷屏，可把 `report.verbose` 改回 `true`。

**输出示例**：
```
============================================================
股票代码: 000001
股票名称: 平安银行
当前价格: 12.35
============================================================

📈 【买入信号】 [███░░]
信号强度: 3/5

理由: K值处于超卖区域(38.5), 中期趋势向上

--- 技术指标 ---
KDJ - K: 38.50, D: 42.30
MACD: 0.0234, DIFF: 0.1567
MA16: 12.10
MA45: 11.95
============================================================

📊 回测结果
============================================================
初始资金: ¥10,000.00
最终资金: ¥12,456.78
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
source venv/bin/activate && python3 stock_trading_advisor/main.py -b 000001 000002 600519 601318 --new-strategy

# 使用自定义配置
source venv/bin/activate && python3 stock_trading_advisor/main.py -b 000001 000002 --new-strategy -c stock_trading_advisor/config/your_config.yaml
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
│   ├── new_strategy.py        # 当前主线策略（多家族 + adaptive runner/reclaim routing）
│   ├── strategy.py            # 旧版策略实现
│   ├── factor_library.py      # 因子库（研究/筛选/策略辅助）
│   ├── indicators.py          # 指标计算
│   ├── divergence.py          # 顶底背离检测
│   ├── analyzer.py            # 单股分析与结果整合
│   ├── plotter.py             # K线图与买卖点标注
│   ├── data_fetcher.py        # 数据获取
│   ├── data_validator.py      # 数据校验
│   ├── indicator_validator.py # 指标/未来函数校验辅助
│   ├── market_hours.py        # 交易时间检测
│   ├── wechat_notifier.py     # 微信通知
│   └── personality/           # 走势画像 / 路由辅助模块
│       ├── classifier.py
│       ├── segmenter.py
│       └── pit_stage.py
├── config/
│   ├── config.yaml            # 主配置
│   ├── scheduler_config.yaml  # 定时任务配置
│   ├── report_codes_132.txt   # 离线报告默认股票池
│   ├── watch_list.txt         # 监控股票列表
│   ├── cn_stock_names.txt     # A股名称映射
│   └── hk_stock_names.txt     # 港股名称映射
├── data/
│   ├── cache/                 # 本地行情缓存
│   ├── market_breadth.csv     # 市场宽度数据
│   ├── realtime_positions.json
├── reports/                   # 回测报告、研究输出、图表
├── logs/                      # 运行日志
├── tests/
│   ├── README.md
│   └── test_lookahead_bias_smart.py  # 未来函数/截断一致性检测
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

当前主线不是单一的 `KDJ + 均线` 规则，而是一个**多家族入场 + 动态路由 + 家族化退出**的组合系统。

#### 主要入场家族

1. **趋势跟随家族**
   - `RSI金叉`
   - `RSI多头延续`
   - `RSI趋势买入`
   - `RSI动量加速`

2. **趋势回踩 / 慢趋势家族**
   - `MA回踩因子`
   - `慢牛回踩因子`
   - `趋势再突破`
   - `趋势跑者突破`

3. **反转 / 低吸家族**
   - `底背离信号`
   - `W底形态`
   - `折价区补仓`
   - `跳空回补`
   - `Aroon震荡入场`
   - `双通道信号`

#### 核心思想

- **按市场结构选交易家族**：不是所有股票、所有阶段都用同一种买点。
- **按走势画像做动态放行**：系统会结合趋势强度、风险分数、runner 画像来决定哪些信号更值得放行。
- **买卖尽量同家族接管**：例如慢牛回踩单、runner/reclaim 单，会尽量走自己的持有与退出逻辑，减少被旧退出链误伤。
- **保留窄修复机制**：对真实坏簇做局部修正，比如 `continuation_weak`、`golden_cross_weak`、`extended_hold` 的定点修复。

#### 退出与风控

系统不是单一止损线，而是多层退出叠加：

- 硬止损 / 动态止损
- trailing stop / gain protection
- `extended_hold`
- `runner_hold_guard`
- `structural_trend_hold`
- 慢牛单专属退出链
- 交易后冷却与家族一致性保护

也就是说，强趋势单会尽量少被过早洗掉，弱势单则会更快止损或退出。

## 配置参数

主配置在 [stock_trading_advisor/config/config.yaml](stock_trading_advisor/config/config.yaml)。

当前参数规模已经比较大，实际调整时建议优先看：

- `strategy`：主策略开关与风控参数
- `report`：离线批量回测配置
- `scheduler`：定时任务和提醒配置
- `data_source`：数据源配置

一个更贴近当前使用方式的示例：

```yaml
data_source:
  provider: akshare

report:
  verbose: false
  default_stock_codes_file: config/report_codes_132.txt

strategy:
  # 当前主线参数主要定义在 src/new_strategy.py 的默认配置中，
  # config.yaml 更适合做数据源、报告模式、调度等外层配置。
  use_cache: true
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
