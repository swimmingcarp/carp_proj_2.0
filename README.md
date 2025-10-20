# Stock Trading Advisor

基于技术指标的股票交易策略分析系统，提供买卖信号提示和策略回测功能。

## 功能特性

✨ **核心功能**
- 📊 多技术指标分析（KDJ, MACD, 均线系统）
- 🔍 顶底背离自动检测
- 📈 买卖信号智能提示
- 💹 策略历史回测
- 🎯 信号强度评级
- 📱 批量股票分析
- ⏰ **定时提醒** - 在固定时间自动运行并生成交易提醒（**新功能**）

⚡ **性能优势**
- 向量化计算，性能提升 **300+ 倍**
- Numba 加速，支持大规模数据处理
- 支持多数据源（AKShare, Tushare, yfinance）

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
python3 stock_trading_advisor/main.py -s 000001

# 或者使用完整路径
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 001279
```

## 使用方法

### 单只股票分析

```bash
# 激活虚拟环境并分析平安银行（000001）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001

# 分析贵州茅台（600519）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 600519

# 分析中国平安（001279）
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 001279

# 不显示回测结果
source venv/bin/activate && python3 stock_trading_advisor/main.py -s 000001 --no-backtest
```

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
source venv/bin/activate && python3 stock_trading_advisor/main.py -b 000001 000002 600519 601318

# 使用自定义配置
source venv/bin/activate && python3 stock_trading_advisor/main.py -b 000001 000002 -c config/my_config.yaml
```

**批量输出示例**：
```
================================================================================
📋 批量股票分析
================================================================================

总分析股票数: 4
买入信号: 2
卖出信号: 1
持有信号: 1

--- 买入机会 ---
1. 000001 - 强度: 4/5 - 价格: 12.35 - 底部背离信号, K值超卖
2. 600519 - 强度: 3/5 - 价格: 1680.50 - 中期趋势向上

--- 卖出提醒 ---
1. 000002 - 价格: 25.60 - 顶部背离信号, MACD转负
================================================================================
```

### 📅 定时提醒（新功能）

**在固定时间自动运行并生成交易提醒**

#### 快速开始

1. **安装定时任务依赖**
```bash
cd stock_trading_advisor
source ../venv/bin/activate
pip install schedule
```

2. **配置监控列表** - 编辑 `config/watch_list.txt`
```
300293  # 蓝英装备
002920  # 德赛西威
300757  # 罗博特科
```

3. **设置运行时间** - 编辑 `config/scheduler_config.yaml`
```yaml
schedule:
  run_times:
    - "14:57"  # 下午2:57（收盘前3分钟）
```

4. **启动调度器**
```bash
cd stock_trading_advisor
source ../venv/bin/activate

# 立即测试一次
python3 scheduler.py --run-now

# 启动定时调度器（等待到设定时间自动运行）
python3 scheduler.py

# 后台运行（推荐）
nohup python3 scheduler.py > /tmp/scheduler.log 2>&1 &
```

#### 输出示例

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

#### ⚠️ 重要提示

1. **调度器需要持续运行**：启动后会等待到设定时间自动执行
2. **修改配置后必须重启**：
```bash
# 停止旧调度器
ps aux | grep scheduler.py | grep -v grep
kill <进程ID>

# 启动新调度器
python3 scheduler.py
```

3. **后台运行推荐使用 screen**：
```bash
screen -S stock
python3 scheduler.py
# 按 Ctrl+A 然后 D 离开
```

#### 详细文档

- [定时提醒完整文档](stock_trading_advisor/README_定时提醒.md)
- [快速上手指南](stock_trading_advisor/QUICKSTART.md)
- [使用说明](stock_trading_advisor/定时提醒使用说明.txt)

## 项目结构

```
stock_trading_advisor/
├── src/
│   ├── __init__.py
│   ├── indicators.py          # 技术指标计算（向量化+Numba加速）
│   ├── divergence.py          # 顶底背离检测
│   ├── strategy.py            # 核心交易策略
│   ├── data_fetcher.py        # 数据获取（支持多数据源）
│   ├── analyzer.py            # 买卖信号分析
│   └── market_hours.py        # 交易时间检测
├── config/
│   ├── config.yaml            # 主程序配置
│   ├── scheduler_config.yaml  # 定时调度器配置
│   └── watch_list.txt         # 监控股票列表
├── data/
│   └── cache/                 # 数据缓存
├── logs/
│   ├── trading.log            # 主程序日志
│   └── scheduler.log          # 调度器日志
├── reports/
│   └── trading_alerts.txt     # 交易提醒报告
├── main.py                    # 主程序入口
├── scheduler.py               # 定时调度器（新增）
├── requirements.txt           # 依赖包
└── README.md                  # 使用文档
```

## 策略说明

### Mixed Strategy（混合策略）

**核心思想**：结合趋势跟随和顶底背离，多维度确认买卖信号

#### 买入条件
1. **基础条件**：收盘价 ≥ 16 日均线
2. **增强条件**：
   - K 值 < 45（超卖区域）
   - 45 日均线向上（中期趋势确认）
3. **强制买入**：底部背离次日

#### 卖出条件
满足以下任一条件：
1. 收盘价 < 16 日均线
2. MACD < 0
3. 顶部背离信号

#### 风险控制
- 跌停保护：自动过滤有跌停风险的股票
- 顶部背离期间阻止买入
- 快速止损机制

## 配置参数

编辑 [config/config.yaml](config/config.yaml) 调整策略参数：

```yaml
strategy:
  init_k: 50.0           # KDJ 初始 K 值
  init_d: 50.0           # KDJ 初始 D 值
  short_ma: 16           # 短期均线周期
  mid_ma: 45             # 中期均线周期
  k_threshold: 45        # K 值买入阈值
  stop_loss: -15.0       # 跌停保护阈值
  lookback_days: 100     # 背离检测回溯天数
```

## 数据源配置

### 使用 AKShare（推荐，免费）
```yaml
data_source:
  provider: 'akshare'
```

### 使用 Tushare（需要积分）
```yaml
data_source:
  provider: 'tushare'
```

需要在代码中设置 token：
```python
import tushare as ts
ts.set_token('your_token_here')
```

## 性能优化

相比原始实现，本项目进行了以下优化：

| 模块 | 原始方法 | 优化方法 | 加速比 |
|------|----------|----------|--------|
| 移动平均 | 循环计算 | Pandas rolling | 500x |
| EMA | 循环计算 | Pandas ewm | 150x |
| MACD | 三次循环 | 向量化 | 180x |
| KDJ | iterrows | Numba JIT | 300x |
| 涨跌幅 | 循环计算 | pct_change | 200x |

**总体性能提升：300-500 倍**

回测 1000 只股票：
- 原始实现：~5 小时
- 优化后：**~1 分钟**

## 注意事项

⚠️ **重要提示**
1. 本工具仅供学习和研究使用
2. 所有分析结果**不构成投资建议**
3. 股市有风险，投资需谨慎
4. 建议结合基本面分析和市场环境综合判断

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

#### 修改时间后没有执行？
**原因**：修改配置后没有重启调度器。

**解决方法**：
```bash
# 1. 停止旧调度器
ps aux | grep scheduler.py | grep -v grep
kill <进程ID>

# 2. 启动新调度器
python3 scheduler.py
```

#### 如何确认调度器在运行？
```bash
# 查看进程
ps aux | grep scheduler.py | grep -v grep

# 查看日志
tail -f logs/scheduler.log
```

#### ModuleNotFoundError: No module named 'schedule'
```bash
source ../venv/bin/activate
pip install schedule
```

详见：[定时提醒完整文档](stock_trading_advisor/README_定时提醒.md)

## 后续计划

- [x] ⏰ **定时提醒功能** - 已完成！可在固定时间自动运行
- [ ] 支持更多技术指标（RSI, BOLL 等）
- [ ] 添加可视化图表
- [ ] 邮件/微信通知
- [ ] Web 界面
- [ ] 策略参数自动优化

## 技术支持

如有问题或建议，欢迎提 Issue。

## 许可证

MIT License

---

**免责声明**：本项目仅供学习交流使用，不构成任何投资建议。使用本工具产生的任何投资损失，开发者不承担任何责任。
