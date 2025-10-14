# 数据验证功能说明

## 概述

数据验证模块为股票数据提供全面的质量检查和异常值过滤，确保策略分析基于可靠的数据基础。

## 功能特性

### 1. 数据完整性检查
- ✅ 检查必需列是否存在（date, open, close, high, low, volume）
- ✅ 检测并报告缺失值
- ✅ 自动删除包含缺失值的记录
- ✅ 检查数据量是否充足（默认至少60天）

### 2. 数据类型验证
- ✅ 自动转换数值列类型
- ✅ 统一日期格式
- ✅ 处理类型转换错误

### 3. 重复数据处理
- ✅ 检测并删除重复日期的数据
- ✅ 保留最早的记录

### 4. 异常值检测
- ✅ **单日涨跌幅检测**：标记超过20%的异常涨跌幅
- ✅ **价格合理性**：检查价格是否在合理范围内（0.01 - 10000）
- ✅ **OHLC逻辑验证**：
  - 最高价 >= 开盘价、收盘价、最低价
  - 最低价 <= 开盘价、收盘价、最高价
- ✅ **连续相同价格检测**：识别可能的停牌情况

### 5. 成交量异常检测
- ✅ 检测零成交量（停牌标志）
- ✅ 检测异常大成交量（超过均值50倍）

### 6. 时间连续性检查
- ✅ 检测数据中的时间间隔
- ✅ 标记超过7天的间隔（假期或停牌）

### 7. 数据质量评分
- 基于发现的问题计算质量评分（0-100分）
- CRITICAL问题扣30分
- 普通问题扣10分
- 警告扣5分

## 使用方法

### 基本使用（自动启用）

默认情况下，数据验证功能已启用：

```python
from src.data_fetcher import DataFetcher

# 创建数据获取器（默认启用验证）
fetcher = DataFetcher(source='akshare')

# 获取数据（返回 DataFrame 和验证报告）
df, report = fetcher.get_k_data('000001', start_date='2020-01-01')

# 检查验证状态
if report['status'] == 'FAILED':
    print("数据验证失败，无法继续")
elif report['status'] == 'WARNING':
    print("数据存在质量问题，建议谨慎使用")
else:
    print(f"数据验证通过，质量评分: {report['data_quality_score']}")
```

### 禁用数据验证

如果需要原始数据（不推荐）：

```python
# 禁用数据验证
fetcher = DataFetcher(source='akshare', validate_data=False)

# 返回值为 (DataFrame, None)
df, report = fetcher.get_k_data('000001')
```

### 自定义验证配置

```python
from src.data_validator import DataValidator

# 自定义配置
custom_config = {
    'max_price_change_pct': 15.0,    # 降低涨跌幅阈值到15%
    'min_data_points': 120,          # 要求至少120天数据
    'max_missing_ratio': 0.05,       # 允许最多5%缺失率
}

validator = DataValidator(config=custom_config)

# 手动验证数据
df_cleaned, report = validator.validate(df, stock_code='000001')

# 打印详细报告
print(validator.format_report(report))
```

## 验证报告示例

### 成功通过验证
```
==================================================
数据验证报告 - 000001
==================================================
✓ 数据验证通过
原始数据行数: 1200
清洗后行数: 1200
数据质量评分: 100.0/100
==================================================
```

### 存在警告
```
==================================================
数据验证报告 - 000001
==================================================
⚠️  数据验证通过（有警告）
原始数据行数: 1200
清洗后行数: 1195
数据质量评分: 85.0/100

问题列表:
  • 删除了 5 行包含缺失值的数据

警告信息:
  • 异常涨跌幅: 2023-03-15 (12.50%)
  • 存在 2 天零成交量数据（可能停牌）
  • 存在 1 个时间间隔 > 7天（可能是假期或停牌）
    - 2023-10-01 到 2023-10-09: 8 天
==================================================
```

### 验证失败
```
==================================================
数据验证报告 - 000001
==================================================
❌ 数据验证失败
原始数据行数: 50
清洗后行数: 0
数据质量评分: 0.0/100

问题列表:
  • CRITICAL: 数据量不足: 50 < 60
  • CRITICAL: 列 'close' 缺失率过高: 15.00%
  • CRITICAL: 3 条数据的最高价小于其他价格
==================================================
```

## 配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_price_change_pct` | 20.0 | 单日最大涨跌幅（%） |
| `max_consecutive_same` | 5 | 最大连续相同值天数 |
| `min_volume` | 100 | 最小成交量 |
| `max_volume_ratio` | 50.0 | 最大成交量比率（与均值比） |
| `min_price` | 0.01 | 最小价格 |
| `max_price` | 10000.0 | 最大价格 |
| `min_data_points` | 60 | 最小数据点数量（约3个月） |
| `max_missing_ratio` | 0.1 | 最大缺失率（10%） |

## 集成示例

### 在策略分析中使用

```python
from src.data_fetcher import DataFetcher
from src.strategy import MixedStrategy

fetcher = DataFetcher(source='akshare')
strategy = MixedStrategy()

# 获取并验证数据
df, report = fetcher.get_k_data('000001', start_date='2020-01-01')

# 检查数据质量
if report['status'] == 'FAILED':
    print("数据质量不合格，跳过该股票")
    exit(1)

# 继续策略分析
df_analyzed = strategy.analyze(df)
signal = strategy.get_latest_signal(df_analyzed)
```

### 批量分析中的应用

```python
stock_codes = ['000001', '000002', '600519']
results = []

for code in stock_codes:
    df, report = fetcher.get_k_data(code)

    # 只分析高质量数据
    if report['data_quality_score'] >= 80:
        df_analyzed = strategy.analyze(df)
        signal = strategy.get_latest_signal(df_analyzed)
        results.append(signal)
    else:
        print(f"跳过 {code}，数据质量评分: {report['data_quality_score']}")
```

## 最佳实践

1. **始终启用数据验证**：除非有特殊原因，建议保持验证功能开启

2. **关注质量评分**：建议只使用评分 >= 80 的数据进行策略分析

3. **处理警告**：
   - 停牌数据：考虑使用更长的历史周期
   - 异常涨跌幅：注意ST股票和新股特殊情况
   - 时间间隔：正常假期间隔可忽略

4. **定期检查**：对于实盘交易，建议定期检查数据质量报告

5. **多源验证**：对关键交易决策，可使用多个数据源交叉验证

## 常见问题

### Q: 为什么我的数据被标记为异常涨跌幅？
A: ST股票、新股上市、重大事件可能导致大幅波动。这是正常现象，系统只会标记警告而不会删除数据。

### Q: 数据验证失败后还能继续分析吗？
A: 不建议。验证失败通常意味着数据存在严重问题（如数据量不足、大量缺失等），可能导致策略分析结果不可靠。

### Q: 如何提高数据质量评分？
A:
- 使用更长的历史周期（获取更多数据点）
- 选择流动性好的股票（减少停牌和异常）
- 验证数据源可靠性

### Q: 能否自定义异常值阈值？
A: 可以。创建自定义配置传递给 `DataValidator`，参考上面的"自定义验证配置"部分。

## 技术架构

```
DataFetcher
    ↓
获取原始数据 (AKShare/Tushare/yfinance)
    ↓
DataValidator.validate()
    ├─ 完整性检查
    ├─ 类型验证
    ├─ 去重处理
    ├─ 异常值检测
    ├─ 价格合理性
    ├─ 成交量检查
    └─ 连续性检查
    ↓
返回清洗后的DataFrame + 验证报告
    ↓
策略分析 (MixedStrategy)
```

## 更新日志

**v1.0 (2025-10-14)**
- ✨ 新增完整的数据验证功能
- ✨ 支持数据完整性检查
- ✨ 支持异常值检测和过滤
- ✨ 支持数据质量评分
- ✨ 集成到数据获取流程
- 📝 添加详细的验证报告输出
