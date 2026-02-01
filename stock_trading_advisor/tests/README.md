# 未来函数检测测试

## 运行测试

```bash
python -m stock_trading_advisor.tests.test_lookahead_bias_smart
```

## 测试说明

- 使用真实数据进行测试
  - `02367_qfq.csv` - 港股（无震荡入场信号，验证基础策略）
  - `300750_qfq.csv` - A股（1个震荡入场信号）
  - `300274_qfq.csv` - A股（2个震荡入场信号）
- 采用智能采样：100%覆盖信号点 + 随机采样平静期
- 检测策略是否使用了未来数据（Look-Ahead Bias）

## 检测覆盖范围

### 买卖信号点
- `entry_signal` - 标准入场信号
- `exit_signal` - 标准退出信号
- `w_bottom_signal` - W底形态信号
- `bullish_divergence_signal` - 底背离信号
- `sideways_entry` - 震荡市场入场信号（Aroon策略）

### 中间状态
- `mtf_bias` - 多时间框架偏差
- `direction` - 趋势方向
- `is_sideways` - 震荡市场识别（Aroon Oscillator）
- `aroon_osc` - Aroon震荡指标值
