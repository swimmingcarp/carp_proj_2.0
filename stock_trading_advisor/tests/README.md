# 未来函数检测测试

## 运行测试

```bash
python -m stock_trading_advisor.tests.test_lookahead_bias_smart
```

## 测试说明

- 使用真实数据 `02367_qfq.csv` 进行测试
- 采用智能采样：100%覆盖信号点 + 随机采样平静期
- 检测策略是否使用了未来数据（Look-Ahead Bias）
