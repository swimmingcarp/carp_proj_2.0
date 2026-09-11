# Research Tools

研究与审计工具，**不属于正式运行路径**。正式回测和信号仍然只走
`stock_trading_advisor/main.py`；本目录的脚本只用于候选策略的隔离验证、策略性质诊断和
未来函数排查。

工具不写入 `stock_trading_advisor/`，输出一律落在 `research/out/`。

## 脚本

| 脚本 | 用途 |
|---|---|
| `hooked.py` | `HookedStrategy`：正式策略 `analyze()` 的可挂钩重实现，逐笔与正式策略一致（内置自检） |
| `harness.py` | 候选策略 × 开发池分组或已揭示 OOS230 的隔离评估；产物带数据池身份 |
| `diagnose.py` | 策略性质诊断：按相同费用与买入持有对比、分位分层、收益归因、捕获率 |
| `prefix_replay.py` | 完整逐日前缀回放，支持注入外部序列并一并截断 |
| `panel.py` | 横截面面板：同市场收益排名、市场宽度、IBD 相对强度，全部因果 |
| `portfolio.py` | 组合回测：只在真实入场日分配固定资金；排序候选必须有当日有效评分 |
| `factors.py` | 生成仅使用当日及历史 OHLCV 的候选因子和带终点日期的未来收益标签 |
| `factor_eval.py` | 按时间段评估因子，并剔除未来收益跨越分段边界的样本 |

## 用法

```bash
# 1. 全量自检：确认可挂钩重实现与正式策略逐笔一致（任何候选实验前必须先跑）
python3 research/harness.py --selfcheck

# 2. 评估候选（候选定义在 research/candidates_local.py，见下）
python3 research/harness.py baseline ma60_12

# 3. 策略性质诊断
python3 research/diagnose.py ma60_12

# 4. 正式未来函数前缀回放（默认检查每一行；--points 仅供探索抽样）
python3 research/prefix_replay.py baseline --stocks 300274 00700

# 5. 已揭示池外回归（禁止在其上调参）
python3 research/harness.py baseline --pool oos

# 6. 横截面面板与组合层评估
python3 research/panel.py && python3 research/panel.py --pool oos
python3 research/portfolio.py baseline --rank cs60 cs20 random --slots 10 20 30 --seeds 30

# 7. 因子研究（OOS230 只允许回归检查）
python3 research/factors.py --pool dev
python3 research/factor_eval.py --pool dev
```

## 两个股票池

- `dev`：`stock_trading_advisor/data/backtest_data/`，305 只，策略已在其上反复开发，不能当作未见样本。
- `oos`：`stock_trading_advisor/data/oos_data/`，230 只，与开发池零重叠，按流动性和数据质量抽样，
  不使用候选策略表现选股。它已用于否决 `ma60_12`，现在是**已揭示外部回归池**，不是冻结验收池。
  数据来自 Yahoo 重建复权序列，行业风格和数据源都与开发池不同，结论必须附带这些限制。

候选策略写在 `research/candidates_local.py`（该文件不纳入版本管理，属于一次性实验草稿）。
它必须定义 `REGISTRY: dict[str, type]`，值是 `HookedStrategy` 的子类。`baseline` 与
`ma60_12` 由 `hooked.py` 内置提供。

产物命名为 `<candidate>__<pool>__<kind>`，例如 `baseline__dev__metrics.json` 与
`baseline__oos_revealed__metrics.json`，不同数据池不会互相覆盖。

## 纪律

- 候选未通过隔离验证前，不得写入 `stock_trading_advisor/src/`。
- `harness.py` 每次运行先对全部开发池股票做自检；自检失败说明正式策略已改动而 `hooked.py` 未同步，
  此时所有候选结果无效。
- 结论必须同时看 7 个分组，并按 `AGENTS.md` 报告退步项。
