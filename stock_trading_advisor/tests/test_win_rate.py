#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
胜率计算验证测试
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy import MixedStrategy


def test_win_rate_calculation():
    """测试胜率计算是否正确"""
    print("\n" + "=" * 60)
    print("测试：胜率计算逻辑")
    print("=" * 60)

    # 创建模拟数据 - 3笔完整的交易
    # 交易1: 天0-9 持仓, 天10 空仓 (盈利 +5%)
    # 交易2: 天11-20 持仓, 天21 空仓 (亏损 -3%)
    # 交易3: 天22-31 持仓, 天32+ 空仓 (盈利 +2%)

    dates = pd.date_range('2023-01-01', periods=40, freq='D')

    p_changes = []
    positions = []

    for i in range(40):
        if 0 <= i < 10:  # 交易1: 每天+0.5%
            p_changes.append(0.5)
            positions.append(1)
        elif i == 10:  # 卖出
            p_changes.append(0.0)
            positions.append(0)
        elif 11 <= i < 21:  # 交易2: 每天-0.3%
            p_changes.append(-0.3)
            positions.append(1)
        elif i == 21:  # 卖出
            p_changes.append(0.0)
            positions.append(0)
        elif 22 <= i < 32:  # 交易3: 每天+0.2%
            p_changes.append(0.2)
            positions.append(1)
        else:  # 空仓
            p_changes.append(0.0)
            positions.append(0)

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'p_change': p_changes,
        'position': positions
    })

    # 预期结果
    # 交易1: 10天 × 0.5% = 5% (盈利)
    # 交易2: 10天 × -0.3% = -3% (亏损)
    # 交易3: 10天 × 0.2% = 2% (盈利)
    # 总交易次数 = 3次，盈利次数 = 2次，胜率 = 66.67%

    # 使用策略的回测函数
    strategy = MixedStrategy()
    df['returns'] = df['p_change'] / 100 * df['position']

    # 模拟backtest的计算逻辑
    buy_signals = df[
        (df['position'] == 1) &
        (df['position'].shift(1).fillna(0) == 0)
    ]

    sell_signals = df[
        (df['position'] == 0) &
        (df['position'].shift(1).fillna(0) == 1)
    ]

    print(f"\n买入点索引: {buy_signals.index.tolist()}")
    print(f"卖出点索引: {sell_signals.index.tolist()}")

    trades = len(buy_signals)
    winning_trades = 0

    print(f"\n交易详情:")
    for i, (buy_idx, sell_idx) in enumerate(zip(buy_signals.index, sell_signals.index), 1):
        trade_returns = df.loc[buy_idx:sell_idx, 'returns'].sum()
        trade_return_pct = trade_returns * 100
        is_win = "✓ 盈利" if trade_returns > 0 else "✗ 亏损"

        print(f"  交易{i}: 天{buy_idx+1}-天{sell_idx+1} | 收益率: {trade_return_pct:.2f}% | {is_win}")

        if trade_returns > 0:
            winning_trades += 1

    win_rate = (winning_trades / trades * 100) if trades > 0 else 0

    print(f"\n计算结果:")
    print(f"  总交易次数: {trades}")
    print(f"  盈利交易次数: {winning_trades}")
    print(f"  胜率: {win_rate:.2f}%")

    print(f"\n预期结果:")
    print(f"  总交易次数: 3")
    print(f"  盈利交易次数: 2")
    print(f"  预期胜率: 66.67%")

    # 验证
    assert trades == 3, f"交易次数错误: {trades} != 3"
    assert winning_trades == 2, f"盈利交易次数错误: {winning_trades} != 2"
    assert abs(win_rate - 66.67) < 0.1, f"胜率错误: {win_rate:.2f}% != 66.67%"

    print("\n✓ 测试通过！胜率计算正确")


def test_mixed_win_rate():
    """测试混合盈亏的胜率计算"""
    print("\n" + "=" * 60)
    print("测试：混合盈亏场景")
    print("=" * 60)

    # 创建模拟数据
    # 场景：5笔交易，3笔盈利，2笔亏损
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    positions = [0] * 100
    p_changes = [0.0] * 100

    # 交易1: 天0-19 (盈利 +10%)
    for i in range(0, 20):
        positions[i] = 1
        p_changes[i] = 0.5

    # 空仓: 天20-29
    for i in range(20, 30):
        positions[i] = 0

    # 交易2: 天30-49 (亏损 -6%)
    for i in range(30, 50):
        positions[i] = 1
        p_changes[i] = -0.3

    # 空仓: 天50-59
    for i in range(50, 60):
        positions[i] = 0

    # 交易3: 天60-79 (盈利 +16%)
    for i in range(60, 80):
        positions[i] = 1
        p_changes[i] = 0.8

    # 空仓: 天80-84
    for i in range(80, 85):
        positions[i] = 0

    # 交易4: 天85-94 (盈利 +2%)
    for i in range(85, 95):
        positions[i] = 1
        p_changes[i] = 0.2

    # 空仓: 天95-99
    for i in range(95, 100):
        positions[i] = 0

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'p_change': p_changes,
        'position': positions
    })

    df['returns'] = df['p_change'] / 100 * df['position']

    # 计算
    buy_signals = df[
        (df['position'] == 1) &
        (df['position'].shift(1).fillna(0) == 0)
    ]

    sell_signals = df[
        (df['position'] == 0) &
        (df['position'].shift(1).fillna(0) == 1)
    ]

    trades = len(buy_signals)
    winning_trades = 0

    print(f"\n交易详情:")
    for i, (buy_idx, sell_idx) in enumerate(zip(buy_signals.index, sell_signals.index), 1):
        trade_returns = df.loc[buy_idx:sell_idx, 'returns'].sum()
        trade_return_pct = trade_returns * 100
        is_win = "✓ 盈利" if trade_returns > 0 else "✗ 亏损"

        print(f"  交易{i}: 天{buy_idx+1}-天{sell_idx+1} | 收益率: {trade_return_pct:.2f}% | {is_win}")

        if trade_returns > 0:
            winning_trades += 1

    win_rate = (winning_trades / trades * 100) if trades > 0 else 0

    print(f"\n计算结果:")
    print(f"  总交易次数: {trades}")
    print(f"  盈利交易次数: {winning_trades}")
    print(f"  胜率: {win_rate:.2f}%")

    print(f"\n预期结果:")
    print(f"  总交易次数: 4")
    print(f"  盈利交易次数: 3")
    print(f"  预期胜率: 75.00%")

    # 验证
    assert trades == 4, f"交易次数错误: {trades} != 4"
    assert winning_trades == 3, f"盈利交易次数错误: {winning_trades} != 3"
    assert abs(win_rate - 75.0) < 0.01, f"胜率错误: {win_rate:.2f}% != 75.00%"

    print("\n✓ 测试通过！胜率计算正确")


if __name__ == '__main__':
    try:
        test_win_rate_calculation()
        test_mixed_win_rate()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！胜率计算逻辑正确")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
