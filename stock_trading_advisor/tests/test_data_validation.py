#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证功能测试
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_validator import DataValidator


def generate_realistic_ohlc(days=100, base_price=12.0, max_daily_change=0.05):
    """
    生成逻辑合理的OHLC数据

    Args:
        days: 天数
        base_price: 起始价格
        max_daily_change: 最大日涨跌幅

    Returns:
        包含open, close, high, low的字典
    """
    close_prices = [base_price]

    for i in range(1, days):
        change = np.random.uniform(-max_daily_change, max_daily_change)
        close_prices.append(close_prices[-1] * (1 + change))

    close_prices = np.array(close_prices)
    open_prices = close_prices * np.random.uniform(0.99, 1.01, days)

    # 确保high是最高，low是最低
    high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(1.001, 1.02, days)
    low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(0.98, 0.999, days)

    return {
        'open': open_prices,
        'close': close_prices,
        'high': high_prices,
        'low': low_prices
    }


def test_complete_data():
    """测试完整且正常的数据"""
    print("\n" + "=" * 60)
    print("测试 1: 完整且正常的数据")
    print("=" * 60)

    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    ohlc = generate_realistic_ohlc(100)

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': ohlc['open'],
        'close': ohlc['close'],
        'high': ohlc['high'],
        'low': ohlc['low'],
        'volume': np.random.uniform(1000000, 5000000, 100),
    })

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST001')

    print(validator.format_report(report))
    assert report['status'] == 'PASSED', f"应该通过验证，但状态是 {report['status']}"
    print("✓ 测试通过")


def test_missing_values():
    """测试缺失值处理"""
    print("\n" + "=" * 60)
    print("测试 2: 包含缺失值的数据")
    print("=" * 60)

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.random.uniform(10, 15, 100),
        'close': np.random.uniform(10, 15, 100),
        'high': np.random.uniform(14, 16, 100),
        'low': np.random.uniform(9, 11, 100),
        'volume': np.random.uniform(1000000, 5000000, 100),
    })

    # 添加一些缺失值
    df.loc[5:7, 'close'] = np.nan

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST002')

    print(validator.format_report(report))
    assert len(df_cleaned) == 97, "应该删除3行缺失数据"
    print(f"✓ 测试通过 - 删除了 {100 - len(df_cleaned)} 行缺失数据")


def test_duplicate_dates():
    """测试重复日期处理"""
    print("\n" + "=" * 60)
    print("测试 3: 包含重复日期的数据")
    print("=" * 60)

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.random.uniform(10, 15, 100),
        'close': np.random.uniform(10, 15, 100),
        'high': np.random.uniform(14, 16, 100),
        'low': np.random.uniform(9, 11, 100),
        'volume': np.random.uniform(1000000, 5000000, 100),
    })

    # 添加重复数据
    duplicate_row = df.iloc[10:11].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True)

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST003')

    print(validator.format_report(report))
    assert len(df_cleaned) == 100, "应该删除重复数据"
    print("✓ 测试通过 - 删除了重复数据")


def test_abnormal_price_change():
    """测试异常涨跌幅检测"""
    print("\n" + "=" * 60)
    print("测试 4: 异常涨跌幅检测")
    print("=" * 60)

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close_prices = np.random.uniform(10, 11, 100)
    # 添加一个异常涨幅
    close_prices[50] = close_prices[49] * 1.25  # 25%涨幅

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': close_prices * 0.99,
        'close': close_prices,
        'high': close_prices * 1.02,
        'low': close_prices * 0.98,
        'volume': np.random.uniform(1000000, 5000000, 100),
    })

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST004')

    print(validator.format_report(report))
    assert any('异常涨跌幅' in str(issue) for issue in report['issues']), \
        "应该检测到异常涨跌幅"
    print("✓ 测试通过 - 检测到异常涨跌幅")


def test_invalid_ohlc():
    """测试 OHLC 逻辑错误检测"""
    print("\n" + "=" * 60)
    print("测试 5: OHLC 逻辑错误检测")
    print("=" * 60)

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.random.uniform(10, 15, 100),
        'close': np.random.uniform(10, 15, 100),
        'high': np.random.uniform(14, 16, 100),
        'low': np.random.uniform(9, 11, 100),
        'volume': np.random.uniform(1000000, 5000000, 100),
    })

    # 添加 OHLC 逻辑错误：最高价低于收盘价
    df.loc[20, 'high'] = df.loc[20, 'close'] - 1

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST005')

    print(validator.format_report(report))
    assert len(df_cleaned) == 99, "应该删除逻辑错误的数据"
    print("✓ 测试通过 - 检测并删除了 OHLC 逻辑错误")


def test_insufficient_data():
    """测试数据量不足"""
    print("\n" + "=" * 60)
    print("测试 6: 数据量不足")
    print("=" * 60)

    # 只有30天数据（少于默认的60天）
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.random.uniform(10, 15, 30),
        'close': np.random.uniform(10, 15, 30),
        'high': np.random.uniform(14, 16, 30),
        'low': np.random.uniform(9, 11, 30),
        'volume': np.random.uniform(1000000, 5000000, 30),
    })

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST006')

    print(validator.format_report(report))
    assert report['status'] == 'FAILED', "应该验证失败"
    assert df_cleaned is None, "应该返回 None"
    print("✓ 测试通过 - 正确检测到数据量不足")


def test_zero_volume():
    """测试零成交量检测"""
    print("\n" + "=" * 60)
    print("测试 7: 零成交量检测（停牌）")
    print("=" * 60)

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    volumes = np.random.uniform(1000000, 5000000, 100)
    # 添加几天零成交量（停牌）
    volumes[30:33] = 0

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.random.uniform(10, 15, 100),
        'close': np.random.uniform(10, 15, 100),
        'high': np.random.uniform(14, 16, 100),
        'low': np.random.uniform(9, 11, 100),
        'volume': volumes,
    })

    validator = DataValidator()
    df_cleaned, report = validator.validate(df, stock_code='TEST007')

    print(validator.format_report(report))
    assert any('零成交量' in str(w) for w in report['warnings']), \
        "应该检测到零成交量"
    print("✓ 测试通过 - 检测到零成交量（停牌标志）")


def test_custom_config():
    """测试自定义配置"""
    print("\n" + "=" * 60)
    print("测试 8: 自定义验证配置")
    print("=" * 60)

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close_prices = np.random.uniform(10, 11, 100)
    # 添加一个15%的涨幅
    close_prices[50] = close_prices[49] * 1.15

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': close_prices * 0.99,
        'close': close_prices,
        'high': close_prices * 1.02,
        'low': close_prices * 0.98,
        'volume': np.random.uniform(1000000, 5000000, 100),
    })

    # 使用自定义配置：降低涨跌幅阈值到10%
    custom_config = {
        'max_price_change_pct': 10.0,
    }

    validator = DataValidator(config=custom_config)
    df_cleaned, report = validator.validate(df, stock_code='TEST008')

    print(validator.format_report(report))
    assert any('异常涨跌幅' in str(issue) for issue in report['issues']), \
        "应该用自定义阈值检测到异常涨跌幅"
    print("✓ 测试通过 - 自定义配置生效")


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("*" * 60)
    print("数据验证功能测试套件")
    print("*" * 60)

    tests = [
        test_complete_data,
        test_missing_values,
        test_duplicate_dates,
        test_abnormal_price_change,
        test_invalid_ohlc,
        test_insufficient_data,
        test_zero_volume,
        test_custom_config,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 测试出错: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
