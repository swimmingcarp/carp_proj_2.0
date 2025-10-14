#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速示例 - 演示如何使用 API
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_fetcher import DataFetcher
from src.strategy import MixedStrategy
from src.analyzer import SignalAnalyzer


def example_single_stock():
    """示例：分析单只股票"""
    print("=" * 60)
    print("示例 1: 分析单只股票")
    print("=" * 60)

    # 1. 初始化组件
    fetcher = DataFetcher(source='akshare')
    strategy = MixedStrategy()
    analyzer = SignalAnalyzer()

    # 2. 获取数据（平安银行）
    stock_code = '000001'
    print(f"\n获取股票 {stock_code} 数据...")
    result = fetcher.get_k_data(stock_code, start_date='2020-01-01')

    # 解包结果
    if isinstance(result, tuple):
        df, validation_report = result
    else:
        df = result
        validation_report = None

    # 显示数据验证结果
    if validation_report:
        print(fetcher.validator.format_report(validation_report))
        if validation_report.get('status') == 'FAILED':
            print("数据验证失败，退出")
            return

    if df is None or len(df) == 0:
        print("获取数据失败")
        return

    print(f"成功获取 {len(df)} 条数据")

    # 3. 执行策略分析
    print("执行策略分析...")
    df_analyzed = strategy.analyze(df)

    if df_analyzed is None:
        print("分析失败")
        return

    # 4. 获取最新信号
    signal = strategy.get_latest_signal(df_analyzed)

    # 5. 显示结果
    print(analyzer.format_signal(signal, {'code': stock_code, 'name': '平安银行'}))

    # 6. 回测
    initial_capital = 10000
    backtest_result = strategy.backtest(df_analyzed, initial_capital=initial_capital)
    print(analyzer.format_backtest_result(backtest_result))

    # 7. 显示历史买卖点
    trading_signals = strategy.get_trading_signals(df_analyzed, initial_capital=initial_capital)
    if trading_signals['total_trades'] > 0:
        print(analyzer.format_trading_signals(trading_signals, show_limit=5))

    print(analyzer.generate_recommendation(signal, backtest_result))


def example_batch_analysis():
    """示例：批量分析"""
    print("\n" + "=" * 60)
    print("示例 2: 批量分析")
    print("=" * 60)

    # 股票列表
    stock_codes = ['000001', '000002', '600519']

    fetcher = DataFetcher(source='akshare')
    strategy = MixedStrategy()
    analyzer = SignalAnalyzer()

    results = []

    for code in stock_codes:
        print(f"\n分析 {code}...")

        try:
            result = fetcher.get_k_data(code, start_date='2020-01-01')

            # 解包结果
            if isinstance(result, tuple):
                df, validation_report = result
            else:
                df = result
                validation_report = None

            # 检查数据验证
            if validation_report and validation_report.get('status') == 'FAILED':
                print(f"✗ {code} 数据验证失败")
                continue

            if df is None or len(df) == 0:
                continue

            df_analyzed = strategy.analyze(df)
            if df_analyzed is None:
                continue

            signal = strategy.get_latest_signal(df_analyzed)
            signal['code'] = code

            results.append(signal)
            print(f"✓ {code} 分析完成")

        except Exception as e:
            print(f"✗ {code} 分析失败: {e}")
            continue

    # 显示批量结果
    print(analyzer.batch_analyze(results))


def example_custom_config():
    """示例：自定义策略参数"""
    print("\n" + "=" * 60)
    print("示例 3: 自定义策略参数")
    print("=" * 60)

    # 自定义配置
    custom_config = {
        'init_k': 50.0,
        'init_d': 50.0,
        'init_date': '2018-01-02',
        'short_ma': 20,      # 使用 20 日均线代替 16 日
        'mid_ma': 60,        # 使用 60 日均线代替 45 日
        'k_threshold': 40,   # K 值阈值调整为 40
        'stop_loss': -10.0,  # 跌停保护调整为 -10%
    }

    fetcher = DataFetcher(source='akshare')
    strategy = MixedStrategy(config=custom_config)
    analyzer = SignalAnalyzer()

    stock_code = '000001'
    print(f"\n使用自定义参数分析 {stock_code}...")
    print(f"参数: 短期均线={custom_config['short_ma']}, 中期均线={custom_config['mid_ma']}")

    result = fetcher.get_k_data(stock_code, start_date='2020-01-01')

    # 解包结果
    if isinstance(result, tuple):
        df, validation_report = result
    else:
        df = result
        validation_report = None

    if df is None or len(df) == 0:
        print("获取数据失败")
        return

    df_analyzed = strategy.analyze(df)
    if df_analyzed is None:
        print("分析失败")
        return

    signal = strategy.get_latest_signal(df_analyzed)
    print(analyzer.format_signal(signal, {'code': stock_code, 'name': '平安银行'}))


if __name__ == '__main__':
    try:
        # 运行示例 1
        example_single_stock()

        # 运行示例 2（批量分析）
        # example_batch_analysis()

        # 运行示例 3（自定义参数）
        # example_custom_config()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
