#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数优化脚本
使用缓存中的股票数据，优化底背离 lookback_days 参数
目标：找出收益率最高的参数组合
"""

import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List
import glob

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.strategy import MixedStrategy
import config as app_config

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告和错误
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cached_stocks() -> List[str]:
    """
    从缓存目录加载所有股票数据

    Returns:
        股票数据 DataFrame 列表
    """
    cache_dir = Path('data/cache')
    cache_files = list(cache_dir.glob('*.csv'))

    if not cache_files:
        print("❌ 未找到缓存的股票数据")
        print("请先运行 main.py 分析一些股票以生成缓存")
        return []

    stocks_data = []
    print(f"\n找到 {len(cache_files)} 个缓存文件")

    for cache_file in cache_files:
        try:
            df = pd.read_csv(cache_file)
            if len(df) > 0:
                # 从文件名提取股票代码
                stock_code = cache_file.stem.split('_')[0]
                df['stock_code'] = stock_code
                stocks_data.append(df)
                print(f"✓ 加载 {stock_code}: {len(df)} 条数据")
        except Exception as e:
            logger.warning(f"加载 {cache_file} 失败: {e}")
            continue

    return stocks_data


def backtest_with_params(df: pd.DataFrame, lookback_days: int,
                         initial_capital: float = 10000.0) -> Dict:
    """
    使用指定参数进行回测

    Args:
        df: 股票数据
        lookback_days: 底背离检测回溯天数
        initial_capital: 初始资金

    Returns:
        回测结果字典
    """
    # 创建配置
    config = {
        'init_k': 50.0,
        'init_d': 50.0,
        'init_date': '2018-01-02',
        'short_ma': 16,
        'mid_ma': 45,
        'k_threshold': 45,
        'stop_loss': -15.0,
        'lookback_days': lookback_days,  # 使用指定的参数
    }

    # 初始化策略
    strategy = MixedStrategy(config=config, validate_indicators=False)

    try:
        # 执行策略分析
        df_analyzed, _ = strategy.analyze(df.copy())

        if df_analyzed is None:
            return None

        # 执行回测
        backtest_result = strategy.backtest(df_analyzed, initial_capital)

        return backtest_result

    except Exception as e:
        logger.warning(f"回测失败: {e}")
        return None


def optimize_lookback_days(stocks_data: List[pd.DataFrame],
                           param_range: List[int],
                           initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    优化 lookback_days 参数

    Args:
        stocks_data: 股票数据列表
        param_range: 要测试的参数范围
        initial_capital: 初始资金

    Returns:
        优化结果 DataFrame
    """
    results = []

    print(f"\n开始参数优化...")
    print(f"测试参数范围: {param_range}")
    print(f"股票数量: {len(stocks_data)}")
    print(f"初始资金: {initial_capital}")
    print("=" * 80)

    for lookback_days in param_range:
        print(f"\n测试参数 lookback_days = {lookback_days}")
        print("-" * 80)

        total_return_list = []
        win_rate_list = []
        total_trades_list = []
        sharpe_list = []
        max_drawdown_list = []

        for df in stocks_data:
            stock_code = df['stock_code'].iloc[0] if 'stock_code' in df.columns else 'unknown'

            # 回测
            result = backtest_with_params(df, lookback_days, initial_capital)

            if result is None:
                continue

            total_return_list.append(result['total_return'])
            win_rate_list.append(result['win_rate'])
            total_trades_list.append(result['total_trades'])
            sharpe_list.append(result['sharpe_ratio'])
            max_drawdown_list.append(result['max_drawdown'])

            print(f"  {stock_code}: 收益率={result['total_return']:>7.2f}%, "
                  f"胜率={result['win_rate']:>5.1f}%, "
                  f"交易次数={result['total_trades']:>3}, "
                  f"夏普={result['sharpe_ratio']:>6.2f}")

        if not total_return_list:
            print(f"  ⚠️  参数 {lookback_days} 无有效结果")
            continue

        # 计算平均指标
        avg_return = sum(total_return_list) / len(total_return_list)
        avg_win_rate = sum(win_rate_list) / len(win_rate_list)
        avg_trades = sum(total_trades_list) / len(total_trades_list)
        avg_sharpe = sum(sharpe_list) / len(sharpe_list)
        avg_drawdown = sum(max_drawdown_list) / len(max_drawdown_list)

        results.append({
            'lookback_days': lookback_days,
            'avg_return': avg_return,
            'avg_win_rate': avg_win_rate,
            'avg_trades': avg_trades,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_drawdown,
            'valid_stocks': len(total_return_list),
            'total_stocks': len(stocks_data)
        })

        print(f"\n  平均收益率: {avg_return:.2f}%")
        print(f"  平均胜率: {avg_win_rate:.1f}%")
        print(f"  平均交易次数: {avg_trades:.1f}")
        print(f"  平均夏普比率: {avg_sharpe:.2f}")
        print(f"  平均最大回撤: {avg_drawdown:.2f}%")

    return pd.DataFrame(results)


def print_optimization_results(results_df: pd.DataFrame):
    """
    打印优化结果

    Args:
        results_df: 优化结果 DataFrame
    """
    if results_df.empty:
        print("\n❌ 没有优化结果")
        return

    print("\n" + "=" * 80)
    print("参数优化结果汇总")
    print("=" * 80)

    # 按平均收益率排序
    results_df = results_df.sort_values('avg_return', ascending=False)

    print("\n所有参数结果（按平均收益率排序）：")
    print("-" * 80)
    print(f"{'lookback_days':<15} {'平均收益率':<12} {'平均胜率':<10} "
          f"{'平均交易':<10} {'夏普比率':<10} {'最大回撤':<10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['lookback_days']:<15} {row['avg_return']:>10.2f}% "
              f"{row['avg_win_rate']:>8.1f}% {row['avg_trades']:>8.1f} "
              f"{row['avg_sharpe']:>9.2f} {row['avg_drawdown']:>9.2f}%")

    # 找出最优参数
    best_return = results_df.iloc[0]
    best_sharpe = results_df.sort_values('avg_sharpe', ascending=False).iloc[0]
    best_winrate = results_df.sort_values('avg_win_rate', ascending=False).iloc[0]

    print("\n" + "=" * 80)
    print("最优参数推荐：")
    print("=" * 80)

    print(f"\n1️⃣  最高收益率参数: lookback_days = {int(best_return['lookback_days'])}")
    print(f"   - 平均收益率: {best_return['avg_return']:.2f}%")
    print(f"   - 平均胜率: {best_return['avg_win_rate']:.1f}%")
    print(f"   - 夏普比率: {best_return['avg_sharpe']:.2f}")

    print(f"\n2️⃣  最高夏普比率参数: lookback_days = {int(best_sharpe['lookback_days'])}")
    print(f"   - 平均收益率: {best_sharpe['avg_return']:.2f}%")
    print(f"   - 平均胜率: {best_sharpe['avg_win_rate']:.1f}%")
    print(f"   - 夏普比率: {best_sharpe['avg_sharpe']:.2f}")

    print(f"\n3️⃣  最高胜率参数: lookback_days = {int(best_winrate['lookback_days'])}")
    print(f"   - 平均收益率: {best_winrate['avg_return']:.2f}%")
    print(f"   - 平均胜率: {best_winrate['avg_win_rate']:.1f}%")
    print(f"   - 夏普比率: {best_winrate['avg_sharpe']:.2f}")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("底背离参数优化工具")
    print("=" * 80)

    # 1. 加载缓存的股票数据
    stocks_data = load_cached_stocks()

    if not stocks_data:
        return

    # 2. 定义参数范围（测试130-160）
    param_range = [130, 140, 150, 160]  # 测试更大参数找出最优值

    # 3. 执行参数优化
    results_df = optimize_lookback_days(
        stocks_data=stocks_data,
        param_range=param_range,
        initial_capital=10000.0
    )

    # 4. 打印结果
    print_optimization_results(results_df)

    # 5. 保存结果到 CSV
    output_file = 'optimization_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
