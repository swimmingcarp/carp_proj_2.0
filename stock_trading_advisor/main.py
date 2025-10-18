#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Trading Advisor - 主程序
提供命令行界面进行股票分析和买卖提示
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_fetcher import DataFetcher
from src.strategy import MixedStrategy
from src.analyzer import SignalAnalyzer
import config as app_config  # 导入应用配置


def setup_logging(config: dict):
    """配置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', 'logs/trading.log')

    # 创建日志目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]

    if log_config.get('console', True):
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def analyze_stock(stock_code: str, config: dict, show_backtest: bool = True):
    """
    分析单只股票

    Args:
        stock_code: 股票代码
        config: 配置字典
        show_backtest: 是否显示回测结果
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始分析股票: {stock_code}")

    # 1. 初始化组件
    data_config = config.get('data_source', {})
    strategy_config = config.get('strategy', {})
    backtest_config = config.get('backtest', {})

    fetcher = DataFetcher(
        source=data_config.get('provider', app_config.DATA_SOURCE),
        cache_enabled=data_config.get('cache_enabled', app_config.CACHE_ENABLED),
        max_retries=app_config.MAX_RETRIES,
        retry_delay=app_config.RETRY_DELAY,
        is_backtest_mode=show_backtest  # 传递回测模式标志
    )
    strategy = MixedStrategy(config=strategy_config)
    analyzer = SignalAnalyzer()

    # 2. 获取数据
    print(f"\n正在获取股票 {stock_code} 的数据...")
    try:
        result = fetcher.get_k_data(
            code=stock_code,
            start_date=backtest_config.get('start_date', '2020-01-01'),
            end_date=backtest_config.get('end_date')
        )

        # 解包结果
        if isinstance(result, tuple):
            df, validation_report = result
        else:
            df = result
            validation_report = None

    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        print(f"❌ 获取数据失败: {e}")
        return

    # 显示数据验证报告
    if validation_report:
        if validation_report.get('status') == 'FAILED':
            print(fetcher.validator.format_report(validation_report))
            print(f"❌ 数据验证失败，无法继续分析")
            return
        elif validation_report.get('status') == 'WARNING':
            print(fetcher.validator.format_report(validation_report))
            print("⚠️  存在数据质量问题，建议谨慎使用分析结果")
        else:
            print(f"✓ 数据验证通过 (质量评分: {validation_report.get('data_quality_score', 0):.1f}/100)")

    if df is None or len(df) == 0:
        print(f"❌ 股票 {stock_code} 无数据")
        return

    print(f"✓ 成功获取 {len(df)} 条数据")

    # 3. 执行策略分析
    print("正在分析...")
    try:
        result = strategy.analyze(df)

        # 解包结果
        if isinstance(result, tuple):
            df_analyzed, indicator_report = result
        else:
            df_analyzed = result
            indicator_report = None

    except Exception as e:
        logger.error(f"策略分析失败: {e}", exc_info=True)
        print(f"❌ 分析失败: {e}")
        return

    # 显示指标验证报告
    if indicator_report:
        if indicator_report.get('status') == 'FAILED':
            print(strategy.indicator_validator.format_report(indicator_report))
            print("⚠️  技术指标存在异常，建议谨慎使用分析结果")
        elif indicator_report.get('status') == 'WARNING':
            print(strategy.indicator_validator.format_report(indicator_report))
        else:
            indicators_str = ', '.join(indicator_report.get('indicators_checked', []))
            print(f"✓ 技术指标验证通过 ({indicators_str})")

    if df_analyzed is None:
        print(f"❌ 分析失败（可能触发跌停保护）")
        return

    print("✓ 分析完成")

    # 4. 获取最新信号
    signal_data = strategy.get_latest_signal(df_analyzed)

    # 5. 获取股票基本信息
    stock_info = fetcher.get_stock_info(stock_code)
    if stock_info:
        signal_data['code'] = stock_code
        signal_data['name'] = stock_info.get('总股本', stock_code)

    # 6. 显示信号
    print(analyzer.format_signal(signal_data, {'code': stock_code, 'name': stock_info.get('股票简称', '') if stock_info else ''}))

    # 7. 回测
    if show_backtest:
        print("正在回测...")
        initial_capital = backtest_config.get('initial_capital', 10000)
        backtest_result = strategy.backtest(
            df_analyzed,
            initial_capital=initial_capital
        )

        if backtest_result:
            print(analyzer.format_backtest_result(backtest_result))

            # 显示历史买卖点
            trading_signals = strategy.get_trading_signals(df_analyzed, initial_capital=initial_capital)
            if trading_signals['total_trades'] > 0:
                print(analyzer.format_trading_signals(trading_signals))

            print(analyzer.generate_recommendation(signal_data, backtest_result))
        else:
            print(analyzer.generate_recommendation(signal_data))
    else:
        print(analyzer.generate_recommendation(signal_data))

    logger.info(f"完成分析股票: {stock_code}")


def batch_analyze(stock_codes: list, config: dict):
    """
    批量分析多只股票

    Args:
        stock_codes: 股票代码列表
        config: 配置字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始批量分析 {len(stock_codes)} 只股票")

    results = []
    total = len(stock_codes)

    for i, code in enumerate(stock_codes, 1):
        print(f"\n[{i}/{total}] 分析 {code}...")

        try:
            # 初始化组件
            data_config = config.get('data_source', {})
            strategy_config = config.get('strategy', {})

            fetcher = DataFetcher(
                source=data_config.get('provider', app_config.DATA_SOURCE),
                cache_enabled=data_config.get('cache_enabled', app_config.CACHE_ENABLED),
                max_retries=app_config.MAX_RETRIES,
                retry_delay=app_config.RETRY_DELAY,
                is_backtest_mode=False  # 批量分析不做回测，使用实时模式
            )
            strategy = MixedStrategy(config=strategy_config)

            # 获取数据
            result = fetcher.get_k_data(code, start_date='2020-01-01')

            # 解包结果
            if isinstance(result, tuple):
                df, validation_report = result
            else:
                df = result
                validation_report = None

            # 检查数据验证状态
            if validation_report and validation_report.get('status') == 'FAILED':
                logger.warning(f"{code} 数据验证失败")
                continue

            if df is None or len(df) == 0:
                continue

            # 分析
            df_analyzed = strategy.analyze(df)
            if df_analyzed is None:
                continue

            # 获取信号
            signal = strategy.get_latest_signal(df_analyzed)
            signal['code'] = code

            results.append(signal)

        except Exception as e:
            logger.error(f"分析 {code} 失败: {e}")
            continue

    # 显示批量结果
    analyzer = SignalAnalyzer()
    print(analyzer.batch_analyze(results))

    logger.info("完成批量分析")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stock Trading Advisor - 股票交易策略分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析单只股票
  python main.py -s 000001

  # 分析单只股票（不显示回测）
  python main.py -s 000001 --no-backtest

  # 批量分析多只股票
  python main.py -b 000001 000002 600519

  # 使用自定义配置文件
  python main.py -s 000001 -c config/my_config.yaml
        """
    )

    parser.add_argument('-s', '--stock', type=str, help='单只股票代码')
    parser.add_argument('-b', '--batch', nargs='+', help='批量股票代码列表')
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--no-backtest', action='store_true', help='不显示回测结果')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    setup_logging(config)

    # 执行分析
    if args.stock:
        analyze_stock(args.stock, config, show_backtest=not args.no_backtest)
    elif args.batch:
        batch_analyze(args.batch, config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
