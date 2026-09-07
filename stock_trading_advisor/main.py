#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Trading Advisor - 主程序
提供命令行界面进行股票分析和买卖提示
"""

import argparse
import os
import sys
import yaml
import logging
import warnings
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional
from numbers import Integral
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
PROJECT_ROOT = BASE_DIR.parent
BACKTEST_DATA_DIR = BASE_DIR / 'data' / 'backtest_data'
BACKTEST_DATA_ADJUST = 'hfq'

# 确保 src 和项目根目录均在 sys.path 中
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher import DataFetcher
from src.new_strategy import RSITrendStrategy
from src.analyzer import SignalAnalyzer
from src.plotter import plot_kline_with_signals
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
    """
    加载配置文件

    说明:
    - 如果配置文件不存在或读取失败，静默返回空配置 {}
    - 调用方会使用 config.py 中的默认参数作为系统默认配置
    """
    try:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            # 优先按当前工作目录解析，兼容从仓库根目录运行时传入
            # `stock_trading_advisor/config/...` 的常见写法。
            cwd_candidate = (Path.cwd() / config_file).resolve()
            if cwd_candidate.exists():
                config_file = cwd_candidate
            else:
                script_dir = Path(__file__).parent
                config_file = (script_dir / config_file).resolve()

        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        # 不打印错误信息，直接使用系统默认配置（config.py）
        return {}


def detect_market_from_code(stock_code: str) -> str:
    """
    根据股票代码推断市场类型，用于确定手续费率
    """
    if not stock_code:
        return 'CN-A'

    code_clean = stock_code.strip()
    if '.' in code_clean:
        code_clean = code_clean.split('.')[0]

    if code_clean.isdigit():
        if len(code_clean) == 5:
            return 'HK'
        if len(code_clean) == 6:
            return 'CN-A'

    if stock_code.endswith(('.HK', '.hk')):
        return 'HK'

    if stock_code.startswith(('sh', 'sz', 'SH', 'SZ')) or stock_code.endswith(('.SH', '.SZ')):
        return 'CN-A'

    if stock_code and stock_code[0].isalpha():
        return 'US'

    return 'CN-A'


def normalize_stock_code(code: str) -> str:
    """标准化股票代码以匹配本地数据文件"""
    if not code:
        return code
    code = code.strip()
    if '.' in code:
        code = code.split('.')[0]
    if len(code) > 2 and code[:2].lower() in ('sh', 'sz', 'hk'):
        code = code[2:]
    return code.upper()


def calculate_mark_to_market_curve(
    df: pd.DataFrame,
    strategy: RSITrendStrategy,
    initial_capital: float,
) -> Dict[str, List]:
    """Build a close-to-close equity curve while preserving legacy trade accounting."""
    capital = float(initial_capital)
    shares = 0.0
    buy_commission = 0.0
    holding = False
    equity = []
    prices = df['close'].to_numpy(dtype=float)
    signals = df['buy_signal'].to_numpy()

    for signal, price in zip(signals, prices):
        if signal == 1 and not holding:
            shares = capital / price
            buy_commission = strategy._calculate_commission(capital, is_buy=True)
            holding = True
        elif signal == 0 and holding:
            transaction_amount = shares * price
            sell_commission = strategy._calculate_commission(transaction_amount, is_buy=False)
            capital = transaction_amount - buy_commission - sell_commission
            shares = 0.0
            buy_commission = 0.0
            holding = False

        current_equity = shares * price - buy_commission if holding else capital
        equity.append(float(current_equity / initial_capital))

    if holding and equity:
        transaction_amount = shares * prices[-1]
        sell_commission = strategy._calculate_commission(transaction_amount, is_buy=False)
        equity[-1] = float(
            (transaction_amount - buy_commission - sell_commission) / initial_capital
        )

    dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').tolist()
    return {'dates': dates, 'equity': equity}


def analyze_stock(stock_code: str, config: dict, show_backtest: bool = True,
                  df_override: Optional[pd.DataFrame] = None,
                  quiet: bool = False, chart_generation: bool = False) -> Optional[Dict]:
    """
    分析单只股票

    Args:
        stock_code: 股票代码
        config: 配置字典
        show_backtest: 是否显示回测结果
        chart_generation: 是否生成K线图
    """
    logger = logging.getLogger(__name__)
    log_func = logger.info if not quiet else logger.debug
    log_func(f"开始分析股票: {stock_code}")

    def echo(message: str):
        if not quiet:
            print(message)

    # 1. 初始化组件
    data_config = config.get('data_source', {})
    backtest_config = config.get('backtest', {})

    fetcher = None
    validation_report = None

    if df_override is None:
        fetcher = DataFetcher(
            source=data_config.get('provider', app_config.DATA_SOURCE),
            cache_enabled=data_config.get('cache_enabled', app_config.CACHE_ENABLED),
            max_retries=app_config.MAX_RETRIES,
            retry_delay=app_config.RETRY_DELAY,
            is_backtest_mode=show_backtest,
            default_adjust=data_config.get('adjust', app_config.DEFAULT_ADJUST),
        )

        echo(f"\n正在获取股票 {stock_code} 的数据...")
        try:
            result = fetcher.get_k_data(
                code=stock_code,
                start_date=backtest_config.get('start_date', '2020-01-01'),
                end_date=backtest_config.get('end_date')
            )

            if isinstance(result, tuple):
                df, validation_report = result
            else:
                df = result
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            echo(f"❌ 获取数据失败: {e}")
            return None

        market = fetcher._detect_market(stock_code)
    else:
        df = df_override.copy()
        market = detect_market_from_code(stock_code)

    echo("使用RSI趋势策略 (RSITrendStrategy)")
    strategy = RSITrendStrategy(
        market=market,
        stock_code=stock_code
    )
    analyzer = SignalAnalyzer()

    if validation_report:
        if validation_report.get('status') == 'FAILED':
            if fetcher and fetcher.validator:
                echo(fetcher.validator.format_report(validation_report))
            echo("❌ 数据验证失败，无法继续分析")
            return None
        elif validation_report.get('status') == 'WARNING':
            if fetcher and fetcher.validator:
                echo(fetcher.validator.format_report(validation_report))
            echo("⚠️  存在数据质量问题，建议谨慎使用分析结果")
        else:
            echo(f"✓ 数据验证通过 (质量评分: {validation_report.get('data_quality_score', 0):.1f}/100)")

    if df is None or len(df) == 0:
        echo(f"❌ 股票 {stock_code} 无数据")
        return None

    echo(f"✓ 成功获取 {len(df)} 条数据")

    echo("正在分析...")
    try:
        df_analyzed, _ = strategy.analyze(df)
    except Exception as e:
        logger.error(f"策略分析失败: {e}", exc_info=True)
        echo(f"❌ 分析失败: {e}")
        return None

    if df_analyzed is None:
        echo("❌ 分析失败（可能触发跌停保护）")
        return None

    echo("✓ 分析完成")

    signal_data = strategy.get_latest_signal(df_analyzed)

    stock_info = None
    if not show_backtest and fetcher:
        stock_info = fetcher.get_stock_info(stock_code)
        if stock_info:
            signal_data['code'] = stock_code
            signal_data['name'] = stock_info.get('总股本', stock_code)

    if not quiet:
        echo(analyzer.format_signal(signal_data, {'code': stock_code, 'name': stock_info.get('股票简称', '') if stock_info else ''}))

    backtest_result = None
    trading_signals = None
    mark_to_market_curve = None
    if show_backtest:
        echo("正在回测...")
        initial_capital = backtest_config.get('initial_capital', 10000)
        backtest_result = strategy.backtest(
            df_analyzed,
            initial_capital=initial_capital
        )

        if backtest_result:
            trading_signals = strategy.get_trading_signals(
                df_analyzed,
                initial_capital=initial_capital
            )
            if df_override is not None:
                mark_to_market_curve = calculate_mark_to_market_curve(
                    df_analyzed,
                    strategy,
                    initial_capital,
                )

            if not quiet:
                echo(analyzer.format_backtest_result(backtest_result))
                if trading_signals and trading_signals.get('total_trades', 0) > 0:
                    echo(analyzer.format_trading_signals(trading_signals))
                echo(analyzer.generate_recommendation(signal_data, backtest_result))
        elif not quiet:
            echo(analyzer.generate_recommendation(signal_data))
    elif not quiet:
        echo(analyzer.generate_recommendation(signal_data))

    # K线图生成逻辑
    if chart_generation and show_backtest and trading_signals:
        def extract_dates(points):
            # points可能为dict列表、日期列表、索引列表
            if isinstance(points, dict):
                return []
            if isinstance(points, list):
                # dict列表（如交易明细）
                if points and isinstance(points[0], dict):
                    return [d.get('date') for d in points if 'date' in d]
                # 日期字符串或索引
                return [d for d in points if isinstance(d, (str, int))]
            return []
        buy_idx = extract_dates(trading_signals.get('buy_indices', []))
        sell_idx = extract_dates(trading_signals.get('sell_indices', []))
        if not buy_idx and 'buy_points' in trading_signals:
            buy_idx = extract_dates(trading_signals['buy_points'])
        if not sell_idx and 'sell_points' in trading_signals:
            sell_points_for_plot = [
                point for point in trading_signals['sell_points']
                if not (isinstance(point, dict) and point.get('is_open', False))
            ]
            sell_idx = extract_dates(sell_points_for_plot)

        try:
            out_path = plot_kline_with_signals(
                df_analyzed,
                buy_idx,
                sell_idx,
                stock_code,
            )
            echo(f"K线图已保存到: {out_path}")
        except Exception as e:
            echo(f"K线图生成失败: {e}")

    log_func(f"完成分析股票: {stock_code}")
    return {
        'signal': signal_data,
        'backtest': backtest_result,
        'trading_signals': trading_signals,
        'mark_to_market_curve': mark_to_market_curve,
    }


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
            fetcher = DataFetcher(
                source=data_config.get('provider', app_config.DATA_SOURCE),
                cache_enabled=data_config.get('cache_enabled', app_config.CACHE_ENABLED),
                max_retries=app_config.MAX_RETRIES,
                retry_delay=app_config.RETRY_DELAY,
                is_backtest_mode=False,  # 批量分析不做回测，使用实时模式
                default_adjust=data_config.get('adjust', app_config.DEFAULT_ADJUST),
            )
            strategy = RSITrendStrategy(
                market=detect_market_from_code(code),
                stock_code=code
            )

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
            analyzed_result = strategy.analyze(df)
            if isinstance(analyzed_result, tuple):
                df_analyzed = analyzed_result[0]
            else:
                df_analyzed = analyzed_result

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


def format_stock_report(stock_code: str, current_price: float, signal_data: Optional[Dict],
                        backtest_result: Dict, trading_signals: Optional[Dict]) -> str:
    """格式化单只股票的交易明细报告"""
    lines = []
    lines.append(f"股票代码: {stock_code}")
    lines.append(f"当前价格: {current_price:.2f}")
    lines.append("=" * 60)
    lines.append("━━━ 交易对收益分析 ━━━")
    lines.append("说明: 买入价和卖出价均为当日收盘价")
    lines.append("      剩余本金已扣除交易手续费和印花税")

    header = "{:<6} {:<15} {:>12} {:<15} {:>10} {:>12} {:>15}".format(
        "序号", "买入日期", "买入价", "卖出日期", "卖出价", "收益率", "剩余本金"
    )
    lines.append(header)
    lines.append("-" * 100)

    buy_points = trading_signals.get('buy_points', []) if trading_signals else []
    sell_points = trading_signals.get('sell_points', []) if trading_signals else []
    trades = trading_signals.get('trades', []) if trading_signals else []
    initial_capital = backtest_result.get('initial_capital', 10000.0)

    def _format_date(value):
        if value is None:
            return "--"
        return str(value).split()[0]

    for idx, (buy, sell) in enumerate(zip_longest(buy_points, sell_points, fillvalue=None), 1):
        buy_price = buy.get('price') if buy else None
        sell_price = sell.get('price') if sell else None
        profit_rate = None
        if buy_price and sell_price:
            profit_rate = (sell_price - buy_price) / buy_price * 100

        trade_entry = trades[idx - 1] if idx - 1 < len(trades) else None
        capital_after = trade_entry.get('capital') if trade_entry else None

        is_open = bool(sell and sell.get('is_open'))

        buy_date_str = _format_date(buy.get('date')) if buy else "--"
        buy_price_str = f"{buy_price:,.2f}" if buy_price is not None else "--"

        if is_open:
            sell_date_str = "持仓中"
            sell_price_display = "--"
        else:
            sell_date_str = _format_date(sell.get('date')) if sell else "--"
            sell_price_display = f"{sell_price:,.2f}" if sell_price is not None else "--"

        if profit_rate is not None:
            profit_suffix = " (浮盈)" if is_open else ""
            profit_display = f"{profit_rate:+.2f}%{profit_suffix}"
        else:
            profit_display = "--"

        capital_display = f"¥{capital_after:,.2f}" if capital_after is not None else "--"

        lines.append(
            "{:<6} {:<15} {:>12} {:<15} {:>10} {:>12} {:>15}".format(
                idx,
                buy_date_str,
                buy_price_str,
                sell_date_str,
                sell_price_display,
                profit_display,
                capital_display
            )
        )

    lines.append("-" * 100)

    final_capital = backtest_result.get('final_capital', initial_capital)
    total_return = backtest_result.get('total_return', 0.0)
    total_commission = backtest_result.get('total_commission', 0.0)
    gross_return = backtest_result.get('gross_return', total_return)
    trades_list = trades if trades else []
    if trades_list:
        avg_profit = sum(t.get('profit_rate', 0) for t in trades_list) / len(trades_list) * 100
    else:
        avg_profit = 0.0

    lines.append(f"初始资金: ¥{initial_capital:,.2f}")
    lines.append(f"最终资金: ¥{final_capital:,.2f} (已完成交易)")
    if signal_data and signal_data.get('signal') == 'HOLD_BUY':
        lines.append("当前持仓浮盈未计入最终资金")
    lines.append(f"累计收益率: {total_return:+.2f}%")
    lines.append(f"累计手续费: ¥{total_commission:,.2f}")
    lines.append(f"毛收益率: {gross_return:.2f}% (扣费前)")
    lines.append(f"平均单次收益: {avg_profit:+.2f}%")
    lines.append(f"盈利交易占比: {backtest_result.get('win_rate', 0):.1f}%")
    lines.append("=" * 80)

    return "\n".join(lines)


def _run_offline_backtest_task(stock_code: str, data_file_str: str, config: dict, initial_capital: float) -> Dict:
    """Top-level worker for parallel offline backtests."""
    data_file = Path(data_file_str)
    try:
        df_raw = pd.read_csv(data_file)
    except Exception as exc:
        return {'success': False, 'code': stock_code, 'error': f"读取离线回测数据失败: {exc}"}

    if df_raw.empty:
        return {'success': False, 'code': stock_code, 'error': "离线回测数据为空，跳过"}

    analysis = analyze_stock(
        stock_code,
        config,
        show_backtest=True,
        df_override=df_raw,
        quiet=True,
    )

    if not analysis:
        return {'success': False, 'code': stock_code, 'error': "分析失败，跳过"}

    backtest_result = analysis.get('backtest')
    if not backtest_result:
        return {'success': False, 'code': stock_code, 'error': "回测失败，跳过"}

    current_price = float(df_raw['close'].iloc[-1])
    stock_report = format_stock_report(
        stock_code,
        current_price,
        analysis.get('signal'),
        backtest_result,
        analysis.get('trading_signals')
    )

    curve = analysis.get('mark_to_market_curve') or {'dates': [], 'equity': []}
    if len(curve['dates']) >= 2:
        elapsed_years = max(
            (pd.Timestamp(curve['dates'][-1]) - pd.Timestamp(curve['dates'][0])).days / 365.25,
            1.0 / 365.25,
        )
        final_multiple = max(
            float(backtest_result.get('final_capital', initial_capital)) / initial_capital,
            1e-12,
        )
        stock_cagr = (final_multiple ** (1.0 / elapsed_years) - 1.0) * 100.0
    else:
        stock_cagr = 0.0

    prior_capital = float(initial_capital)
    amount_profit = 0.0
    amount_loss = 0.0
    for trade in backtest_result.get('trades', []):
        current_capital = float(trade.get('capital', prior_capital))
        change = current_capital - prior_capital
        if change > 0:
            amount_profit += change
        elif change < 0:
            amount_loss += abs(change)
        prior_capital = current_capital

    summary_entry = {
        'code': stock_code,
        'total_return': backtest_result.get('total_return', 0.0),
        'max_drawdown': backtest_result.get('max_drawdown', 0.0),
        'win_rate': backtest_result.get('win_rate', 0.0),
        'profit_factor': backtest_result.get('profit_factor', 0.0),
        'total_trades': backtest_result.get('total_trades', 0),
        'final_capital': backtest_result.get('final_capital', initial_capital),
        'total_profit_pct': backtest_result.get('total_profit_pct', 0.0),
        'total_loss_pct': backtest_result.get('total_loss_pct', 0.0),
        'win_count': backtest_result.get('win_count', 0),
        'lose_count': backtest_result.get('lose_count', 0),
        'stock_cagr': stock_cagr,
        'equity_dates': curve['dates'],
        'equity_curve': curve['equity'],
        'amount_profit': amount_profit,
        'amount_loss': amount_loss,
    }

    log_message = (
        f"✓ {stock_code} 完成 - 收益 {backtest_result.get('total_return', 0.0):.2f}% | "
        f"最大回撤 {backtest_result.get('max_drawdown', 0.0):.2f}% | "
        f"胜率 {backtest_result.get('win_rate', 0.0):.2f}% | "
        f"交易 {backtest_result.get('total_trades', 0)}"
    )

    return {
        'success': True,
        'code': stock_code,
        'report': stock_report,
        'summary': summary_entry,
        'log': log_message,
    }


def generate_offline_backtest_report(config: dict, stock_codes: Optional[List[str]] = None):
    """
    对离线回测数据目录中的所有股票执行回测并生成汇总报告
    """
    logger = logging.getLogger(__name__)
    data_dir = BACKTEST_DATA_DIR

    if not data_dir.exists():
        print(f"✗ 未找到离线回测数据目录: {data_dir}")
        return

    data_files = sorted(data_dir.glob(f'*_{BACKTEST_DATA_ADJUST}.csv'))
    if not data_files:
        print(f"✗ 离线回测数据目录 {data_dir} 中未找到 *_{BACKTEST_DATA_ADJUST}.csv，无法生成回测报告")
        return

    data_map = {}
    for file in data_files:
        code = file.stem.removesuffix(f'_{BACKTEST_DATA_ADJUST}')
        data_map[code] = file

    if stock_codes:
        normalized_codes = [normalize_stock_code(code) for code in stock_codes]
        target_files = []
        for raw, norm in zip(stock_codes, normalized_codes):
            target = data_map.get(norm)
            if not target:
                print(f"✗ 未找到股票 {raw} 的离线回测数据文件，已跳过")
                continue
            target_files.append((norm, target))
        if not target_files:
            print("✗ 未找到任何匹配的离线回测数据文件，无法生成报告")
            return
    else:
        target_files = [
            (file.stem.removesuffix(f'_{BACKTEST_DATA_ADJUST}'), file)
            for file in data_files
        ]

    backtest_config = config.get('backtest', {})
    initial_capital = backtest_config.get('initial_capital', 10000)

    report_config = config.get('report', {})
    report_verbose = bool(report_config.get('verbose', False))
    report_blas_threads = int(report_config.get('blas_threads', 1) or 1)

    executor_name = str(
        os.environ.get('STOCK_ADVISOR_REPORT_EXECUTOR')
        or report_config.get('executor')
        or 'process'
    ).strip().lower()
    executor_cls = ProcessPoolExecutor if executor_name in ('process', 'proc', 'processpool') else ThreadPoolExecutor

    total = len(target_files)
    env_workers = os.environ.get('STOCK_ADVISOR_REPORT_WORKERS')
    max_workers_config = env_workers if env_workers is not None else report_config.get('max_workers')
    if isinstance(max_workers_config, str) and max_workers_config.isdigit():
        max_workers = int(max_workers_config)
    elif isinstance(max_workers_config, Integral) and max_workers_config > 0:
        max_workers = int(max_workers_config)
    else:
        cpu_count = os.cpu_count() or 1
        max_workers = min(total, cpu_count)

    max_workers = max(1, min(max_workers, total))

    if executor_cls is ProcessPoolExecutor:
        for var in (
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ):
            os.environ.setdefault(var, str(report_blas_threads))
        if not report_verbose:
            logging.getLogger().setLevel(logging.WARNING)

    if stock_codes:
        print(f"对指定的 {total} 只股票生成回测报告（需存在离线回测数据）...")
    else:
        if max_workers <= 1:
            mode_label = "单线程"
        else:
            mode_label = "多进程" if executor_cls is ProcessPoolExecutor else "多线程"
        print(f"在离线回测数据目录中找到 {total} 只股票，开始{mode_label}离线回测...")
    if max_workers > 1:
        worker_label = "进程" if executor_cls is ProcessPoolExecutor else "线程"
        print(f"本次将使用 {max_workers} 个{worker_label}并行处理离线回测数据文件")

    summary = []
    failures = []
    stock_reports: Dict[str, str] = {}

    futures_map = {}
    with executor_cls(max_workers=max_workers) as executor:
        for idx, (stock_code, data_file) in enumerate(target_files, 1):
            if report_verbose:
                print(f"\n[{idx}/{total}] 回测 {stock_code} ...")
            future = executor.submit(
                _run_offline_backtest_task,
                stock_code,
                str(data_file),
                config,
                initial_capital,
            )
            futures_map[future] = stock_code

        for future in as_completed(futures_map):
            stock_code = futures_map[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error(f"处理 {stock_code} 时出现未捕获异常: {exc}", exc_info=True)
                print(f"✗ {stock_code} 执行异常: {exc}")
                failures.append(stock_code)
                continue

            if result.get('success'):
                summary.append(result['summary'])
                stock_reports[stock_code] = result['report']
                if report_verbose:
                    print(result.get('log', f"✓ {stock_code} 完成"))
            else:
                failures.append(stock_code)
                if report_verbose:
                    print(f"✗ {stock_code} {result.get('error', '未知错误')}")

    success_count = len(summary)
    if success_count == 0:
        print("\n✗ 回测失败，未生成任何有效结果")
        return

    summary.sort(key=lambda x: x['total_return'], reverse=True)

    header = "{:<8}{:>10}{:>12}{:>10}{:>10}{:>8}{:>14}".format(
        "代码", "收益%", "最大回撤%", "胜率%", "盈亏比", "交易数", "最终资金"
    )
    summary_header = "{:<10}{:>14}{:>14}{:>14}{:>14}{:>14}".format(
        "股票数量", "平均回撤%", "盈利股票%", "收益中位数%", "平均胜率%", "同公式盈亏比"
    )
    separator_line = "=" * 88
    dash_line = "-" * 88

    print("\n" + separator_line)
    print("📊 离线回测报告（按收益率排序）")
    print(separator_line)
    table_lines = [header, dash_line]

    for row in summary:
        pf = row['profit_factor']
        pf_str = f"{pf:.2f}" if pf < 100 else "99+"
        line = (
            "{:<10}{:>10.2f}{:>12.2f}{:>10.2f}{:>10}{:>8d}{:>14,.2f}".format(
                row['code'],
                row['total_return'],
                row['max_drawdown'],
                row['win_rate'],
                pf_str,
                row['total_trades'],
                row['final_capital'],
            )
        )
        table_lines.append(line)

    return_series = pd.Series([row['total_return'] for row in summary], dtype=float)
    drawdown_series = pd.Series([row['max_drawdown'] for row in summary], dtype=float)
    avg_total_return = float(return_series.mean())
    avg_max_drawdown = sum(row['max_drawdown'] for row in summary) / success_count
    avg_win_rate = sum(row['win_rate'] for row in summary) / success_count
    profitable_stock_rate = float((return_series > 0).mean() * 100)
    median_total_return = float(return_series.median())
    median_drawdown = float(drawdown_series.median())
    ordered_returns = return_series.sort_values().reset_index(drop=True)
    trim_count = int(success_count * 0.1)
    trimmed_returns = ordered_returns.iloc[trim_count:-trim_count] if trim_count else ordered_returns
    trimmed_mean_return = float(trimmed_returns.mean())
    p25_return = float(return_series.quantile(0.25))
    p75_return = float(return_series.quantile(0.75))
    positive_returns = return_series[return_series > 0].sort_values()
    top5_share = (
        float(positive_returns.iloc[-5:].sum() / positive_returns.sum() * 100)
        if not positive_returns.empty and positive_returns.sum() > 0
        else 0.0
    )
    # 总盈亏比：汇总所有股票的总盈利/总亏损
    all_profit = sum(row['total_profit_pct'] for row in summary)
    all_loss = sum(row['total_loss_pct'] for row in summary)
    total_profit_factor = all_profit / all_loss if all_loss > 0 else 99.0
    all_amount_profit = sum(row['amount_profit'] for row in summary)
    all_amount_loss = sum(row['amount_loss'] for row in summary)
    amount_profit_factor = all_amount_profit / all_amount_loss if all_amount_loss > 0 else 99.0
    avg_trades = sum(row['total_trades'] for row in summary) / success_count
    avg_final_capital = sum(row['final_capital'] for row in summary) / success_count
    total_wins = sum(row['win_count'] for row in summary)
    total_closed_trades = total_wins + sum(row['lose_count'] for row in summary)
    total_trade_win_rate = total_wins / total_closed_trades * 100 if total_closed_trades else 0.0

    equity_series = []
    for row in summary:
        if row['equity_dates'] and row['equity_curve']:
            equity_series.append(
                pd.Series(
                    row['equity_curve'],
                    index=pd.to_datetime(row['equity_dates']),
                    name=row['code'],
                    dtype=float,
                )
            )
    equity_frame = pd.concat(equity_series, axis=1).sort_index().ffill().fillna(1.0)
    portfolio_equity = equity_frame.mean(axis=1)
    portfolio_daily_returns = portfolio_equity.pct_change().dropna()
    portfolio_drawdown = float(
        (portfolio_equity / portfolio_equity.cummax() - 1.0).min() * 100.0
    )
    portfolio_years = max(
        (portfolio_equity.index[-1] - portfolio_equity.index[0]).days / 365.25,
        1.0 / 365.25,
    )
    portfolio_cagr = float(
        (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) ** (1.0 / portfolio_years)
        - 1.0
    ) * 100.0
    portfolio_daily_std = float(portfolio_daily_returns.std())
    portfolio_sharpe = (
        float(portfolio_daily_returns.mean()) / portfolio_daily_std * (252.0 ** 0.5)
        if portfolio_daily_std > 0
        else 0.0
    )
    median_stock_cagr = float(
        pd.Series([row['stock_cagr'] for row in summary], dtype=float).median()
    )

    total_pf_str = f"{total_profit_factor:.2f}" if total_profit_factor < 100 else "99+"
    summary_line = "{:<10}{:>14.2f}{:>14.2f}{:>14.2f}{:>14.2f}{:>14}".format(
        success_count,
        avg_max_drawdown,
        profitable_stock_rate,
        median_total_return,
        avg_win_rate,
        total_pf_str,
    )
    distribution_line = (
        f"分布辅助: 10%截尾均值 {trimmed_mean_return:.2f}% | "
        f"P25/P75 {p25_return:.2f}%/{p75_return:.2f}% | "
        f"平均收益 {avg_total_return:.2f}% | 前5赢家占正收益 {top5_share:.2f}%"
    )
    risk_line = (
        f"风险/交易辅助: 回撤中位数 {median_drawdown:.2f}% | "
        f"总交易胜率 {total_trade_win_rate:.2f}% | 平均交易数 {avg_trades:.2f} | "
        f"金额口径盈亏比 {amount_profit_factor:.2f} | 平均最终资金 {avg_final_capital:,.2f}"
    )
    portfolio_line = (
        f"每日盯市等权组合: CAGR {portfolio_cagr:.2f}% | 最大回撤 {portfolio_drawdown:.2f}% | "
        f"Sharpe {portfolio_sharpe:.2f} | 个股CAGR中位数 {median_stock_cagr:.2f}%"
    )

    table_lines.append("")
    table_lines.append(summary_header)
    table_lines.append(dash_line)
    table_lines.append(summary_line)
    table_lines.append(distribution_line)
    table_lines.append(risk_line)
    table_lines.append(portfolio_line)

    for line in table_lines:
        print(line)
    print(separator_line)

    print("\n处理完成！")
    print(f"成功: {success_count} 只，失败: {total - success_count} 只")
    if failures:
        failed_preview = ", ".join(failures[:10])
        suffix = "..." if len(failures) > 10 else ""
        print(f"失败列表（最多显示10只）: {failed_preview}{suffix}")

    # 写入报告文件
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    timestamp_slug = now.strftime('%Y%m%d_%H%M%S')
    report_lines = [
        f"离线回测报告 - 生成时间: {timestamp}",
        separator_line,
        "按收益率排序："
    ]
    report_lines.extend(table_lines)
    report_lines.append(separator_line)
    report_lines.append("")
    report_lines.append(f"成功: {success_count} 只，失败: {total - success_count} 只")
    if failures:
        failed_preview = ", ".join(failures[:10])
        suffix = "..." if len(failures) > 10 else ""
        report_lines.append(f"失败列表（最多显示10只）: {failed_preview}{suffix}")
    report_lines.append("\n=== 个股详细报告 ===")

    for row in summary:
        section = stock_reports.get(row['code'])
        if section:
            report_lines.append(section)
            report_lines.append("")

    report_dir = Path(__file__).parent / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f'offline_backtest_report_{timestamp_slug}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines).strip() + "\n")

    print(f"\n报告已保存到: {report_file}")


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
    parser.add_argument('--report', action='store_true',
                        help='离线模式：对 data/backtest_data 中所有或指定股票（-s/-b）进行回测并输出报告')
    parser.add_argument('--chart-generation', action='store_true',
                        help='回测后自动生成K线图并标注买卖点，图片保存到reports/')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    setup_logging(config)

    # 执行分析
    if args.report:
        if args.no_backtest:
            print("⚠️ --report 模式默认执行回测，将忽略 --no-backtest")
        report_codes = []
        if args.stock:
            report_codes.append(args.stock)
        if args.batch:
            report_codes.extend(args.batch)
        if report_codes:
            print(f"仅对指定股票生成离线报告: {', '.join(report_codes)}")
        else:
            print("未指定股票，将对离线回测数据目录中所有股票生成离线报告")
        generate_offline_backtest_report(
            config,
            stock_codes=report_codes if report_codes else None,
        )
    elif args.stock:
        analyze_stock(
            args.stock,
            config,
            show_backtest=not args.no_backtest,
            quiet=False,
            df_override=None,
            chart_generation=args.chart_generation,
        )
    elif args.batch:
        batch_analyze(args.batch, config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
