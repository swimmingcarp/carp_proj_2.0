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
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional
from numbers import Integral
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
PROJECT_ROOT = BASE_DIR.parent

# 确保 src 和项目根目录均在 sys.path 中
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher import DataFetcher
from src.strategy import MixedStrategy
from src.new_strategy import RSITrendStrategy
from src.oscillation_strategy import OscillationStrategy
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
        # 如果是相对路径，转换为相对于main.py的绝对路径
        from pathlib import Path
        config_file = Path(config_path)
        if not config_file.is_absolute():
            script_dir = Path(__file__).parent
            config_file = script_dir / config_path
        
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
    """标准化股票代码以匹配缓存文件"""
    if not code:
        return code
    code = code.strip()
    if '.' in code:
        code = code.split('.')[0]
    if len(code) > 2 and code[:2].lower() in ('sh', 'sz', 'hk'):
        code = code[2:]
    return code.upper()


def analyze_stock(stock_code: str, config: dict, show_backtest: bool = True,
                  use_fixed_strategy: bool = False, df_override: Optional[pd.DataFrame] = None,
                  quiet: bool = False, chart_generation: bool = False,
                  oscillation_driven: bool = False, use_new_strategy: bool = False) -> Optional[Dict]:
    """
    分析单只股票

    Args:
        stock_code: 股票代码
        config: 配置字典
        show_backtest: 是否显示回测结果
        chart_generation: 是否生成K线图
        use_new_strategy: 是否启用RSI趋势策略 (RSITrendStrategy)
    """
    logger = logging.getLogger(__name__)
    log_func = logger.info if not quiet else logger.debug
    log_func(f"开始分析股票: {stock_code}")

    def echo(message: str):
        if not quiet:
            print(message)

    # 1. 初始化组件
    data_config = config.get('data_source', {})
    strategy_config = config.get('strategy', {})
    backtest_config = config.get('backtest', {})

    fetcher = None
    validation_report = None

    if df_override is None:
        fetcher = DataFetcher(
            source=data_config.get('provider', app_config.DATA_SOURCE),
            cache_enabled=data_config.get('cache_enabled', app_config.CACHE_ENABLED),
            max_retries=app_config.MAX_RETRIES,
            retry_delay=app_config.RETRY_DELAY,
            is_backtest_mode=show_backtest
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

    # 根据参数选择策略类型
    if oscillation_driven:
        echo("使用震荡间隔交易策略 (OscillationStrategy)")
        strategy = OscillationStrategy(
            config=strategy_config,
            market=market
        )
    elif use_new_strategy:
        echo("使用RSI趋势策略 (RSITrendStrategy)")
        strategy = RSITrendStrategy(
            config=strategy_config,
            market=market,
            stock_code=stock_code
        )
    else:
        strategy = MixedStrategy(
            config=strategy_config,
            market=market,
            stock_code=stock_code,
            use_strategy_cache=show_backtest and use_fixed_strategy,
            oscillation_driven=False
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
        result = strategy.analyze(df)
        if isinstance(result, tuple):
            df_analyzed, indicator_report = result
        else:
            df_analyzed = result
            indicator_report = None
    except Exception as e:
        logger.error(f"策略分析失败: {e}", exc_info=True)
        echo(f"❌ 分析失败: {e}")
        return None

    if indicator_report:
        if indicator_report.get('status') == 'FAILED':
            echo(strategy.indicator_validator.format_report(indicator_report))
            echo("⚠️  技术指标存在异常，建议谨慎使用分析结果")
        elif indicator_report.get('status') == 'WARNING':
            echo(strategy.indicator_validator.format_report(indicator_report))
        else:
            indicators_str = ', '.join(indicator_report.get('indicators_checked', []))
            echo(f"✓ 技术指标验证通过 ({indicators_str})")

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

            if not quiet:
                echo(analyzer.format_backtest_result(backtest_result))
                if use_new_strategy:
                    timeline = format_trend_timeline(df, stock_code)
                    if timeline:
                        echo(timeline)
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

        oscillation_periods_for_plot = []
        # 使用震荡确认时间（逐日判断的结果，无前瞻性偏差）
        raw_periods = strategy.get_oscillation_confirmed_periods()

        if raw_periods:
            def _ensure_timestamp(value):
                if value is None:
                    return None
                try:
                    return pd.to_datetime(value)
                except Exception:
                    return None

            total_len = len(df_analyzed)
            has_date_col = 'date' in df_analyzed.columns

            for period in raw_periods:
                # 震荡确认时间格式: (confirmed_idx, end_idx, confirmed_date, end_date)
                if not isinstance(period, (list, tuple)) or len(period) < 4:
                    continue

                confirmed_idx, end_idx, confirmed_date, end_date = period[:4]
                # 使用确认时间作为起始点（而不是回溯的起始时间）
                start_ts = _ensure_timestamp(confirmed_date)
                end_ts = _ensure_timestamp(end_date)

                if has_date_col:
                    if start_ts is None and isinstance(confirmed_idx, Integral):
                        safe_start = max(0, min(total_len - 1, confirmed_idx))
                        start_ts = _ensure_timestamp(df_analyzed.iloc[safe_start]['date'])
                    if end_ts is None and isinstance(end_idx, Integral):
                        safe_end = max(0, min(total_len - 1, end_idx))
                        end_ts = _ensure_timestamp(df_analyzed.iloc[safe_end]['date'])
                else:
                    if start_ts is None and isinstance(confirmed_idx, Integral):
                        safe_start = max(0, min(total_len - 1, confirmed_idx))
                        start_ts = _ensure_timestamp(df_analyzed.index[safe_start])
                    if end_ts is None and isinstance(end_idx, Integral):
                        safe_end = max(0, min(total_len - 1, end_idx))
                        end_ts = _ensure_timestamp(df_analyzed.index[safe_end])

                if start_ts is None or end_ts is None:
                    continue

                # 计算趋势方向：确认点到结束点的价格变化
                trend = 'range'
                start_idx_for_trend = confirmed_idx if isinstance(confirmed_idx, Integral) else None
                end_idx_for_trend = end_idx if isinstance(end_idx, Integral) else None
                if (
                    'close' in df_analyzed.columns
                    and start_idx_for_trend is not None
                    and end_idx_for_trend is not None
                ):
                    try:
                        safe_start = max(0, min(total_len - 1, start_idx_for_trend))
                        safe_end = max(0, min(total_len - 1, end_idx_for_trend))
                        start_price = df_analyzed.iloc[safe_start]['close']
                        end_price = df_analyzed.iloc[safe_end]['close']
                        if start_price:
                            pct_change = (end_price - start_price) / start_price
                            if pct_change <= -0.02:
                                trend = 'decline'
                    except Exception:
                        pass

                oscillation_periods_for_plot.append({
                    'start': start_ts,  # 震荡确认时间
                    'end': end_ts,      # 震荡结束时间
                    'score': None,
                    'start_idx': confirmed_idx,
                    'end_idx': end_idx,
                    'trend': trend,
                    'type': 'confirmed'  # 标记为确认时间
                })

        try:
            out_path = plot_kline_with_signals(
                df_analyzed,
                buy_idx,
                sell_idx,
                stock_code,
                oscillation_periods=oscillation_periods_for_plot
            )
            echo(f"K线图已保存到: {out_path}")
        except Exception as e:
            echo(f"K线图生成失败: {e}")

    log_func(f"完成分析股票: {stock_code}")
    return {
        'signal': signal_data,
        'backtest': backtest_result,
        'trading_signals': trading_signals,
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


_STOCK_NAMES_CACHE: Optional[Dict[str, str]] = None


def _get_stock_name(stock_code: str) -> str:
    """从配置文件中查找股票名称"""
    global _STOCK_NAMES_CACHE
    if _STOCK_NAMES_CACHE is None:
        _STOCK_NAMES_CACHE = {}
        for fname in ('cn_stock_names.txt', 'hk_stock_names.txt'):
            fpath = BASE_DIR / 'config' / fname
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            _STOCK_NAMES_CACHE[parts[0]] = parts[1]
    return _STOCK_NAMES_CACHE.get(stock_code, '')


def format_trend_timeline(df_raw: pd.DataFrame, stock_code: str) -> str:
    """
    格式化趋势区间段落表

    利用 PIT 趋势检测模块，将股票的历史走势分段并展示时间线。
    """
    try:
        from personality.pit_stage import (
            compute_pit_states, make_personality_config,
            compute_stock_dna, compute_downtrend_phase, smooth_macro_phase
        )
        from personality.segmenter import StockPersonalityEngine
    except ImportError:
        return ""

    import numpy as np

    df = df_raw.copy()
    if 'date' not in df.columns:
        return ""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float) if 'volume' in df.columns else None
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    n = len(df)

    if n < 200:
        return ""

    try:
        dna = compute_stock_dna(close, high, low)
        pe = StockPersonalityEngine(close, dates)
        pers = pe.get_personality_at_bar(n - 1)
        cfg = make_personality_config(pers, dna=dna)

        states = compute_pit_states(close, high, low, cfg=cfg)
        phases, _ = compute_downtrend_phase(
            close, high, low, states, cfg=cfg, dna=dna, return_data=True
        )
        confirmed = smooth_macro_phase(
            phases, close, high, low, volume, confirmed_only=True
        )
    except Exception:
        return ""

    # ── 家族分类 ──
    _DN = {'downtrend_steep', 'downtrend_gradual', 'downtrend_range', 'downtrend_bottom'}
    STATE_SYM = {
        'consolidation':     '整理  ',
        'uptrend':           '上涨  ',
        'downtrend_steep':   '急跌↓↓',
        'downtrend_gradual': '缓跌↓ ',
        'downtrend_range':   '震荡↔ ',
        'downtrend_bottom':  '底部离',
    }

    def fam(ph):
        if ph in _DN or ph == 'pullback':
            return 'dn'
        if ph in ('uptrend', 'bounce'):
            return 'up'
        return 'co'

    # ── 构建 segments ──
    segs = []
    prev = confirmed[0]
    start = 0
    for i in range(1, n):
        if confirmed[i] != prev:
            sc = close[start:i]
            segs.append(dict(
                state=prev, start_date=dates[start], end_date=dates[i - 1],
                sp=float(close[start]), ep=float(close[i - 1]),
                lo=float(sc.min()), hi=float(sc.max()), bars=i - start,
            ))
            prev = confirmed[i]
            start = i
    sc = close[start:]
    segs.append(dict(
        state=prev, start_date=dates[start], end_date=dates[-1],
        sp=float(close[start]), ep=float(close[-1]),
        lo=float(sc.min()), hi=float(sc.max()), bars=n - start,
    ))

    # ── 格式化表格 ──
    import unicodedata

    def _vw(s):
        """Visual width accounting for CJK double-width characters."""
        return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1
                   for c in str(s))

    def _pad(s, width, align='left'):
        """Pad string to target visual width."""
        s = str(s)
        gap = max(0, width - _vw(s))
        return s + ' ' * gap if align == 'left' else ' ' * gap + s

    stock_name = _get_stock_name(stock_code)
    name_str = f" {stock_name}" if stock_name else ""

    def _row(num, state, period, days, sp, ep, chg, note):
        return (
            f" {_pad(num, 3, 'right')}"
            f"  {_pad(state, 8)}"
            f"{_pad(period, 25)}"
            f"  {_pad(days, 4, 'right')}"
            f"  {_pad(sp, 8, 'right')}"
            f"  {_pad(ep, 8, 'right')}"
            f"  {_pad(chg, 8, 'right')}"
            f"  {note}"
        )

    lines = []
    W = 100
    lines.append("━━━ 趋势区间段落表 ━━━")
    lines.append(f"{stock_code}{name_str} — 趋势时间线（共 {len(segs)} 段）")
    lines.append("=" * W)
    lines.append(_row('#', '状态', '区间', '天数', '起始价', '结束价', '涨跌幅', '备注'))
    lines.append("-" * W)

    for idx, seg in enumerate(segs):
        chg = (seg['ep'] - seg['sp']) / seg['sp'] * 100
        sym = STATE_SYM.get(seg['state'], seg['state']).rstrip()
        date_range = f"{seg['start_date']} ~ {seg['end_date']}"
        chg_s = f"{chg:+.1f}%"
        remark = f"低:{seg['lo']:.2f} 高:{seg['hi']:.2f}"

        lines.append(_row(
            str(idx + 1), sym, date_range,
            str(seg['bars']), f"{seg['sp']:.2f}", f"{seg['ep']:.2f}",
            chg_s, remark
        ))

    lines.append("=" * W)
    return "\n".join(lines)


def generate_cache_backtest_report(config: dict, use_fixed_strategy: bool = False,
                                   stock_codes: Optional[List[str]] = None,
                                   oscillation_driven: bool = False,
                                   use_new_strategy: bool = False):
    """
    对缓存中的所有股票执行回测并生成汇总报告
    """
    logger = logging.getLogger(__name__)
    cache_dir = Path(__file__).parent / 'data' / 'cache'

    if not cache_dir.exists():
        print(f"✗ 未找到缓存目录: {cache_dir}")
        return

    cache_files = sorted(cache_dir.glob('*.csv'))
    if not cache_files:
        print(f"✗ 缓存目录 {cache_dir} 为空，无法生成回测报告")
        return

    cache_map = {}
    for file in cache_files:
        code = file.stem.split('_')[0]
        cache_map[code] = file

    if stock_codes:
        normalized_codes = [normalize_stock_code(code) for code in stock_codes]
        target_files = []
        for raw, norm in zip(stock_codes, normalized_codes):
            target = cache_map.get(norm)
            if not target:
                print(f"✗ 未找到股票 {raw} 的缓存文件，已跳过")
                continue
            target_files.append((norm, target))
        if not target_files:
            print("✗ 未找到任何匹配的缓存文件，无法生成报告")
            return
    else:
        target_files = [(file.stem.split('_')[0], file) for file in cache_files]

    backtest_config = config.get('backtest', {})
    initial_capital = backtest_config.get('initial_capital', 10000)

    report_config = config.get('report', {})
    max_workers_config = report_config.get('max_workers')
    if isinstance(max_workers_config, Integral) and max_workers_config > 0:
        max_workers = int(max_workers_config)
    else:
        cpu_count = os.cpu_count() or 1
        max_workers = cpu_count

    total = len(target_files)
    max_workers = max(1, min(max_workers, total))

    if stock_codes:
        print(f"对指定的 {total} 只股票生成回测报告（需存在缓存）...")
    else:
        mode_label = "多线程" if max_workers > 1 else "单线程"
        print(f"在缓存目录中找到 {total} 只股票，开始{mode_label}离线回测...")
    if max_workers > 1:
        print(f"本次将使用 {max_workers} 个线程并行处理缓存文件")

    summary = []
    failures = []
    stock_reports: Dict[str, str] = {}

    def run_single_backtest(stock_code: str, cache_file: Path) -> Dict:
        try:
            df_raw = pd.read_csv(cache_file)
        except Exception as exc:
            logger.error(f"读取缓存 {cache_file} 失败: {exc}")
            return {'success': False, 'code': stock_code, 'error': f"读取缓存失败: {exc}"}

        if df_raw.empty:
            return {'success': False, 'code': stock_code, 'error': "缓存数据为空，跳过"}

        analysis = analyze_stock(
            stock_code,
            config,
            show_backtest=True,
            use_fixed_strategy=use_fixed_strategy,
            df_override=df_raw,
            quiet=True,
            oscillation_driven=oscillation_driven,
            use_new_strategy=use_new_strategy
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

        # 趋势区间段落表（仅 --new-strategy 模式）
        if use_new_strategy:
            timeline = format_trend_timeline(df_raw, stock_code)
            if timeline:
                stock_report = stock_report + "\n\n" + timeline

        summary_entry = {
            'code': stock_code,
            'total_return': backtest_result.get('total_return', 0.0),
            'max_drawdown': backtest_result.get('max_drawdown', 0.0),
            'win_rate': backtest_result.get('win_rate', 0.0),
            'profit_factor': backtest_result.get('profit_factor', 0.0),
            'total_trades': backtest_result.get('total_trades', 0),
            'final_capital': backtest_result.get('final_capital', initial_capital),
            # 用于汇总计算盈亏比
            'total_profit_pct': backtest_result.get('total_profit_pct', 0.0),
            'total_loss_pct': backtest_result.get('total_loss_pct', 0.0),
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

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (stock_code, cache_file) in enumerate(target_files, 1):
            print(f"\n[{idx}/{total}] 回测 {stock_code} ...")
            future = executor.submit(run_single_backtest, stock_code, cache_file)
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
                print(result.get('log', f"✓ {stock_code} 完成"))
            else:
                failures.append(stock_code)
                print(f"✗ {stock_code} {result.get('error', '未知错误')}")

    success_count = len(summary)
    if success_count == 0:
        print("\n✗ 回测失败，未生成任何有效结果")
        return

    summary.sort(key=lambda x: x['total_return'], reverse=True)

    header = "{:<8}{:>10}{:>12}{:>10}{:>10}{:>8}{:>14}".format(
        "代码", "收益%", "最大回撤%", "胜率%", "盈亏比", "交易数", "最终资金"
    )
    summary_header = "{:<8}{:>10}{:>12}{:>10}{:>10}{:>10}{:>8}{:>14}".format(
        "股票数量", "平均收益%", "平均回撤%", "平均胜率%", "总盈亏比", "平均盈亏比", "交易数", "最终资金"
    )
    separator_line = "=" * 88
    dash_line = "-" * 88

    print("\n" + separator_line)
    print("📊 缓存回测报告（按收益率排序）")
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

    avg_total_return = sum(row['total_return'] for row in summary) / success_count
    avg_max_drawdown = sum(row['max_drawdown'] for row in summary) / success_count
    avg_win_rate = sum(row['win_rate'] for row in summary) / success_count
    # 总盈亏比：汇总所有股票的总盈利/总亏损
    all_profit = sum(row['total_profit_pct'] for row in summary)
    all_loss = sum(row['total_loss_pct'] for row in summary)
    total_profit_factor = all_profit / all_loss if all_loss > 0 else 99.0
    # 平均盈亏比：各股票盈亏比的平均值（排除无亏损的极端值）
    valid_pfs = [row['profit_factor'] for row in summary if row['profit_factor'] < 100]
    avg_profit_factor = sum(valid_pfs) / len(valid_pfs) if valid_pfs else 99.0
    avg_trades = sum(row['total_trades'] for row in summary) / success_count
    avg_final_capital = sum(row['final_capital'] for row in summary) / success_count

    total_pf_str = f"{total_profit_factor:.2f}" if total_profit_factor < 100 else "99+"
    avg_pf_str = f"{avg_profit_factor:.2f}" if avg_profit_factor < 100 else "99+"
    summary_line = "{:<10}{:>14.2f}{:>16.2f}{:>14.2f}{:>14}{:>14}{:>14.2f}{:>17,.2f}".format(
        success_count,
        avg_total_return,
        avg_max_drawdown,
        avg_win_rate,
        total_pf_str,
        avg_pf_str,
        avg_trades,
        avg_final_capital,
    )

    table_lines.append("")
    table_lines.append(summary_header)
    table_lines.append(dash_line)
    table_lines.append(summary_line)

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
        f"缓存回测报告 - 生成时间: {timestamp}",
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
    report_file = report_dir / f'cache_backtest_report_{timestamp_slug}.txt'
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
    parser.add_argument('--fixed-strategy', action='store_true',
                        help='回测模式下使用缓存的最优组合（原始/渐进 + 高频/高质量），'
                             '若无缓存则自动评估并写入缓存')
    parser.add_argument('--oscillation-driven', action='store_true',
                        help='使用震荡间隔交易策略（只在震荡区间外的正常行情中交易一笔）')
    parser.add_argument('--no-backtest', action='store_true', help='不显示回测结果')
    parser.add_argument('--new-strategy', action='store_true',
                        help='启用RSI趋势策略 (RSITrendStrategy)，替代默认混合策略')
    parser.add_argument('--report', action='store_true',
                        help='离线模式：对缓存中所有或指定股票（-s/-b）进行回测并输出报告')
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
            print("未指定股票，将对缓存中所有股票生成离线报告")
        generate_cache_backtest_report(
            config,
            use_fixed_strategy=args.fixed_strategy,
            stock_codes=report_codes if report_codes else None,
            oscillation_driven=args.oscillation_driven,
            use_new_strategy=args.new_strategy
        )
    elif args.stock:
        analyze_stock(
            args.stock,
            config,
            show_backtest=not args.no_backtest,
            use_fixed_strategy=args.fixed_strategy,
            quiet=False,
            df_override=None,
            chart_generation=args.chart_generation,
            oscillation_driven=args.oscillation_driven,
            use_new_strategy=args.new_strategy
        )
    elif args.batch:
        batch_analyze(args.batch, config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
