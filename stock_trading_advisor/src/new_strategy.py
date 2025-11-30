"""
RSI Trend Strategy
Based on https://github.com/zchancode/trading/blob/main/RSI趋势策略.tv

This module recreates the TradingView RSI Cross strategy inside the Python
backtest engine. The logic follows the original Pine Script:
    - dual RSI cross (fast RSI over slow RSI) triggers entries
    - ATR-based trailing stop controls the underlying trend direction
    - Heikin Ashi candles are used as a confirmation filter
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .indicators import (
        atr_indicator,
        ma_indicator,
        ema_indicator,
        rsi_indicator,
        macd_indicator,
    )
    from .strategy import MixedStrategy
except ImportError:  # pragma: no cover - fallback for standalone usage
    from indicators import atr_indicator, rsi_indicator, ma_indicator, ema_indicator, macd_indicator
    from strategy import MixedStrategy

logger = logging.getLogger(__name__)


class RSITrendStrategy(MixedStrategy):
    """
    RSI趋势策略

    特征:
        1. 使用 fast RSI (默认25) 与 slow RSI (默认100) 的金叉作为触发信号
        2. ATR × Multiplier 组成的动态止损决定多空方向（等价于SuperTrend思想）
        3. Heikin Ashi 阳线作为额外的趋势确认
    """

    def __init__(self, config: Optional[Dict] = None, market: str = 'CN-A',
                 stock_code: str = ''):
        defaults = {
            'trend_rsi_fast_period': 25,
            'trend_rsi_slow_period': 100,
            'trend_atr_period': 20,
            'trend_atr_multiplier': 3.0,
            'trend_use_close_for_extrema': True,
            'trend_relaxed_entry': True,
            'trend_relaxed_min_gap': 1.0,

            'trend_stop_loss_pct': 7.0,  # ATR趋势判断止损（恢复原版）
            'trend_exit_use_ma_filter': True,
            'trend_exit_fast_ema_period': 16,
            'trend_exit_slow_ma_period': 45,
            'trend_exit_confirm_ma_period': 16,
            'trend_lr_filter_enabled': True,
            'trend_lr_lookback': 30,
            'trend_lr_max_slope_pct': 1.5,
            # 多时间框架配置
            'trend_mtf_enabled': True,
            'trend_mtf_ratio': 5,  # 5倍周期作为更高时间框架
            'trend_mtf_min_periods': 50,  # 更高时间框架最少需要的数据点
            'trend_mtf_adaptive_mode': True,  # 自适应模式：上升用早期，下跌用严格
            'trend_mtf_early_entry': True,  # 启用早期入场模式
            'trend_mtf_early_threshold': 0.7,  # 早期入场RSI阈值(0-1)
            'trend_mtf_strict_mode': False,  # 严格模式：必须高时间框架完全确认
            'trend_mtf_trend_lookback': 20,  # 判断趋势状态的回溯周期

            # 主升浪持仓优化配置（优化后的参数）
            'trend_main_wave_enabled': True,  # 启用主升浪检测
            'trend_main_wave_min_gain': 10.0,  # 主升浪最小涨幅阈值(%) - 降低
            'trend_main_wave_min_days': 3,  # 主升浪最小持续天数 - 降低
            'trend_main_wave_rsi_threshold': 80,  # 主升浪期间RSI阈值 - 提高
            'trend_main_wave_volume_factor': 1.2,  # 主升浪成交量放大倍数 - 降低
            'trend_main_wave_hold_extension': True,  # 主升浪延长持仓
            
            # 底背离策略配置（买入信号）
            'trend_bullish_divergence_enabled': True,  # 启用底背离检测
            'trend_divergence_lookback': 30,  # 底背离检测回溯周期
            'trend_divergence_min_consecutive': 2,  # 连续底背离最小次数
            'trend_divergence_min_hold_days': 10,  # 底背离买入后的最短持有天数
            'trend_divergence_profit_target': 15.0,  # 底背离买入的止盈目标(%)
            'trend_divergence_ignore_rsi_exit': False,  # 底背离买入是否忽略RSI退出信号
            'trend_divergence_use_rsi_trend': False,  # 底背离买入使用RSI趋势判断（恢复原版，关闭优化）
            'trend_divergence_rsi_decline_threshold': -5.0,  # RSI相对下降阈值（负数表示下降）
        }
        if config:
            defaults.update(config)

        super().__init__(
            config=defaults,
            validate_indicators=False,
            sell_strategy='original',
            order='high_frequency',
            market=market,
            use_simple_divergence=False,
            adaptive_oscillation=False,
            stock_code=stock_code,
            precomputed_indicators=False,
            use_strategy_cache=False,
            oscillation_driven=False
        )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def analyze(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """执行RSI趋势策略分析"""
        if df is None or len(df) == 0:
            logger.warning("RSITrendStrategy: 数据为空，无法分析")
            return None, None

        data = self._prepare_dataframe(df)
        fast_period = int(self.config.get('trend_rsi_fast_period', 25))
        slow_period = int(self.config.get('trend_rsi_slow_period', 100))
        atr_period = int(self.config.get('trend_atr_period', 20))
        atr_multiplier = float(self.config.get('trend_atr_multiplier', 3.0))
        use_close = bool(self.config.get('trend_use_close_for_extrema', True))
        exit_ma_filter_enabled = bool(self.config.get('trend_exit_use_ma_filter', True))
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))
        exit_fast_ema_period = max(1, int(self.config.get('trend_exit_fast_ema_period', 16)))
        exit_slow_ma_period = max(1, int(self.config.get('trend_exit_slow_ma_period', 45)))
        exit_confirm_ma_period = max(
            1,
            int(self.config.get('trend_exit_confirm_ma_period', exit_fast_ema_period))
        )
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))
        lr_lookback = max(5, int(self.config.get('trend_lr_lookback', 30)))
        lr_max_slope_pct = max(0.0, float(self.config.get('trend_lr_max_slope_pct', 2.0)))

        data['fast_rsi'] = rsi_indicator(data['close'], period=fast_period)
        data['slow_rsi'] = rsi_indicator(data['close'], period=slow_period)

        # 计算MACD指标（用于底背离检测）
        macd_diff, macd_dea, macd_hist = macd_indicator(data['close'])
        data['macd_diff'] = macd_diff
        data['macd_dea'] = macd_dea
        data['macd_hist'] = macd_hist

        atr_values = atr_indicator(data, period=atr_period) * atr_multiplier
        data['atr_trailing'] = atr_values

        long_stop, short_stop, direction = self._compute_trend_levels(
            data,
            atr_values,
            atr_period,
            use_close
        )
        data['long_stop'] = long_stop
        data['short_stop'] = short_stop
        data['trend_direction'] = direction

        ha_open, ha_close, ha_high, ha_low = self._calculate_heikin_ashi(data)
        data['ha_open'] = ha_open
        data['ha_close'] = ha_close
        data['ha_high'] = ha_high
        data['ha_low'] = ha_low
        data['is_heikin_bullish'] = ha_close > ha_open

        data['golden_cross'] = self._crossover(data['fast_rsi'], data['slow_rsi'])
        data['death_cross'] = self._crossunder(data['fast_rsi'], data['slow_rsi'])
        data['rsi_diff'] = data['fast_rsi'] - data['slow_rsi']

        relaxed_enabled = bool(self.config.get('trend_relaxed_entry', True))
        relaxed_gap = max(0.0, float(self.config.get('trend_relaxed_min_gap', 1.0)))
        if relaxed_enabled:
            rsi_relaxed_condition = (data['rsi_diff'] >= relaxed_gap)
        else:
            rsi_relaxed_condition = pd.Series(False, index=data.index)
        data['rsi_relaxed_condition'] = rsi_relaxed_condition

        if lr_filter_enabled:
            def _lr_slope(arr: np.ndarray) -> float:
                if len(arr) == 0 or np.isnan(arr).any():
                    return np.nan
                x = np.arange(len(arr), dtype=float)
                return np.polyfit(x, arr, 1)[0]

            def _lr_zscore(arr: np.ndarray) -> float:
                if len(arr) == 0 or np.isnan(arr).any():
                    return np.nan
                x = np.arange(len(arr), dtype=float)
                slope, intercept = np.polyfit(x, arr, 1)
                fitted = slope * x + intercept
                residuals = arr - fitted
                std = np.std(residuals, ddof=1)
                if std == 0 or np.isnan(std):
                    return 0.0
                last_idx = len(arr) - 1
                last_fit = slope * last_idx + intercept
                return (arr[-1] - last_fit) / std

            lr_slope = data['close'].rolling(
                lr_lookback,
                min_periods=lr_lookback
            ).apply(_lr_slope, raw=True)
            steep_down = (lr_slope < 0) & (lr_slope.abs() > lr_max_slope_pct)
            lr_filter_condition = (~steep_down)

            data['lr_slope'] = lr_slope
            data['lr_steep_down'] = steep_down.fillna(False)
            data['lr_filter_condition'] = lr_filter_condition.fillna(False)
        else:
            nan_series = pd.Series(np.nan, index=data.index)
            data['lr_slope'] = nan_series.copy()
            data['lr_steep_down'] = pd.Series(False, index=data.index)
            lr_filter_condition = pd.Series(True, index=data.index)
            data['lr_filter_condition'] = lr_filter_condition

        strong_ma_uptrend = pd.Series(False, index=data.index)
        close_below_confirm = pd.Series(False, index=data.index)
        if exit_ma_filter_enabled:
            exit_ema_fast = ema_indicator(data['close'], period=exit_fast_ema_period)
            exit_ma_slow = ma_indicator(data['close'], period=exit_slow_ma_period)
            exit_ma_confirm = ma_indicator(data['close'], period=exit_confirm_ma_period)
            data['exit_ema_fast'] = exit_ema_fast
            data['exit_ma_slow'] = exit_ma_slow
            data['exit_ma_confirm'] = exit_ma_confirm
            strong_ma_uptrend = exit_ema_fast > exit_ma_slow
            close_below_confirm = (data['close'] < exit_ma_confirm).fillna(False)
            close_below_arr = close_below_confirm.to_numpy(dtype=bool, copy=True)
            streak_arr = np.zeros(len(close_below_arr), dtype=int)
            if len(streak_arr) > 0 and close_below_arr[0]:
                streak_arr[0] = 1
            for i in range(1, len(streak_arr)):
                if close_below_arr[i]:
                    streak_arr[i] = streak_arr[i - 1] + 1
                else:
                    streak_arr[i] = 0
            streak_series = pd.Series(streak_arr, index=data.index)
            ma_filter_break = strong_ma_uptrend & (streak_series >= 3)
            data['exit_close_below_ma_confirm'] = close_below_confirm
            data['exit_ma_confirm_below_streak'] = streak_series
            data['exit_ma_filter_break'] = ma_filter_break
            data['exit_strong_ma_trend'] = strong_ma_uptrend
            data['exit_ma_filter_active'] = strong_ma_uptrend & (streak_series < 3)
        else:
            nan_series = pd.Series(np.nan, index=data.index)
            data['exit_ema_fast'] = nan_series.copy()
            data['exit_ma_slow'] = nan_series.copy()
            data['exit_ma_confirm'] = nan_series.copy()
            data['exit_close_below_ma_confirm'] = pd.Series(False, index=data.index)
            data['exit_ma_confirm_below_streak'] = pd.Series(0, index=data.index)
            data['exit_ma_filter_break'] = pd.Series(False, index=data.index)
            data['exit_strong_ma_trend'] = pd.Series(False, index=data.index)
            data['exit_ma_filter_active'] = pd.Series(False, index=data.index)



        # 多时间框架趋势确认
        htf_bias, htf_info = self._compute_higher_timeframe_bias(data)
        data['mtf_bias'] = htf_bias
        data['mtf_info'] = str(htf_info)  # 存储诊断信息

        stop_loss_pct = max(0.0, float(self.config.get('trend_stop_loss_pct', 7.0)))



        # 底背离检测（买入信号）
        bullish_divergence_enabled = bool(self.config.get('trend_bullish_divergence_enabled', True))
        if bullish_divergence_enabled:
            bullish_divergence_signals = self._detect_bullish_divergence(data)
            data['bullish_divergence_signal'] = bullish_divergence_signals
        else:
            bullish_divergence_signals = pd.Series(False, index=data.index)
            data['bullish_divergence_signal'] = bullish_divergence_signals

        # 主升浪检测（需要在所有技术指标计算完成后进行）
        main_wave_enabled = bool(self.config.get('trend_main_wave_enabled', True))
        if main_wave_enabled:
            main_wave_signals = self._detect_main_wave_signals(data)
            data['main_wave_signal'] = main_wave_signals
        else:
            main_wave_signals = pd.Series(False, index=data.index)
            data['main_wave_signal'] = main_wave_signals

        # 入场条件：原有条件 或 底背离信号
        standard_entry = (
            (direction == 1) &
            data['is_heikin_bullish'] &
            (data['golden_cross'] | rsi_relaxed_condition) &
            lr_filter_condition &
            htf_bias  # 添加多时间框架确认
        )
        
        # 底背离入场条件（独立生效，不需要其他确认）
        divergence_entry = pd.Series(False, index=data.index)
        if bullish_divergence_enabled:
            # 底背离信号独立生效，与其他指标相互独立
            divergence_entry = bullish_divergence_signals
        
        entry_condition = standard_entry | divergence_entry
        
        # 底背离买入保护：标记底背离买入，用于后续退出逻辑
        data['divergence_entry'] = divergence_entry
        
        # 基础退出条件
        basic_exit_condition = (direction != 1)
        if exit_ma_filter_enabled:
            basic_exit_condition = basic_exit_condition & (
                ~strong_ma_uptrend | data['exit_ma_filter_break']
            )
        
        # 主升浪优化：在主升浪期间抑制卖出
        if main_wave_enabled:
            # 在主升浪期间，只有更强的退出信号才能卖出
            main_wave_exit_suppression = main_wave_signals
            
            # 特殊情况：MA多头排列时的主升浪延长保护
            ma_bullish_protection = pd.Series(False, index=data.index)
            if 'exit_ema_fast' in data.columns and 'exit_ma_slow' in data.columns:
                # MA多头排列 + 价格未大幅下跌时延长保护
                ma_bullish = data['exit_ema_fast'] > data['exit_ma_slow']
                price_not_crashed = data['close'] > data['exit_ma_slow'] * 0.90  # 价格在MA45的90%以上
                recent_main_wave = main_wave_signals.rolling(window=5, min_periods=1).sum() > 0  # 最近5天有主升浪
                
                ma_bullish_protection = ma_bullish & price_not_crashed & recent_main_wave
            
            # 综合的主升浪保护：当前主升浪 + MA多头排列保护
            comprehensive_protection = main_wave_exit_suppression | ma_bullish_protection
            
            # 主升浪期间的退出条件更严格
            exit_condition = basic_exit_condition & (~comprehensive_protection)
        else:
            exit_condition = basic_exit_condition

        # 底背离买入需要特殊的退出处理
        position, entry_flags, exit_flags, stop_loss_flags, profit_target_flags = self._build_position_series_with_divergence(
            entry_condition,
            exit_condition,
            divergence_entry,
            data['close'],
            stop_loss_pct,
            data  # 传入完整数据用于MA计算
        )

        data['buy_signal'] = position
        data['entry_signal'] = entry_flags
        data['exit_signal'] = exit_flags
        data['stop_loss_exit'] = stop_loss_flags
        data['profit_target_exit'] = profit_target_flags  # 添加止盈标记
        data['stop_loss_pct'] = stop_loss_pct if stop_loss_pct > 0 else np.nan
        data['entry_reason'] = self._build_entry_reasons(data, entry_flags)
        data['exit_reason'] = self._build_exit_reasons(data, exit_flags)

        return data, None

    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """输出最新的策略信号"""
        if df is None or len(df) == 0:
            return {'signal': 'NO_DATA', 'reason': '数据不足'}

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        exit_ma_filter_enabled = bool(self.config.get('trend_exit_use_ma_filter', True))
        lr_filter_enabled = bool(self.config.get('trend_lr_filter_enabled', True))

        signal = 'HOLD'
        reasons: List[str] = []
        strength = 1

        stop_loss_pct = float(self.config.get('trend_stop_loss_pct', 7.0))

        if latest.get('entry_signal', 0) == 1:
            signal = 'BUY'
            
            # 检查是否为底背离入场
            if latest.get('bullish_divergence_signal', False):
                reasons.append("检测到连续底背离信号（独立生效）")
                strength = 4  # 底背离信号强度较高
            # 标准RSI入场
            else:
                if latest.get('golden_cross'):
                    reasons.append("RSI快线金叉慢线")
                elif latest.get('rsi_relaxed_condition'):
                    reasons.append("RSI保持在慢线之上（放宽入场）")
                else:
                    reasons.append("RSI多头信号")
                if latest.get('is_heikin_bullish'):
                    reasons.append("Heikin Ashi 阳线确认")
                if latest.get('trend_direction') == 1:
                    reasons.append("ATR趋势仍为多头")
                if latest.get('mtf_bias', True):
                    # 根据多时间框架模式显示不同信息
                    mtf_info_str = latest.get('mtf_info', '{}')
                    try:
                        import ast
                        mtf_info = ast.literal_eval(mtf_info_str) if isinstance(mtf_info_str, str) else {}
                    except:
                        mtf_info = {}
                        
                    adaptive_mode = mtf_info.get('adaptive_mode', False)
                    if adaptive_mode:
                        latest_trend = mtf_info.get('latest_market_trend', 'neutral')
                        latest_mode = mtf_info.get('latest_mode_used', 'standard')
                        mode_dist = mtf_info.get('mode_distribution', {})
                        
                        if latest_trend == 'bullish' and latest_mode == 'early':
                            reasons.append("当前上升阶段，高时间框架早期确认")
                        elif latest_trend == 'bearish' and latest_mode == 'strict':
                            reasons.append("当前下跌阶段，高时间框架严格确认")
                        elif latest_trend == 'neutral':
                            reasons.append("当前震荡阶段，高时间框架标准确认")
                        else:
                            reasons.append(f"高时间框架{latest_mode}确认")
                            
                        # 如果有模式分布信息，可以加入更多细节
                        if len(mode_dist) > 1:
                            dominant_mode = max(mode_dist.items(), key=lambda x: x[1])[0]
                            if dominant_mode != latest_mode:
                                reasons.append(f"(历史以{dominant_mode}模式为主)")
                    else:
                        mode = mtf_info.get('latest_mode_used', mtf_info.get('mode_used', 'standard'))
                        if mode == 'early':
                            reasons.append("高时间框架早期确认")
                        elif mode == 'strict':
                            reasons.append("高时间框架严格确认")
                        else:
                            reasons.append("高时间框架趋势一致")
                else:
                    reasons.append("高时间框架趋势不一致")
                strength = 2 + int(latest.get('is_heikin_bullish', False)) + int(latest.get('trend_direction', 0) == 1) + int(latest.get('mtf_bias', True))

        elif latest.get('exit_signal', 0) == 1 or (
            previous.get('buy_signal', 0) == 1 and latest.get('buy_signal', 0) == 0
        ):
            signal = 'SELL'
            if latest.get('stop_loss_exit'):
                reasons.append(f"触发{stop_loss_pct:.1f}%止损")
            else:
                reasons.append("ATR趋势转空，触发止盈/止损")
                if latest.get('death_cross'):
                    reasons.append("RSI死叉确认转弱")
                if exit_ma_filter_enabled and latest.get('exit_ma_filter_break'):
                    reasons.append("EMA16>MA45但已连续3日跌破MA16")
            strength = 3 + int(latest.get('death_cross', False)) + int(latest.get('stop_loss_exit', False))

        elif latest.get('buy_signal', 0) == 1:
            signal = 'HOLD_BUY'
            reasons.append("多头持仓中")
            if latest.get('trend_direction') == 1:
                reasons.append("ATR多头趋势未破坏")
            if latest.get('is_heikin_bullish'):
                reasons.append("Heikin Ashi 保持阳线")
            if latest.get('fast_rsi', 0) > latest.get('slow_rsi', 0):
                reasons.append("RSI保持强势")
            if exit_ma_filter_enabled and latest.get('exit_ma_filter_active'):
                reasons.append("EMA16>MA45，均线保护持仓")
            strength = min(5, 1 + int(latest.get('trend_direction', 0) == 1) + int(latest.get('is_heikin_bullish', False)))
        else:
            reasons.append("等待下一次RSI金叉")
            if lr_filter_enabled and not bool(latest.get('lr_filter_condition', True)):
                reasons.append("线性回归趋势/偏离未满足，暂缓抄底")
            if not latest.get('mtf_bias', True):
                # 根据模式给出不同建议
                mtf_info_str = latest.get('mtf_info', '{}')
                try:
                    import ast
                    mtf_info = ast.literal_eval(mtf_info_str) if isinstance(mtf_info_str, str) else {}
                except:
                    mtf_info = {}
                    
                adaptive_mode = mtf_info.get('adaptive_mode', False)
                if adaptive_mode:
                    latest_trend = mtf_info.get('latest_market_trend', 'neutral')
                    if latest_trend == 'bearish':
                        reasons.append("当前下跌阶段，高时间框架严格过滤")
                    elif latest_trend == 'bullish':
                        reasons.append("当前上升阶段但高时间框架未确认")
                    else:
                        reasons.append("当前震荡阶段，早期入场模式待确认")
                else:
                    mode = mtf_info.get('latest_mode_used', mtf_info.get('mode_used', 'standard'))
                    if mode == 'strict':
                        reasons.append("高时间框架趋势偏弱，严格模式暂缓买入")
                    elif mode == 'early_neutral':
                        reasons.append("震荡市早期入场模式待确认")
                    else:
                        reasons.append("高时间框架趋势偏弱，暂缓买入")
            strength = 1

        return {
            'signal': signal,
            'strength': min(strength, 5),
            'reason': '；'.join(reasons) if reasons else '无明确信号',
            'price': latest.get('close', 0),
            'date': latest.get('date'),
            'fast_rsi': latest.get('fast_rsi'),
            'slow_rsi': latest.get('slow_rsi'),
            'trend_direction': latest.get('trend_direction'),
            'ha_bullish': latest.get('is_heikin_bullish'),
            'mtf_bias': latest.get('mtf_bias', True),  # 多时间框架偏向
            'bullish_divergence': latest.get('bullish_divergence_signal', False),  # 底背离信号
            # 兼容 signal analyzer 的字段
            'k': latest.get('fast_rsi', 0),
            'd': latest.get('slow_rsi', 0),
            'macd': latest.get('macd_hist', 0.0),
            'diff': latest.get('macd_diff', 0.0),
            'ema_16': latest.get('exit_ema_fast', np.nan),
            'ma_16': latest.get('exit_ma_confirm', np.nan),
            'ma_45': latest.get('exit_ma_slow', np.nan),
            'lr_slope': latest.get('lr_slope', np.nan),
            'lr_filter_ok': latest.get('lr_filter_condition', True),
        }

    def get_trading_signals(self, df: pd.DataFrame,
                            initial_capital: float = 10000.0) -> Dict:
        """返回可视化买卖点"""
        base_result = {
            'buy_points': [],
            'sell_points': [],
            'total_trades': 0,
            'initial_capital': initial_capital,
        }

        if df is None or 'buy_signal' not in df.columns:
            return base_result

        backtest_result = super().backtest(df, initial_capital)
        if not backtest_result or 'trades' not in backtest_result:
            return base_result

        trades = backtest_result['trades']
        if not trades:
            return base_result

        buy_points: List[Dict] = []
        sell_points: List[Dict] = []
        last_is_open = bool(df.iloc[-1]['buy_signal'] == 1)

        for idx, trade in enumerate(trades):
            buy_row = self._find_row_by_date(df, trade['buy_date'])
            sell_row = self._find_row_by_date(df, trade['sell_date'])
            is_last_trade_open = (idx == len(trades) - 1) and last_is_open

            if buy_row is not None:
                buy_points.append({
                    'date': trade['buy_date'],
                    'price': trade['buy_price'],
                    'reason': buy_row.get('entry_reason', 'RSI金叉信号'),
                    'fast_rsi': buy_row.get('fast_rsi'),
                    'slow_rsi': buy_row.get('slow_rsi'),
                    'trend_direction': buy_row.get('trend_direction'),
                    'ha_bullish': buy_row.get('is_heikin_bullish'),
                    'commission': trade.get('buy_commission', 0),
                })

            if sell_row is not None:
                reason = sell_row.get('exit_reason', 'ATR趋势转空')
                if is_last_trade_open:
                    reason = "持仓中：ATR趋势仍为多头"

                sell_points.append({
                    'date': trade['sell_date'],
                    'price': trade['sell_price'],
                    'reason': reason,
                    'fast_rsi': sell_row.get('fast_rsi'),
                    'slow_rsi': sell_row.get('slow_rsi'),
                    'trend_direction': sell_row.get('trend_direction'),
                    'ha_bullish': sell_row.get('is_heikin_bullish'),
                    'is_open': is_last_trade_open,
                    'commission': trade.get('sell_commission', 0),
                })

        return {
            'buy_points': buy_points,
            'sell_points': sell_points,
            'total_trades': len(trades),
            'initial_capital': initial_capital,
            'trades': trades,
            'gross_return': backtest_result.get('gross_return', 0),
        }

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #
    
    def _detect_bullish_divergence(self, data: pd.DataFrame) -> pd.Series:
        """
        检测底背离信号（完全避免未来函数）
        
        判断底背离方案（真正无未来函数版本）:
        1. 低点定义：当前是过去30日的最低价（不需要确认后一天是否更高）
        2. 判断底背离：当前低点价格 < 前一低点价格，但当前MACD diff > 前一低点MACD diff
        3. 必须找到连续2个底背离
        4. 当天收盘后立即发出信号（不延迟）
        
        例如：08-06是30日最低点，当天收盘后立即判断是否底背离并发出信号
        
        Returns: pd.Series 底背离信号序列
        """
        divergence_enabled = bool(self.config.get('trend_bullish_divergence_enabled', True))
        if not divergence_enabled:
            return pd.Series(False, index=data.index)
            
        lookback = int(self.config.get('trend_divergence_lookback', 30))
        min_consecutive = int(self.config.get('trend_divergence_min_consecutive', 2))
        
        # 初始化结果
        divergence_signals = pd.Series(False, index=data.index)
        
        if len(data) < lookback + 10:
            return divergence_signals
            
        close = data['close'].values
        macd_diff = data['macd_diff'].values if 'macd_diff' in data.columns else None
        
        if macd_diff is None:
            logger.warning("底背离检测需要MACD diff指标")
            return divergence_signals
            
        # 步骤1: 找到所有低点（真正无未来函数：低点=过去N日最低价）
        # 可以遍历到最后一天，不需要未来数据
        lows_indices = []
        for i in range(lookback, len(close)):  # 从lookback开始，确保有足够历史数据
            # 检查是否为过去lookback天的最低价
            start_idx = i - lookback
            if close[i] == np.min(close[start_idx:i+1]):
                # 额外确认：不能是平台（避免连续多天最低价）
                # 只在首次触及最低价时记录
                if i == start_idx or close[i] < close[i-1]:
                    lows_indices.append(i)
        
        if len(lows_indices) < min_consecutive:
            return divergence_signals
            
        # 步骤2: 对每个低点检查是否形成底背离
        divergence_lows = []  # 存储形成底背离的低点索引
        
        for i in range(1, len(lows_indices)):
            curr_idx = lows_indices[i]
            prev_idx = lows_indices[i-1]
            
            # 当前低点的收盘价比前一个低点低
            price_lower = close[curr_idx] < close[prev_idx]
            
            # 当前低点的MACD diff比前一个低点高（背离）
            macd_higher = macd_diff[curr_idx] > macd_diff[prev_idx]
            
            if price_lower and macd_higher:
                divergence_lows.append(curr_idx)
        
        # 步骤3: 检查连续底背离
        if len(divergence_lows) < min_consecutive:
            return divergence_signals
            
        # 寻找连续的底背离点（不做假背离检测，避免未来函数）
        consecutive_count = 1
        for i in range(1, len(divergence_lows)):
            # 检查在lows_indices中的位置是否连续
            curr_pos = lows_indices.index(divergence_lows[i])
            prev_pos = lows_indices.index(divergence_lows[i-1])
            
            if curr_pos == prev_pos + 1:
                consecutive_count += 1
                
                # 如果达到连续要求，当天立即发出信号（不延迟）
                if consecutive_count >= min_consecutive:
                    signal_idx = divergence_lows[i]
                    
                    # 斜率过滤：检查过去20天是否处于急速下跌中
                    slope_period = 20
                    slope_threshold = -0.003  # 约-15%/20天
                    pass_slope_filter = True
                    
                    if signal_idx >= slope_period:
                        # 计算过去20天的价格斜率（线性回归）
                        start_idx = signal_idx - slope_period
                        price_window = close[start_idx:signal_idx+1]
                        x = np.arange(len(price_window))
                        
                        # 使用最小二乘法计算斜率
                        mean_price = np.mean(price_window)
                        if mean_price > 0:
                            # 计算标准化斜率 = 斜率 / 平均价格
                            slope = np.polyfit(x, price_window, 1)[0]
                            normalized_slope = slope / mean_price
                            
                            # 如果斜率太负（急速下跌），不发出信号
                            if normalized_slope < slope_threshold:
                                pass_slope_filter = False
                                signal_date = data.index[signal_idx] if hasattr(data.index[signal_idx], 'strftime') else str(data.index[signal_idx])
                                logger.info(f"[斜率过滤] {self.stock_code} {signal_date}: "
                                          f"底背离被过滤 (斜率={normalized_slope:.6f} < {slope_threshold})")
                    
                    # 只有通过斜率过滤才发出信号
                    if pass_slope_filter:
                        # 当天立即发出买入信号
                        divergence_signals.iloc[signal_idx] = True
                        # 信号持续2天，便于触发买入
                        for j in range(signal_idx, min(signal_idx + 2, len(divergence_signals))):
                            divergence_signals.iloc[j] = True
            else:
                consecutive_count = 1
        
        return divergence_signals
    
    def _detect_main_wave_signals(self, data):
        """
        主升浪检测逻辑 - 返回整个Series
        
        参数:
        - data: 包含技术指标的数据DataFrame
        
        返回: pd.Series 主升浪信号序列
        """
        # 获取配置参数
        min_gain = self.config.get('trend_main_wave_min_gain', 10.0) / 100  # 降低到10%
        min_days = self.config.get('trend_main_wave_min_days', 3)  # 降低到3天
        rsi_threshold = self.config.get('trend_main_wave_rsi_threshold', 80)  # 提高到80
        volume_factor = self.config.get('trend_main_wave_volume_factor', 1.2)  # 降低到1.2倍
        
        # 初始化结果Series
        main_wave_signals = pd.Series(False, index=data.index)
        
        # 对每个时间点进行主升浪检测
        for i in range(len(data)):
            if i < min_days:
                continue
                
            # 条件1: 价格涨幅超过最小收益要求
            start_index = max(0, i - min_days)
            price_start = data['close'].iloc[start_index]
            price_current = data['close'].iloc[i]
            gain = (price_current - price_start) / price_start
            gain_condition = gain >= min_gain
            
            # 条件2: RSI仍有上涨空间（未过度超买）
            rsi_condition = data['fast_rsi'].iloc[i] <= rsi_threshold
            
            # 条件3: 成交量放大（相对于过去20天平均）
            if i >= 20:
                avg_volume = data['volume'].iloc[i-20:i].mean()
                current_volume = data['volume'].iloc[i]
                volume_condition = current_volume >= avg_volume * volume_factor
            else:
                volume_condition = True  # 历史数据不足时不限制
                
            # 条件4: 趋势方向向上（ATR趋势确认）
            trend_condition = data['trend_direction'].iloc[i] == 1
            
            # 条件5: MA多头排列（新增关键条件）
            ma_bullish_condition = False
            if 'exit_ema_fast' in data.columns and 'exit_ma_slow' in data.columns:
                ema16 = data['exit_ema_fast'].iloc[i]
                ma45 = data['exit_ma_slow'].iloc[i]
                ma_bullish_condition = ema16 > ma45
            
            # 条件6: 价格在关键均线之上（新增）
            price_above_ma_condition = True
            if 'exit_ma_slow' in data.columns:
                ma45 = data['exit_ma_slow'].iloc[i]
                price_above_ma_condition = data['close'].iloc[i] > ma45 * 0.95  # 允许5%的偏差
            
            # 条件7: 价格连续性上涨（放宽条件）
            if i >= 2:  # 降低到2天
                # 检查最近2天是否有1天以上上涨（放宽条件）
                recent_changes = data['close'].iloc[i-1:i+1].pct_change().dropna()
                up_days = (recent_changes > 0).sum()
                continuity_condition = up_days >= 1  # 放宽到至少1天上涨
            else:
                continuity_condition = True
                
            # 主升浪判断：核心条件必须满足，其他条件可以放宽
            # 核心条件：MA多头排列 + ATR趋势向上
            core_conditions = ma_bullish_condition and trend_condition
            
            # 辅助条件：至少满足3个
            auxiliary_conditions = [
                gain_condition,
                rsi_condition, 
                volume_condition,
                price_above_ma_condition,
                continuity_condition
            ]
            auxiliary_score = sum(auxiliary_conditions)
            
            # 主升浪信号：核心条件满足 + 至少3个辅助条件满足
            main_wave_signals.iloc[i] = core_conditions and auxiliary_score >= 3
        
        return main_wave_signals
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        else:
            data = data.sort_index().reset_index(drop=True)

        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(
                f"RSI趋势策略缺少必需字段: {', '.join(missing)}，"
                "请确认数据源包含 open/high/low/close。"
            )
        return data

    def _detect_market_trend_bias(self, data: pd.DataFrame, lookback: int = 20) -> str:
        """
        检测当前市场的趋势偏向
        
        Args:
            data: 价格数据
            lookback: 回溯周期
            
        Returns:
            'bullish': 上升趋势
            'bearish': 下跌趋势  
            'neutral': 震荡趋势
        """
        if len(data) < lookback:
            return 'neutral'
            
        recent_data = data.tail(lookback)
        
        # 1. 价格趋势分析
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1] 
        price_change_pct = (end_price - start_price) / start_price
        
        # 2. 移动平均趋势
        ma_short = recent_data['close'].rolling(5).mean().iloc[-1]
        ma_long = recent_data['close'].rolling(lookback).mean().iloc[-1]
        ma_trend = 1 if ma_short > ma_long else -1
        
        # 3. 波动性分析
        volatility = recent_data['close'].pct_change().std()
        high_volatility = volatility > 0.03  # 3%以上日波动率认为高波动
        
        # 4. 价格相对位置
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        current_position = (end_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        # 综合判断
        bullish_score = 0
        bearish_score = 0
        
        # 价格变化权重 (30%)
        if price_change_pct > 0.05:  # 上涨5%以上
            bullish_score += 3
        elif price_change_pct < -0.05:  # 下跌5%以上
            bearish_score += 3
        elif price_change_pct > 0:
            bullish_score += 1
        else:
            bearish_score += 1
            
        # 均线趋势权重 (25%)
        if ma_trend > 0:
            bullish_score += 2.5
        else:
            bearish_score += 2.5
            
        # 相对位置权重 (25%)
        if current_position > 0.7:  # 接近高点
            bullish_score += 2.5
        elif current_position < 0.3:  # 接近低点
            bearish_score += 2.5
        else:
            # 中性位置，根据最近趋势
            if price_change_pct > 0:
                bullish_score += 1
            else:
                bearish_score += 1
                
        # 波动性调整 (20%)
        if high_volatility:
            # 高波动时更谨慎
            bearish_score += 2
        else:
            # 低波动时可以稍微积极
            bullish_score += 2
            
        # 判断结果
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return 'neutral'
            
        bullish_ratio = bullish_score / total_score
        
        if bullish_ratio >= 0.6:
            return 'bullish'
        elif bullish_ratio <= 0.4:
            return 'bearish'
        else:
            return 'neutral'

    def _resample_to_higher_timeframe(self, data: pd.DataFrame, ratio: int) -> pd.DataFrame:
        """
        将数据重采样到更高时间框架（避免未来函数）
        
        Args:
            data: 原始数据框
            ratio: 时间框架比率（如5表示5倍周期）
            
        Returns:
            重采样后的数据框
        """
        if len(data) < ratio:
            return pd.DataFrame()  # 数据不足，返回空DataFrame
            
        # 每 ratio 个数据点合并为一个（只使用完整周期，避免未来函数）
        htf_data = []
        
        # 关键修改：只遍历完整的周期，最后不完整的周期不使用
        num_complete_periods = len(data) // ratio
        
        for i in range(num_complete_periods):
            start_idx = i * ratio
            end_idx = start_idx + ratio
            chunk = data.iloc[start_idx:end_idx]
            
            if len(chunk) < ratio:  # 确保是完整周期
                continue
                
            # 合并OHLC数据
            htf_row = {
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
            }
            
            # 如果有日期列，使用最后一个日期
            if 'date' in data.columns:
                htf_row['date'] = chunk['date'].iloc[-1]
                
            # 如果有成交量，使用总和
            if 'volume' in data.columns:
                htf_row['volume'] = chunk['volume'].sum()
                
            htf_data.append(htf_row)
            
        return pd.DataFrame(htf_data).reset_index(drop=True)

    def _compute_higher_timeframe_bias(self, data: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        计算更高时间框架的趋势偏向
        
        Returns:
            (htf_bias, htf_info) - 高时间框架偏向序列和相关信息
        """
        mtf_enabled = bool(self.config.get('trend_mtf_enabled', True))
        if not mtf_enabled:
            # 如果禁用多时间框架，返回全部为True的序列
            return pd.Series(True, index=data.index), {}
            
        mtf_ratio = max(2, int(self.config.get('trend_mtf_ratio', 5)))
        min_periods = max(20, int(self.config.get('trend_mtf_min_periods', 50)))
        adaptive_mode = bool(self.config.get('trend_mtf_adaptive_mode', True))
        trend_lookback = max(10, int(self.config.get('trend_mtf_trend_lookback', 20)))
        early_entry = bool(self.config.get('trend_mtf_early_entry', True))
        early_threshold = float(self.config.get('trend_mtf_early_threshold', 0.7))
        strict_mode = bool(self.config.get('trend_mtf_strict_mode', False))
        
        # 检查数据是否足够
        if len(data) < min_periods:
            logger.warning(f"数据量不足({len(data)})，无法进行多时间框架分析，需要至少{min_periods}条记录")
            return pd.Series(True, index=data.index), {'status': 'insufficient_data'}
            
        # 重采样到更高时间框架
        try:
            htf_data = self._resample_to_higher_timeframe(data, mtf_ratio)
            if len(htf_data) < 20:  # 高时间框架数据也要有足够的点
                return pd.Series(True, index=data.index), {'status': 'htf_insufficient_data'}
                
            # 计算高时间框架指标
            fast_period = int(self.config.get('trend_rsi_fast_period', 25))
            slow_period = int(self.config.get('trend_rsi_slow_period', 100))
            atr_period = int(self.config.get('trend_atr_period', 20))
            atr_multiplier = float(self.config.get('trend_atr_multiplier', 3.0))
            use_close = bool(self.config.get('trend_use_close_for_extrema', True))
            
            # 计算高时间框架RSI
            htf_fast_rsi = rsi_indicator(htf_data['close'], period=fast_period)
            htf_slow_rsi = rsi_indicator(htf_data['close'], period=slow_period)
            
            # 计算高时间框架ATR趋势
            htf_atr_values = atr_indicator(htf_data, period=atr_period) * atr_multiplier
            htf_long_stop, htf_short_stop, htf_direction = self._compute_trend_levels(
                htf_data, htf_atr_values, atr_period, use_close
            )
            
            # 动态模式选择：为每个高时间框架点计算趋势状态
            htf_bullish_bias = pd.Series(False, index=htf_data.index)
            mode_sequence = []  # 记录每个时期使用的模式
            
            for i in range(len(htf_data)):
                # 对每个高时间框架点，检测其局部趋势状态
                start_idx = max(0, i - trend_lookback + 1)
                end_idx = i + 1
                local_htf_data = htf_data.iloc[start_idx:end_idx]
                
                if len(local_htf_data) < 5:  # 数据不够，使用标准模式
                    local_trend = 'neutral'
                    effective_mode = 'standard'
                    effective_threshold = early_threshold
                else:
                    # 检测局部趋势
                    local_trend = self._detect_market_trend_bias(local_htf_data, min(trend_lookback, len(local_htf_data)))
                    
                    if adaptive_mode:
                        # 根据局部趋势动态调整模式
                        if local_trend == 'bullish':
                            effective_early_entry = True
                            effective_strict_mode = False
                            effective_threshold = early_threshold * 0.8  # 更积极
                            effective_mode = 'early'
                        elif local_trend == 'bearish':
                            effective_early_entry = False  
                            effective_strict_mode = True
                            effective_threshold = early_threshold * 1.2  # 更保守
                            effective_mode = 'strict'
                        else:  # neutral - 震荡市也使用早期入场模式
                            effective_early_entry = True
                            effective_strict_mode = False
                            effective_threshold = early_threshold * 0.9  # 适度积极
                            effective_mode = 'early_neutral'
                    else:
                        # 非自适应模式，使用配置值
                        effective_early_entry = early_entry
                        effective_strict_mode = strict_mode
                        effective_threshold = early_threshold
                        effective_mode = 'standard'
                
                mode_sequence.append({
                    'index': i,
                    'trend': local_trend,
                    'mode': effective_mode,
                    'threshold': effective_threshold
                })
                
                # 根据当前模式计算信号
                current_fast_rsi = htf_fast_rsi.iloc[i] if i < len(htf_fast_rsi) else np.nan
                current_slow_rsi = htf_slow_rsi.iloc[i] if i < len(htf_slow_rsi) else np.nan
                current_direction = htf_direction.iloc[i] if i < len(htf_direction) else np.nan
                
                if pd.isna(current_fast_rsi) or pd.isna(current_slow_rsi) or pd.isna(current_direction):
                    bias_value = True  # 数据不足时不限制
                elif effective_mode == 'strict':
                    # 严格模式：RSI和ATR趋势都必须看多
                    rsi_bullish = current_fast_rsi > current_slow_rsi
                    trend_bullish = current_direction == 1
                    bias_value = rsi_bullish and trend_bullish
                elif effective_mode == 'early':
                    # 早期入场模式：更灵活的确认条件
                    rsi_diff = current_fast_rsi - current_slow_rsi
                    rsi_strength = rsi_diff / 100.0  # 标准化到[-1, 1]
                    
                    # 条件1：RSI趋势向上且强度足够
                    rsi_early_bullish = (rsi_strength >= effective_threshold - 1.0) and (rsi_diff > 0)
                    
                    # 条件2：ATR趋势确认或即将转多
                    current_long_stop = htf_long_stop.iloc[i] if i < len(htf_long_stop) else np.nan
                    current_close = htf_data['close'].iloc[i]
                    trend_supportive = (current_direction == 1) or (
                        (current_direction == -1) and 
                        (not pd.isna(current_long_stop)) and
                        (current_close > current_long_stop * 0.98)  # 接近突破多头止损线
                    )
                    
                    # 条件3：价格动量确认（短期上涨）
                    momentum_start = max(0, i - 2)
                    if momentum_start < i:
                        momentum_change = htf_data['close'].iloc[i] / htf_data['close'].iloc[momentum_start] - 1
                        momentum_ok = momentum_change > -0.02  # 3期内跌幅不超过2%
                    else:
                        momentum_ok = True
                    
                    # 综合判断：至少满足两个条件
                    condition_count = sum([rsi_early_bullish, trend_supportive, momentum_ok])
                    bias_value = condition_count >= 2
                elif effective_mode == 'early_neutral':
                    # 震荡市早期入场模式：介于early和standard之间
                    rsi_diff = current_fast_rsi - current_slow_rsi
                    rsi_strength = rsi_diff / 100.0  # 标准化到[-1, 1]
                    
                    # 条件1：RSI趋势向上且强度适中
                    rsi_early_bullish = (rsi_strength >= effective_threshold - 1.0) and (rsi_diff > 0)
                    
                    # 条件2：ATR趋势确认或中性
                    current_long_stop = htf_long_stop.iloc[i] if i < len(htf_long_stop) else np.nan
                    current_close = htf_data['close'].iloc[i]
                    trend_supportive = (current_direction == 1) or (
                        (current_direction == -1) and 
                        (not pd.isna(current_long_stop)) and
                        (current_close > current_long_stop * 0.99)  # 震荡市中稍微宽松的突破条件
                    ) or (current_direction == 0)  # 震荡市中性方向也可接受
                    
                    # 条件3：价格动量确认（更宽松的动量要求）
                    momentum_start = max(0, i - 3)  # 看更长周期的动量
                    if momentum_start < i:
                        momentum_change = htf_data['close'].iloc[i] / htf_data['close'].iloc[momentum_start] - 1
                        momentum_ok = momentum_change > -0.03  # 震荡市中允许更大的短期回调
                    else:
                        momentum_ok = True
                    
                    # 震荡市更宽松：只要满足一个主要条件即可
                    bias_value = rsi_early_bullish and (trend_supportive or momentum_ok)
                else:
                    # 标准模式：平衡的确认条件
                    rsi_bullish = current_fast_rsi > current_slow_rsi
                    trend_bullish = current_direction == 1
                    bias_value = rsi_bullish and trend_bullish
                
                htf_bullish_bias.iloc[i] = bias_value
            
            # 将高时间框架信号映射回原始时间框架
            expanded_bias = []
            expanded_mode_info = []
            
            for i in range(len(data)):
                current_htf_idx = i // mtf_ratio
                if current_htf_idx < len(htf_bullish_bias):
                    bias_value = htf_bullish_bias.iloc[current_htf_idx]
                    mode_info = mode_sequence[current_htf_idx] if current_htf_idx < len(mode_sequence) else {'mode': 'standard', 'trend': 'neutral'}
                    if pd.isna(bias_value):
                        bias_value = True  # 默认不限制
                else:
                    bias_value = True  # 超出范围时不限制
                    mode_info = {'mode': 'standard', 'trend': 'neutral'}
                    
                expanded_bias.append(bias_value)
                expanded_mode_info.append(mode_info)
                
            htf_bias_series = pd.Series(expanded_bias, index=data.index)
            
            # 统计模式使用情况
            mode_stats = {}
            for mode_info in mode_sequence:
                mode = mode_info['mode']
                mode_stats[mode] = mode_stats.get(mode, 0) + 1
                
            # 返回诊断信息
            latest_htf_idx = min(len(htf_bullish_bias) - 1, (len(data) - 1) // mtf_ratio)
            latest_mode_info = mode_sequence[latest_htf_idx] if latest_htf_idx >= 0 and latest_htf_idx < len(mode_sequence) else {'mode': 'standard', 'trend': 'neutral'}
            
            htf_info = {
                'status': 'success',
                'adaptive_mode': adaptive_mode,
                'latest_market_trend': latest_mode_info.get('trend', 'neutral'),
                'latest_mode_used': latest_mode_info.get('mode', 'standard'),
                'mode_distribution': mode_stats,
                'htf_periods': len(htf_data),
                'latest_htf_rsi_fast': htf_fast_rsi.iloc[latest_htf_idx] if latest_htf_idx >= 0 and len(htf_fast_rsi) > latest_htf_idx else None,
                'latest_htf_rsi_slow': htf_slow_rsi.iloc[latest_htf_idx] if latest_htf_idx >= 0 and len(htf_slow_rsi) > latest_htf_idx else None,
                'latest_htf_trend_direction': htf_direction.iloc[latest_htf_idx] if latest_htf_idx >= 0 and len(htf_direction) > latest_htf_idx else None,
                'latest_htf_bias': htf_bullish_bias.iloc[latest_htf_idx] if latest_htf_idx >= 0 and len(htf_bullish_bias) > latest_htf_idx else None,
                'bias_ratio': htf_bias_series.sum() / len(htf_bias_series) if len(htf_bias_series) > 0 else 0,
            }
            
            return htf_bias_series, htf_info
            
        except Exception as e:
            logger.warning(f"多时间框架计算失败: {e}")
            return pd.Series(True, index=data.index), {'status': 'calculation_error', 'error': str(e)}

    def _compute_trend_levels(self, data: pd.DataFrame, atr_values: pd.Series,
                              period: int, use_close: bool) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算ATR止损线与趋势方向"""
        src_high = data['close'] if use_close else data['high']
        src_low = data['close'] if use_close else data['low']

        highest = src_high.rolling(period).max()
        lowest = src_low.rolling(period).min()
        long_base = highest - atr_values
        short_base = lowest + atr_values

        close_arr = data['close'].to_numpy()
        long_base_arr = long_base.to_numpy()
        short_base_arr = short_base.to_numpy()

        n = len(data)
        long_stop = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)
        direction = np.ones(n, dtype=int)

        for i in range(n):
            base_long = long_base_arr[i]
            base_short = short_base_arr[i]
            prev_long = long_stop[i - 1] if i > 0 else base_long
            prev_short = short_stop[i - 1] if i > 0 else base_short

            if np.isnan(prev_long):
                prev_long = base_long
            if np.isnan(prev_short):
                prev_short = base_short

            candidate_long = base_long
            if (
                i > 0
                and not np.isnan(prev_long)
                and not np.isnan(close_arr[i - 1])
                and close_arr[i - 1] > prev_long
            ):
                candidate_long = prev_long if np.isnan(candidate_long) else max(candidate_long, prev_long)
            if np.isnan(candidate_long):
                candidate_long = prev_long
            long_stop[i] = candidate_long

            candidate_short = base_short
            if (
                i > 0
                and not np.isnan(prev_short)
                and not np.isnan(close_arr[i - 1])
                and close_arr[i - 1] < prev_short
            ):
                candidate_short = prev_short if np.isnan(candidate_short) else min(candidate_short, prev_short)
            if np.isnan(candidate_short):
                candidate_short = prev_short
            short_stop[i] = candidate_short

            prev_dir = direction[i - 1] if i > 0 else 1
            dir_value = prev_dir
            if not np.isnan(prev_short) and not np.isnan(close_arr[i]) and close_arr[i] > prev_short:
                dir_value = 1
            elif not np.isnan(prev_long) and not np.isnan(close_arr[i]) and close_arr[i] < prev_long:
                dir_value = -1
            direction[i] = dir_value

        return (
            pd.Series(long_stop, index=data.index, name='long_stop'),
            pd.Series(short_stop, index=data.index, name='short_stop'),
            pd.Series(direction, index=data.index, name='trend_direction')
        )

    @staticmethod
    def _calculate_heikin_ashi(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """计算Heikin Ashi数据"""
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4.0
        ha_open = ha_close.copy()
        if len(data) > 0:
            ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2.0
            for i in range(1, len(data)):
                ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)
        return ha_open, ha_close, ha_high, ha_low

    @staticmethod
    def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
        return (fast > slow) & (fast.shift(1) <= slow.shift(1))

    @staticmethod
    def _crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
        return (fast < slow) & (fast.shift(1) >= slow.shift(1))

    @staticmethod
    def _build_position_series(entry_condition: pd.Series,
                               exit_condition: pd.Series,
                               price_series: pd.Series,
                               stop_loss_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """根据条件构造持仓序列"""
        n = len(entry_condition)
        position = np.zeros(n, dtype=int)
        entry_flags = np.zeros(n, dtype=int)
        exit_flags = np.zeros(n, dtype=int)
        stop_flags = np.zeros(n, dtype=int)
        in_position = False
        entry_price = None

        for i in range(n):
            entry_active = bool(entry_condition.iloc[i]) if not pd.isna(entry_condition.iloc[i]) else False
            exit_active = bool(exit_condition.iloc[i]) if not pd.isna(exit_condition.iloc[i]) else False
            curr_price = price_series.iloc[i] if i < len(price_series) else np.nan

            if not in_position and entry_active:
                in_position = True
                entry_flags[i] = 1
                entry_price = curr_price if not pd.isna(curr_price) else None

            if in_position and exit_active:
                in_position = False
                exit_flags[i] = 1
                entry_price = None

            elif in_position and stop_loss_pct > 0 and entry_price and not pd.isna(curr_price):
                threshold = entry_price * (1 - stop_loss_pct / 100.0)
                if curr_price <= threshold:
                    in_position = False
                    exit_flags[i] = 1
                    stop_flags[i] = 1
                    entry_price = None

            position[i] = 1 if in_position else 0

        return position, entry_flags, exit_flags, stop_flags

    def _build_position_series_with_divergence(self, entry_condition: pd.Series,
                                              exit_condition: pd.Series,
                                              divergence_entry: pd.Series,
                                              price_series: pd.Series,
                                              stop_loss_pct: float,
                                              data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """根据条件构造持仓序列（支持底背离买入保护）
        
        Args:
            entry_condition: 综合入场条件（标准入场 | 底背离）
            exit_condition: 退出条件
            divergence_entry: 底背离入场信号
            price_series: 价格序列
            stop_loss_pct: 止损百分比
            data: 完整数据
            
        Returns:
            position, entry_flags, exit_flags, stop_flags, profit_target_flags
        """
        min_hold_days = int(self.config.get('trend_divergence_min_hold_days', 10))
        profit_target_pct = float(self.config.get('trend_divergence_profit_target', 15.0))
        ignore_rsi_exit = bool(self.config.get('trend_divergence_ignore_rsi_exit', False))
        use_rsi_trend = bool(self.config.get('trend_divergence_use_rsi_trend', True))
        rsi_decline_threshold = float(self.config.get('trend_divergence_rsi_decline_threshold', -5.0))
        
        # 获取RSI数据用于趋势判断
        rsi_fast = None
        if use_rsi_trend and data is not None and 'fast_rsi' in data.columns:
            rsi_fast = data['fast_rsi']
        
        n = len(entry_condition)
        position = np.zeros(n, dtype=int)
        entry_flags = np.zeros(n, dtype=int)
        exit_flags = np.zeros(n, dtype=int)
        stop_flags = np.zeros(n, dtype=int)
        profit_target_flags = np.zeros(n, dtype=int)  # 止盈标记
        in_position = False
        entry_price = None
        is_divergence_entry = False  # 标记当前持仓是否为底背离买入
        hold_days = 0  # 持仓天数
        entry_rsi = None  # 记录买入时的RSI值

        for i in range(n):
            entry_active = bool(entry_condition.iloc[i]) if not pd.isna(entry_condition.iloc[i]) else False
            exit_active = bool(exit_condition.iloc[i]) if not pd.isna(exit_condition.iloc[i]) else False
            is_div_entry = bool(divergence_entry.iloc[i]) if not pd.isna(divergence_entry.iloc[i]) else False
            curr_price = price_series.iloc[i] if i < len(price_series) else np.nan

            if not in_position and entry_active:
                in_position = True
                entry_flags[i] = 1
                entry_price = curr_price if not pd.isna(curr_price) else None
                is_divergence_entry = is_div_entry  # 记录是否为底背离买入
                hold_days = 0  # 重置持仓天数
                # 记录买入时的RSI值（用于底背离买入的趋势判断）
                if is_div_entry and rsi_fast is not None and i < len(rsi_fast):
                    entry_rsi = rsi_fast.iloc[i] if not pd.isna(rsi_fast.iloc[i]) else None
                else:
                    entry_rsi = None

            if in_position:
                hold_days += 1
                
                # 底背离买入的特殊退出逻辑（不使用15%止盈，只用ATR+止损控制）
                if is_divergence_entry and entry_price and not pd.isna(curr_price):
                    # 条件1：未达到最短持有天数，只有止损才退出
                    if hold_days < min_hold_days:
                        # 只有触发止损时才退出
                        if stop_loss_pct > 0 and entry_price:
                            threshold = entry_price * (1 - stop_loss_pct / 100.0)
                            if curr_price <= threshold:
                                in_position = False
                                exit_flags[i] = 1
                                stop_flags[i] = 1
                                entry_price = None
                                is_divergence_entry = False
                                entry_rsi = None
                                hold_days = 0
                    # 条件2：达到最短持有天数后
                    else:
                        # 使用RSI相对变化判断（推荐）
                        if use_rsi_trend and rsi_fast is not None and entry_rsi is not None:
                            # 获取当前RSI值
                            curr_rsi = rsi_fast.iloc[i] if i < len(rsi_fast) and not pd.isna(rsi_fast.iloc[i]) else None
                            
                            if curr_rsi is not None:
                                # 计算RSI相对变化
                                rsi_change = curr_rsi - entry_rsi
                                
                                # 如果RSI相对于买入时显著下降，才考虑退出
                                if exit_active and rsi_change < rsi_decline_threshold:
                                    in_position = False
                                    exit_flags[i] = 1
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                                # 否则只检查止损
                                elif stop_loss_pct > 0 and entry_price:
                                    threshold = entry_price * (1 - stop_loss_pct / 100.0)
                                    if curr_price <= threshold:
                                        in_position = False
                                        exit_flags[i] = 1
                                        stop_flags[i] = 1
                                        entry_price = None
                                        is_divergence_entry = False
                                        entry_rsi = None
                                        hold_days = 0
                            else:
                                # RSI数据不可用，按正常逻辑
                                if exit_active:
                                    in_position = False
                                    exit_flags[i] = 1
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                        # 完全忽略RSI退出
                        elif ignore_rsi_exit:
                            if stop_loss_pct > 0 and entry_price:
                                threshold = entry_price * (1 - stop_loss_pct / 100.0)
                                if curr_price <= threshold:
                                    in_position = False
                                    exit_flags[i] = 1
                                    stop_flags[i] = 1
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                        # 默认逻辑（按正常退出）
                        else:
                            if exit_active:
                                in_position = False
                                exit_flags[i] = 1
                                entry_price = None
                                is_divergence_entry = False
                                entry_rsi = None
                                hold_days = 0
                            elif stop_loss_pct > 0 and entry_price:
                                threshold = entry_price * (1 - stop_loss_pct / 100.0)
                                if curr_price <= threshold:
                                    in_position = False
                                    exit_flags[i] = 1
                                    stop_flags[i] = 1
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                
                # 非底背离买入，按正常逻辑处理
                elif not is_divergence_entry:
                    if exit_active:
                        in_position = False
                        exit_flags[i] = 1
                        entry_price = None
                        hold_days = 0
                    elif stop_loss_pct > 0 and entry_price and not pd.isna(curr_price):
                        threshold = entry_price * (1 - stop_loss_pct / 100.0)
                        if curr_price <= threshold:
                            in_position = False
                            exit_flags[i] = 1
                            stop_flags[i] = 1
                            entry_price = None
                            hold_days = 0

            position[i] = 1 if in_position else 0

        return position, entry_flags, exit_flags, stop_flags, profit_target_flags

    @staticmethod
    def _build_entry_reasons(data: pd.DataFrame, entry_flags: np.ndarray) -> pd.Series:
        reasons = [''] * len(data)
        for idx, flag in enumerate(entry_flags):
            if flag:
                row = data.iloc[idx]
                parts = []
                
                # 检查是否为底背离入场（独立生效）
                if row.get('bullish_divergence_signal', False):
                    parts.append("底背离信号（独立生效）")
                # 标准RSI入场
                else:
                    if row.get('golden_cross'):
                        parts.append("RSI金叉")
                    elif row.get('rsi_relaxed_condition'):
                        parts.append("RSI多头延续")
                    else:
                        parts.append("RSI多头")
                    if row.get('is_heikin_bullish'):
                        parts.append("Heikin Ashi 阳线")
                    if row.get('trend_direction') == 1:
                        parts.append("ATR趋势多头")
                    if row.get('mtf_bias', True):
                        parts.append("高时间框架一致")
                        
                reasons[idx] = ' + '.join(parts)
        return pd.Series(reasons, index=data.index)

    @staticmethod
    def _build_exit_reasons(data: pd.DataFrame, exit_flags: np.ndarray) -> pd.Series:
        reasons = [''] * len(data)
        for idx, flag in enumerate(exit_flags):
            if flag:
                parts = []
                
                # 优先检查止盈退出
                if data.iloc[idx].get('profit_target_exit'):
                    parts.append("底背离止盈15%")
                else:
                    parts.append("ATR趋势转空")
                
                if data.iloc[idx].get('death_cross'):
                    parts.append("RSI死叉确认")
                if data.iloc[idx].get('exit_ma_filter_break'):
                    parts.append("EMA16>MA45下连续3日跌破MA16")
                if data.iloc[idx].get('stop_loss_exit'):
                    stop_val = data.iloc[idx].get('stop_loss_pct')
                    if stop_val and not pd.isna(stop_val):
                        parts.append(f"触发{stop_val:.1f}%止损")
                    else:
                        parts.append("触发止损")
                reasons[idx] = ' + '.join(parts)
        return pd.Series(reasons, index=data.index)

    @staticmethod
    def _find_row_by_date(df: pd.DataFrame, target_date) -> Optional[pd.Series]:
        """根据日期字符串查找行"""
        if 'date' in df.columns:
            mask = df['date'].astype(str).str.split().str[0] == str(target_date).split()[0]
            if mask.any():
                return df.loc[mask].iloc[0]
        else:
            mask = df.index.astype(str).str.split().str[0] == str(target_date).split()[0]
            if mask.any():
                return df.loc[mask].iloc[0]
        return None
