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
from scipy import stats

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

    # 类级别缓存：大盘指数数据（避免每只股票重复获取）
    _index_cache: Dict[str, pd.DataFrame] = {}
    _index_regime: Dict[str, pd.Series] = {}  # 预计算的regime信号(date→bool)

    def __init__(self, config: Optional[Dict] = None, market: str = 'CN-A',
                 stock_code: str = ''):
        defaults = {
            'trend_rsi_fast_period': 29,
            'trend_rsi_slow_period': 65,
            'trend_atr_period': 20,
            'trend_atr_multiplier': 3.0,
            'trend_use_close_for_extrema': True,
            'trend_relaxed_entry': True,
            'trend_relaxed_min_gap': 1.5,

            'trend_stop_loss_pct': 8.5,  # 优化：7.0→8.5，+4.70%收益，+1.30%胜率
            'trend_exit_use_ma_filter': True,
            'trend_exit_fast_ema_period': 20,
            'trend_exit_slow_ma_period': 40,
            'trend_exit_confirm_ma_period': 21,
            'trend_lr_filter_enabled': True,
            'trend_lr_lookback': 30,
            'trend_lr_max_slope_pct': 1.5,
            # 多时间框架配置
            # 重要：为避免MTF带来的信号滞后/漏买，且避免任何潜在未来函数争议，默认强制关闭。
            # 如需重新启用，请修改代码（当前版本忽略外部配置）。
            'trend_mtf_enabled': False,
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

            # 高抛低吸参数（HYBRID策略：BB+RSI+涨幅多条件组合）
            'swing_trade_enabled': True,            # 高抛低吸开关
            'swing_min_hold_days': 8,               # 最少持仓8天才考虑
            'swing_min_profit_pct': 3.0,            # 最少浮盈3%才考虑
            'swing_max_profit_pct': 25.0,           # 浮盈超过此值不高抛（保护大牛股）
            'swing_sell_gain_threshold': 10.0,      # 涨幅超过10%才允许卖出（必须与RSI同时满足）
            'swing_aroon_threshold': 25,            # aroon_osc绝对值 < 25 = 震荡市
            'swing_bb_sell_threshold': 0.80,        # bb_percent > 0.80 = 高位（必须满足）
            'swing_rsi_sell_threshold': 65,         # fast_rsi > 65 = 超买
            'swing_volume_surge_block': 1.8,        # 成交量 > 1.8倍均量时不卖（放量突破保护）
            'swing_bb_rebuy_threshold': 0.35,       # bb_percent < 0.35 = 回到低位买回（优化：0.50→0.35⭐）
            'swing_rsi_rebuy_threshold': 40,        # fast_rsi < 40 = 超卖买回
            'swing_stoch_k_rebuy_threshold': 30,    # KDJ K线 < 30 = 超卖买回
            'swing_rebuy_drop_pct': 999.0,          # 禁用纯跌幅回买（优化：4.0→禁用，提升胜率⭐）
            'swing_breakout_chase_pct': 999.0,      # 禁用普通追高买回（分析显示追高胜率低）
            'swing_breakout_max_gap_pct': 999.0,    # 禁用普通追高买回
            'swing_breakout_min_wait_days': 999,    # 禁用普通追高买回
            'swing_next_day_up_rebuy': False,       # 次日收涨立即买回（分析发现容易追高，默认关闭）
            'swing_volume_breakout_rebuy': True,    # 放量突破买回（真突破信号）
            'swing_volume_breakout_ratio': 1.8,     # 放量突破的量比阈值
            'swing_max_wait_days': 8,               # 最多等8天买回（优化：12→8⭐）
            'swing_max_loss_from_sell_pct': 5.0,    # 跌超过卖出价5%放弃买回
            'swing_trend_reversal_giveup': True,    # trend_direction变-1则放弃

            # W底形态配置
            'trend_w_bottom_enabled': True,
            'trend_w_bottom_lookback': 20,
            'trend_w_bottom_min_gap': 25,
            'trend_w_bottom_price_tolerance': 0.016,

            # 震荡市场入场配置
            'sideways_aroon_threshold': 22,        # Aroon震荡阈值（|osc|<此值=震荡）
            'sideways_bb_width_enabled': False,    # 启用BB宽度辅助震荡检测
            'sideways_bb_width_threshold': 0.10,   # BB宽度<此值=窄幅震荡（百分比）
            'sideways_entry_bb_pct': 0.15,         # 入场BB percent阈值（价格接近下轨）
            'sideways_entry_rsi': 28,              # 入场RSI阈值（超卖）
            'sideways_exit_bb_upper': 0.85,        # 退出条件1：BB percent上轨
            'sideways_exit_rsi_upper': 70,         # 退出条件1：RSI超买
            'sideways_exit_tp_pct': 8.5,           # 退出条件2：止盈%
            'sideways_exit_sl_pct': 5.0,           # 退出条件3：止损%

            # 主升浪延长持仓（Extended Hold）
            'extended_hold_profit_threshold': 30, # 浮盈>X%时触发延长持仓
            'extended_hold_peak_trailing': 20,    # 从最高浮盈回撤X%后退出
            'extended_hold_peak_activation_offset': 999,  # 峰值回撤激活偏移
            'extended_hold_gain_protection_ratio': 0,     # 比例保护（0=关闭）
            'extended_hold_min_days': 35,         # 延长持仓最少天数
            'extended_hold_profit_cap': 150,      # 利润上限

            # 退场后回补（Post-Wave Reentry）
            'post_wave_reentry_window': 120,      # 回补窗口天数
            'post_wave_profit_threshold': 99999,  # 触发回补的浮盈阈值（99999=禁用）
            'post_wave_min_hold': 35,             # 回补前最少持仓天数
            'post_wave_price_confirm_pct': 5,     # 价格突破确认%

            # 滞涨退出（浮盈达标后连续N天未创新高）
            'stale_peak_enabled': True,
            'stale_peak_min_profit': 40,          # 浮盈>X%时才检查
            'stale_peak_max_days': 25,            # 未创新高天数阈值

            # 早期止损收紧
            'early_stop_days': 0,                 # 前N天使用更紧止损（0=关闭）
            'early_stop_loss_pct': 5.0,           # 早期止损百分比

            # 入场成交量确认
            'entry_vol_confirm_mult': 0,          # 要求入场日成交量达均量X倍（0=关闭）

            # EH做T放量阴线信号
            'eh_swing_vol_signal_enabled': True,
            'eh_swing_vol_signal_mult': 2.5,      # 放量倍数阈值
            'eh_swing_vol_signal_lookback': 7,    # 回看天数
            'eh_swing_vol_signal_count': 1,       # 需要N根放量阴线

            # 止盈保护（trailing stop）
            'trailing_stop_trigger': 6,           # 浮盈X%后激活保本止损（0=关闭）
            'trailing_stop_level': 0,             # 回到入场价+level%就卖
            'trailing_stop_trigger2': 0,          # 双层trailing: 更高利润时使用更紧floor（0=关闭）
            'trailing_stop_level2': 8,            # 高层trailing floor
            'trailing_stop_confirm': 0,           # 确认K线数（0=立即卖出）
            'trailing_stop_panic_skip': 0,        # 恐慌过滤（0=关闭）
            'trailing_stop_calm_threshold': 0,    # 平稳期突跌过滤（0=关闭）
            'trailing_stop_calm_lookback': 5,     # 平稳期回看天数

            # 成交量分布退出（窗口内多次放量阴线=机构派发）
            'dist_exit_enabled': True,
            'dist_exit_min_profit': 20,           # 浮盈>X%时才检查
            'dist_exit_lookback': 30,             # 回看窗口天数
            'dist_exit_vol_threshold': 2.5,       # 放量阈值（倍均量）
            'dist_exit_count': 3,                 # 窗口内需要N次放量阴线

            # 放量阴线+均线偏离退出
            'dist_madev_exit_enabled': True,
            'dist_madev_exit_min_profit': 20,     # 浮盈>X%时才检查
            'dist_madev_exit_ma_period': 20,      # 均线周期
            'dist_madev_exit_dev_pct': 18,        # 偏离均线>X%
            'dist_madev_exit_vol_mult': 2.2,      # 放量阈值（倍均量）

            # Hurst Exponent趋势过滤器
            'hurst_enabled': True,                  # 启用Hurst指数动态切换
            'hurst_window': 100,                    # Hurst计算窗口期
            'hurst_mean_revert_threshold': 0.45,    # <0.45使用均值回归策略
            'hurst_trending_threshold': 0.65,       # >0.65使用趋势跟踪策略

            # Volume Quality Filter（成交量质量过滤）
            'volume_quality_enabled': True,         # 启用成交量质量评分
            'vq_min_score': 50,                     # 最低成交量质量评分（0-100）

            # Slope-Pearson Downtrend Filter（斜率+确定性下跌过滤）
            'slope_downtrend_filter_enabled': True,   # 启用斜率下跌过滤
            'slope_downtrend_period': 180,            # 超长期斜率周期（天）
            'slope_downtrend_threshold': -0.0042,     # 斜率阈值（负数，越小=越严格）
            'slope_downtrend_pearson': 0.70,          # Pearson R²最低确定性

            # Hard Loss Cap（硬性最大亏损上限）
            'hard_loss_cap_enabled': True,            # 启用硬性亏损上限
            'hard_loss_cap_pct': 8.5,                 # 最大允许亏损百分比

            # W底缓冲期参数
            'wb_buffer_stop_pct': 8,                  # W底缓冲期止损：跌破W2低点X%
            'wb_buffer_profit_pct': 999,              # W底缓冲期止盈：涨幅X%（999=禁用）

            # Gap Fade（跳空回补入场）
            'gap_fade_enabled': True,                 # 启用跳空回补策略
            'gap_fade_threshold': 3.0,                # 跳空阈值%（向下跳空）

            # 底背离质量过滤器
            'trend_divergence_min_drop_pct': 15,      # 底背离需要从近期高点下跌 > X%(0=关闭)

            # 追涨/跳空参数（a6d优化值）
            'chase_rise_vol_max': 2.5,                # 追涨最大量比
            'chase_min_drop_speed': 0.8,              # 追跌最小下跌速度
            'chase_max_drop_pct': 15,                 # 追跌最大跌幅%

            # 反弹卖出参数（a6d优化值）
            'bounce_exit_drop_threshold': -2.2,       # 触发延迟的当日跌幅阈值%
            'bounce_exit_max_wait': 2,                # 最大等待天数

            # 多指标超买集群退出（利润10-22%区间，多个振荡指标同时超买时退出）
            'ob_cluster_exit_enabled': False,
            'ob_cluster_exit_min_profit': 12,     # 最低利润%
            'ob_cluster_exit_max_profit': 22,     # 最高利润%（以上由dist_exit接管）
            'ob_cluster_exit_min_count': 3,       # 至少N个指标超买（共4个）
            'ob_cluster_exit_rsi_thresh': 70,     # RSI超买阈值
            'ob_cluster_exit_stk_thresh': 80,     # StochK超买阈值
            'ob_cluster_exit_cci_thresh': 200,    # CCI超买阈值
            'ob_cluster_exit_mfi_thresh': 80,     # MFI超买阈值

            # 放量冲高回落退出（上影线>实体+收盘下半区+放量）
            'vol_climax_exit_enabled': True,
            'vol_climax_exit_min_profit': 15,     # 最低利润%（优化：8→15, dPF +0.022）
            'vol_climax_exit_vol_mult': 3.0,      # 放量倍数阈值（优化：2.5→3.0）
            'vol_climax_exit_require_new_high': True,  # 要求近5日新高

            # ROC动量衰竭退出（ROC正值但连续下降=加速度为负）
            'roc_fade_exit_enabled': False,
            'roc_fade_exit_min_profit': 8,        # 最低利润%
            'roc_fade_exit_declining_days': 3,    # ROC连续下降天数
            'roc_fade_exit_roc_floor': 0,         # ROC下限（0=仅要求下降，不要求转负）

            # BB Squeeze突破入场（波动率收缩→扩张+突破中轨）
            'bb_squeeze_entry_enabled': False,
            'bb_squeeze_entry_lookback': 20,      # BB宽度N日最低
            'bb_squeeze_entry_expansion_pct': 5,  # 宽度扩张X%
            'bb_squeeze_entry_require_dir': True,  # 要求direction=1

            # Stochastic超卖反转入场（KD金叉从超卖区）
            'stoch_reversal_entry_enabled': False,
            'stoch_reversal_entry_k_thresh': 20,  # K线超卖阈值
            'stoch_reversal_entry_require_cross': True,  # 要求K上穿D
            'stoch_reversal_entry_require_ha': True,     # 要求HA阳线
        }
        if config:
            defaults.update(config)

        # 代码层面强制关闭MTF：忽略外部配置文件/入参对该开关的覆盖
        defaults['trend_mtf_enabled'] = False

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
    # Market Regime Filter                                                  #
    # --------------------------------------------------------------------- #
    @classmethod
    def _load_index_regime(cls, index_code: str, ma_period: int,
                           start_date: str = '2018-01-01',
                           buffer_pct: float = 0.0) -> pd.Series:
        """获取并缓存大盘指数regime信号。返回以日期为index的bool Series (True=允许入场)

        buffer_pct: 缓冲区百分比。0=标准(close>MA就允许)，5=只在close<MA*0.95时阻止
        """
        cache_key = f"{index_code}_{ma_period}_buf{buffer_pct}"
        if cache_key in cls._index_regime:
            return cls._index_regime[cache_key]

        try:
            import akshare as ak
            # A股指数
            if index_code.isdigit() and len(index_code) == 6:
                df_idx = ak.index_zh_a_hist(
                    symbol=index_code, period='daily',
                    start_date=start_date.replace('-', ''),
                    end_date='21000101'
                )
                close_col = '收盘'
                date_col = '日期'
            else:
                # 港股指数（如HSI）- 暂不支持，默认全部通过
                cls._index_regime[cache_key] = pd.Series(dtype=bool)
                return cls._index_regime[cache_key]

            if df_idx is None or df_idx.empty:
                logger.warning(f"无法获取指数 {index_code} 数据，regime filter 将被跳过")
                cls._index_regime[cache_key] = pd.Series(dtype=bool)
                return cls._index_regime[cache_key]

            df_idx[date_col] = pd.to_datetime(df_idx[date_col])
            df_idx = df_idx.sort_values(date_col).reset_index(drop=True)
            close = df_idx[close_col].astype(float)
            ma = close.rolling(ma_period).mean()
            # 带缓冲区的regime: 只有当close < MA * (1 - buffer/100)时才标记为熊市
            # buffer=0: 标准模式 (close > MA → 牛市)
            # buffer=5: 只有close < MA*0.95才是熊市 (轻微回调仍允许入场)
            threshold = ma * (1.0 - buffer_pct / 100.0)
            regime = close > threshold
            regime.index = df_idx[date_col].dt.strftime('%Y-%m-%d')
            cls._index_regime[cache_key] = regime
            bear_pct = (1 - regime.mean()) * 100
            logger.info(f"大盘regime已加载: {index_code}, MA{ma_period}, buffer={buffer_pct}%, "
                        f"允许入场={regime.sum()}/{len(regime)} ({regime.mean()*100:.1f}%), "
                        f"阻止={bear_pct:.1f}%天数")
            return regime
        except Exception as e:
            logger.warning(f"获取指数regime失败: {e}, filter将被跳过")
            cls._index_regime[cache_key] = pd.Series(dtype=bool)
            return cls._index_regime[cache_key]

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def analyze(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """执行RSI趋势策略分析"""
        if df is None or len(df) == 0:
            logger.warning("RSITrendStrategy: 数据为空，无法分析")
            return None, None

        data = self._prepare_dataframe(df)
        
        fast_period = int(self.config['trend_rsi_fast_period'])
        slow_period = int(self.config['trend_rsi_slow_period'])
        atr_period = int(self.config['trend_atr_period'])
        atr_multiplier = float(self.config['trend_atr_multiplier'])
        use_close = bool(self.config['trend_use_close_for_extrema'])
        exit_ma_filter_enabled = bool(self.config['trend_exit_use_ma_filter'])
        lr_filter_enabled = bool(self.config['trend_lr_filter_enabled'])
        lr_filter_enabled = bool(self.config['trend_lr_filter_enabled'])
        exit_fast_ema_period = max(1, int(self.config['trend_exit_fast_ema_period']))
        exit_slow_ma_period = max(1, int(self.config['trend_exit_slow_ma_period']))
        exit_confirm_ma_period = max(
            1,
            int(self.config['trend_exit_confirm_ma_period'])
        )
        lr_filter_enabled = bool(self.config['trend_lr_filter_enabled'])
        lr_lookback = max(5, int(self.config['trend_lr_lookback']))
        lr_max_slope_pct = max(0.0, float(self.config['trend_lr_max_slope_pct']))

        data['fast_rsi'] = rsi_indicator(data['close'], period=fast_period)
        data['slow_rsi'] = rsi_indicator(data['close'], period=slow_period)

        # 成交量过滤：低成交量信号过滤（0.75倍MA20）
        if 'volume' in data.columns:
            data['volume_ma20'] = data['volume'].rolling(window=20, min_periods=1).mean()
            data['volume_weak'] = data['volume'] < data['volume_ma20'] * 0.75
        else:
            data['volume_weak'] = pd.Series(False, index=data.index)

        # 大周期M顶过滤：规避危险高位追高（ID021最优参数）
        # shift=60, window=120, lower=0.92, upper=1.05, threshold=-3
        # 优化结果：172.60% (+6.27% vs baseline 166.33%)
        # shift:window = 1:2 黄金比例，经过全面验证
        data['high_1y'] = data['high'].rolling(window=250, min_periods=100).max()
        data['position_vs_1y_high'] = (data['close'] / data['high_1y'] - 1) * 100
        data['prev_peak_60_120'] = data['high'].shift(60).rolling(window=120, min_periods=30).max()
        data['is_m_top'] = (
            (data['close'] / data['prev_peak_60_120']).between(0.92, 1.05) &
            (data['position_vs_1y_high'] < -3)
        )

        # 计算MACD指标（用于底背离检测）
        macd_diff, macd_dea, macd_hist = macd_indicator(data['close'])
        data['macd_diff'] = macd_diff
        data['macd_dea'] = macd_dea
        data['macd_hist'] = macd_hist

        # 计算MA120和短期涨幅（用于规避极端追高）
        data['ma_120'] = data['close'].rolling(120).mean()
        data['short_gain_10d'] = (data['close'] / data['close'].shift(10) - 1) * 100

        # Aroon震荡市场检测
        from .indicators import calculate_aroon
        data = calculate_aroon(data, period=25)
        _sw_aroon_thresh = float(self.config.get('sideways_aroon_threshold', 22))
        data['is_sideways'] = (data['aroon_osc'].abs() < _sw_aroon_thresh)

        atr_values = atr_indicator(data, period=atr_period) * atr_multiplier
        data['atr_trailing'] = atr_values
        data['atr'] = atr_indicator(data, period=14)  # 用于强弱判断的14日ATR
        # ATR波动率过滤：当波动率异常高时（ATR > 2倍MA20），避免入场
        data['atr_ma20'] = data['atr'].rolling(window=20, min_periods=10).mean()
        data['atr_expanding'] = data['atr'] > data['atr_ma20'] * 1.5

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
        # RSI动量加速：3日变化量
        data['rsi_momentum'] = data['fast_rsi'] - data['fast_rsi'].shift(3)
        # ROC (Rate of Change) 10日价格动量
        data['roc_10'] = (data['close'] / data['close'].shift(10) - 1) * 100

        # 双通道策略：长期趋势确认 + 短期回调买点
        dual_channel_enabled = True
        ultra_long_period = 180
        very_long_period = 120
        long_period = 60
        short_period = 20
        dev_multiplier = 2.0

        log_close = np.log(data['close'])

        def calc_channel_params(arr, period):
            if len(arr) < period or np.isnan(arr).any():
                return np.nan, np.nan, np.nan, np.nan
            n = len(arr)
            x = np.arange(n)
            sum_x = np.sum(x)
            sum_xx = np.sum(x * x)
            sum_y = np.sum(arr)
            sum_yx = np.sum(x * arr)
            slope = (n * sum_yx - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            average = sum_y / n
            intercept = average - slope * sum_x / n + slope
            fitted_val = intercept
            sum_dev = 0.0
            for i in range(n):
                residual = arr[i] - fitted_val
                fitted_val += slope
                sum_dev += residual * residual
            std_dev = np.sqrt(sum_dev / (n - 1))
            regres = intercept + slope * (n - 1) * 0.5
            sum_dxx = 0.0
            sum_dyy = 0.0
            sum_dyx = 0.0
            fitted_val = intercept
            for i in range(n):
                dxt = arr[i] - average
                dyt = fitted_val - regres
                fitted_val += slope
                sum_dxx += dxt * dxt
                sum_dyy += dyt * dyt
                sum_dyx += dxt * dyt
            pearson = sum_dyx / np.sqrt(sum_dxx * sum_dyy) if sum_dxx * sum_dyy > 0 else 0.0
            return slope, intercept, std_dev, abs(pearson)

        ultra_long_slope = pd.Series(index=data.index, dtype=float)
        ultra_long_pearson = pd.Series(index=data.index, dtype=float)
        for i in range(ultra_long_period - 1, len(data)):
            arr = log_close.iloc[i - ultra_long_period + 1:i + 1].values
            slope, _, _, pearson = calc_channel_params(arr, ultra_long_period)
            ultra_long_slope.iloc[i] = slope
            ultra_long_pearson.iloc[i] = pearson

        very_long_slope = pd.Series(index=data.index, dtype=float)
        very_long_pearson = pd.Series(index=data.index, dtype=float)
        for i in range(very_long_period - 1, len(data)):
            arr = log_close.iloc[i - very_long_period + 1:i + 1].values
            slope, _, _, pearson = calc_channel_params(arr, very_long_period)
            very_long_slope.iloc[i] = slope
            very_long_pearson.iloc[i] = pearson

        long_slope = pd.Series(index=data.index, dtype=float)
        long_intercept = pd.Series(index=data.index, dtype=float)
        long_std = pd.Series(index=data.index, dtype=float)
        long_pearson = pd.Series(index=data.index, dtype=float)
        for i in range(long_period - 1, len(data)):
            arr = log_close.iloc[i - long_period + 1:i + 1].values
            slope, intercept, std, pearson = calc_channel_params(arr, long_period)
            long_slope.iloc[i] = slope
            long_intercept.iloc[i] = intercept
            long_std.iloc[i] = std
            long_pearson.iloc[i] = pearson

        short_slope = pd.Series(index=data.index, dtype=float)
        short_intercept = pd.Series(index=data.index, dtype=float)
        short_std = pd.Series(index=data.index, dtype=float)
        short_lower = pd.Series(index=data.index, dtype=float)
        for i in range(short_period - 1, len(data)):
            arr = log_close.iloc[i - short_period + 1:i + 1].values
            slope, intercept, std, _ = calc_channel_params(arr, short_period)
            short_slope.iloc[i] = slope
            short_intercept.iloc[i] = intercept
            short_std.iloc[i] = std
            midline = np.exp(intercept)
            short_lower.iloc[i] = midline / np.exp(dev_multiplier * std)

        ultra_long_uptrend = (ultra_long_slope > 0) & (ultra_long_pearson > 0.65)
        very_long_uptrend = (very_long_slope > 0) & (very_long_pearson > 0.70)
        long_uptrend = (long_slope > 0) & (long_pearson > 0.75)
        price_near_lower = data['close'] <= short_lower * 1.02

        data['dual_channel_signal'] = ultra_long_uptrend & very_long_uptrend & long_uptrend & price_near_lower
        data['ultra_long_channel_uptrend'] = ultra_long_uptrend
        data['ultra_long_channel_pearson'] = ultra_long_pearson
        data['ultra_long_slope'] = ultra_long_slope
        data['very_long_channel_uptrend'] = very_long_uptrend
        data['very_long_channel_pearson'] = very_long_pearson
        data['very_long_slope'] = very_long_slope
        data['long_channel_uptrend'] = long_uptrend
        data['long_channel_pearson'] = long_pearson
        data['short_channel_lower'] = short_lower

        relaxed_enabled = bool(self.config['trend_relaxed_entry'])
        relaxed_gap = max(0.0, float(self.config['trend_relaxed_min_gap']))
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

        stop_loss_pct = max(0.0, float(self.config['trend_stop_loss_pct']))



        # 底背离检测（买入信号）
        bullish_divergence_enabled = bool(self.config['trend_bullish_divergence_enabled'])
        if bullish_divergence_enabled:
            bullish_divergence_signals = self._detect_bullish_divergence(data)
            data['bullish_divergence_signal'] = bullish_divergence_signals
        else:
            bullish_divergence_signals = pd.Series(False, index=data.index)
            data['bullish_divergence_signal'] = bullish_divergence_signals
        
        # W底形态检测（买入信号）
        w_bottom_enabled = bool(self.config['trend_w_bottom_enabled'])
        if w_bottom_enabled:
            w_bottom_signals = self._detect_w_bottom(data)
            data['w_bottom_signal'] = w_bottom_signals
        else:
            w_bottom_signals = pd.Series(False, index=data.index)
            data['w_bottom_signal'] = w_bottom_signals

        # 主升浪检测（需要在所有技术指标计算完成后进行）
        main_wave_enabled = bool(self.config['trend_main_wave_enabled'])
        if main_wave_enabled:
            main_wave_signals = self._detect_main_wave_signals(data)
            data['main_wave_signal'] = main_wave_signals
        else:
            main_wave_signals = pd.Series(False, index=data.index)
            data['main_wave_signal'] = main_wave_signals

        # 溢价/折价区间计算（价格位置分析）
        # 在120天周期内计算价格相对位置（0-100%）
        lookback = 120
        rolling_high = data['close'].rolling(window=lookback, min_periods=1).max()
        rolling_low = data['close'].rolling(window=lookback, min_periods=1).min()
        price_range = rolling_high - rolling_low

        # 避免除以零
        price_position = pd.Series(0.5, index=data.index)  # 默认50%（中性）
        valid_range = price_range > 0
        price_position[valid_range] = (
            (data['close'][valid_range] - rolling_low[valid_range]) /
            price_range[valid_range]
        )

        data['price_position'] = price_position
        in_extreme_discount = price_position < 0.25
        in_discount = (price_position >= 0.25) & (price_position < 0.50)
        in_premium = (price_position >= 0.50) & (price_position < 0.75)
        in_extreme_premium = price_position >= 0.75
        in_buy_zone = price_position < 0.50
        in_sell_zone = price_position >= 0.75
        data['in_extreme_discount'] = in_extreme_discount
        data['in_discount'] = in_discount
        data['in_premium'] = in_premium
        data['in_extreme_premium'] = in_extreme_premium
        data['in_buy_zone'] = in_buy_zone
        data['in_sell_zone'] = in_sell_zone

        # 入场条件：成交量+M顶过滤（选择性应用）
        standard_entry = (
            (direction == 1) &
            data['is_heikin_bullish'] &
            (data['golden_cross'] | rsi_relaxed_condition) &
            lr_filter_condition &
            htf_bias &
            (~data['volume_weak']) &  # 成交量过滤
            (~data['is_m_top']) &  # M顶过滤
            (~data['atr_expanding'])  # ATR波动率过滤
        )

        # 双通道买点：作为独立的加仓信号（恢复原版，不过滤）
        dual_channel_entry = pd.Series(False, index=data.index)
        if dual_channel_enabled:
            dual_channel_entry = data['dual_channel_signal']

        # 折价区补充买入：在极度折价区（0-25%）且趋势向上时额外买入
        discount_zone_entry = (
            in_extreme_discount &
            (direction == 1) &
            data['is_heikin_bullish']
        )

        # 底背离入场条件（独立生效，不需要其他确认）
        divergence_entry = pd.Series(False, index=data.index)
        if bullish_divergence_enabled:
            divergence_entry = bullish_divergence_signals
        
        # W底形态入场条件（独立生效）
        w_bottom_entry = pd.Series(False, index=data.index)
        if w_bottom_enabled:
            # W底信号独立生效
            w_bottom_entry = w_bottom_signals
            # 调试：检查W底信号数量
            w_bottom_count = w_bottom_signals.sum()
            if w_bottom_count > 0:
                logger.info(f"[W底买入] 检测到{w_bottom_count}个W底信号，准备生成买入条件")

        # 布林带指标（用于EH做T等多处）
        from .indicators import bollinger_bands
        bb_upper, bb_middle, bb_lower, bb_width, bb_percent = bollinger_bands(data['close'], period=20, std_dev=2.0)
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        data['bb_width'] = bb_width
        data['bb_percent'] = bb_percent

        # BB宽度辅助震荡检测（捕获Aroon无法识别的长周期震荡）
        if bool(self.config.get('sideways_bb_width_enabled', False)):
            _sw_bb_width_thresh = float(self.config.get('sideways_bb_width_threshold', 0.10))
            _bb_width_sideways = (bb_width < _sw_bb_width_thresh)
            data['is_sideways'] = data['is_sideways'] | _bb_width_sideways

        # EH做T多因子指标计算（用于多因子评分高抛判定）
        # Stochastic K/D (14-period)
        _stoch_lowest = data['low'].rolling(14).min()
        _stoch_highest = data['high'].rolling(14).max()
        _stoch_denom = (_stoch_highest - _stoch_lowest).replace(0, np.nan)
        data['stoch_k'] = 100 * (data['close'] - _stoch_lowest) / _stoch_denom
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()

        # Williams %R (14-period, same window as Stochastic)
        data['williams_r'] = -100 * (_stoch_highest - data['close']) / _stoch_denom

        # CCI (20-period)
        _cci_tp = (data['high'] + data['low'] + data['close']) / 3
        _cci_ma = _cci_tp.rolling(20).mean()
        _cci_md = _cci_tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        data['cci_20'] = (_cci_tp - _cci_ma) / (0.015 * _cci_md.replace(0, np.nan))

        # Distance from MA20 (%) - bb_middle is the 20-day SMA
        data['dist_ma20'] = (data['close'] - bb_middle) / bb_middle.replace(0, np.nan) * 100

        # MA60 and distance from it (%)
        data['ma_60'] = data['close'].rolling(60).mean()
        data['dist_ma60'] = (data['close'] - data['ma_60']) / data['ma_60'].replace(0, np.nan) * 100

        # RSI 14-period (for scoring, separate from fast_rsi which may be 5-period)
        data['rsi_14'] = rsi_indicator(data['close'], period=14)

        # Linear regression slope (10-day, normalized % per day)
        def _lr_slope_norm(x):
            if np.any(np.isnan(x)):
                return np.nan
            slope = np.polyfit(np.arange(len(x)), x, 1)[0]
            mean_val = np.mean(x)
            return slope / mean_val * 100 if mean_val != 0 else 0
        data['lr_slope_10'] = data['close'].rolling(10).apply(_lr_slope_norm, raw=True)

        # MFI (Money Flow Index, 14-period) - 量价RSI，做T超买超卖信号
        _mfi_tp = (data['high'] + data['low'] + data['close']) / 3
        _mfi_raw_money_flow = _mfi_tp * data['volume']
        _mfi_tp_change = _mfi_tp.diff()
        _mfi_pos_flow = (_mfi_raw_money_flow * (_mfi_tp_change > 0).astype(float)).rolling(14).sum()
        _mfi_neg_flow = (_mfi_raw_money_flow * (_mfi_tp_change < 0).astype(float)).rolling(14).sum()
        _mfi_neg_flow = _mfi_neg_flow.replace(0, np.nan)
        data['mfi_14'] = 100 - (100 / (1 + _mfi_pos_flow / _mfi_neg_flow))

        # ATR百分比（ATR / close * 100），用于做T自适应回撤阈值
        data['atr_pct'] = data['atr'] / data['close'] * 100

        # EH做T反转确认指标
        data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
        # StochK死叉：K线下穿D线（从高位区域）
        data['stk_prev'] = data['stoch_k'].shift(1)
        data['std_prev'] = data['stoch_d'].shift(1)

        _sw_entry_bb = float(self.config.get('sideways_entry_bb_pct', 0.15))
        _sw_entry_rsi = float(self.config.get('sideways_entry_rsi', 28))
        sideways_entry = (
            data['is_sideways'] &                    # 震荡市（Aroon或BB宽度）
            (data['bb_percent'] <= _sw_entry_bb) &   # 价格接近下轨
            (data['fast_rsi'] < _sw_entry_rsi)       # RSI超卖
        )

        # 统计震荡策略触发情况
        sideways_count = data['is_sideways'].sum()
        sideways_entry_count = sideways_entry.sum()
        if sideways_count > 0:
            logger.info(f"[Aroon震荡] 震荡天数: {sideways_count}/{len(data)} ({sideways_count/len(data)*100:.1f}%), 入场信号: {sideways_entry_count}")

        # RSI动量加速入场：RSI从低位快速上升（3日变化>10）
        rsi_momentum_entry = (
            (direction == 1) &
            data['is_heikin_bullish'] &
            (data['rsi_momentum'] > 10) &
            (data['fast_rsi'] < 60) &
            (data['fast_rsi'].shift(3) < 40) &
            lr_filter_condition &
            (~data['volume_weak']) &
            (~data['is_m_top']) &
            (~data['atr_expanding'])
        )

        # BB Squeeze突破入场：波动率收缩→扩张+突破中轨
        bb_squeeze_entry = pd.Series(False, index=data.index)
        if bool(self.config.get('bb_squeeze_entry_enabled', False)):
            _bbs_lookback = int(self.config.get('bb_squeeze_entry_lookback', 20))
            _bbs_expansion_pct = float(self.config.get('bb_squeeze_entry_expansion_pct', 5))
            _bbs_require_dir = bool(self.config.get('bb_squeeze_entry_require_dir', True))
            _bbs_width_min = data['bb_width'].rolling(_bbs_lookback).min()
            # 近3天内曾处于squeeze状态（宽度接近N日最低）
            _bbs_at_min = data['bb_width'] <= _bbs_width_min * 1.02
            _bbs_was_squeezed = _bbs_at_min.shift(1).fillna(False) | _bbs_at_min.shift(2).fillna(False) | _bbs_at_min.shift(3).fillna(False)
            # 当前宽度已扩张
            _bbs_expanding = data['bb_width'] > _bbs_width_min * (1 + _bbs_expansion_pct / 100)
            # 价格突破中轨
            _bbs_above_mid = data['close'] > data['bb_middle']
            bb_squeeze_entry = (
                _bbs_was_squeezed & _bbs_expanding & _bbs_above_mid &
                data['is_heikin_bullish'] &
                (~data['volume_weak']) &
                (~data['is_m_top'])
            )
            if _bbs_require_dir:
                bb_squeeze_entry = bb_squeeze_entry & (direction == 1)

        # Stochastic超卖反转入场：KD金叉从超卖区
        stoch_reversal_entry = pd.Series(False, index=data.index)
        if bool(self.config.get('stoch_reversal_entry_enabled', False)):
            _sr_k_thresh = float(self.config.get('stoch_reversal_entry_k_thresh', 20))
            _sr_require_cross = bool(self.config.get('stoch_reversal_entry_require_cross', True))
            _sr_require_ha = bool(self.config.get('stoch_reversal_entry_require_ha', True))
            # 前一天K在超卖区
            _sr_was_oversold = data['stk_prev'] < _sr_k_thresh
            stoch_reversal_entry = _sr_was_oversold & (direction == 1) & (~data['volume_weak']) & (~data['is_m_top'])
            if _sr_require_cross:
                # K上穿D
                _sr_k_cross_d = (data['stk_prev'] < data['std_prev']) & (data['stoch_k'] > data['stoch_d'])
                stoch_reversal_entry = stoch_reversal_entry & _sr_k_cross_d
            if _sr_require_ha:
                stoch_reversal_entry = stoch_reversal_entry & data['is_heikin_bullish']

        # 入场质量过滤器（基于多因子分析，按类型选择性应用）
        data['standard_entry_raw'] = standard_entry.copy()

        entry_filter_enabled = self.config.get('entry_filter_enabled', True)
        if entry_filter_enabled:
            ma60 = data['close'].rolling(60).mean()
            vs_ma60 = ((data['close'] / ma60 - 1) * 100).fillna(0)

            # 参数（支持优化调参）
            ef_ma60_max = self.config.get('ef_ma60_max', 18)  # 价格超MA60 X%时过滤
            ef_macd_filter = self.config.get('ef_macd_filter', False)  # MACD<0时过滤（仅标准RSI）
            ef_vol_trend_max = self.config.get('ef_vol_trend_max', 0)  # 成交量趋势上限，0=不限
            ef_up_streak_max = self.config.get('ef_up_streak_max', 5)  # 连涨天数上限，0=不限

            # vs_ma60过滤：应用于所有可安全过滤的类型
            if ef_ma60_max > 0:
                ma60_block = vs_ma60 > ef_ma60_max
                standard_entry = standard_entry & ~ma60_block
                rsi_momentum_entry = rsi_momentum_entry & ~ma60_block
                discount_zone_entry = discount_zone_entry & ~ma60_block

            # MACD过滤：仅限标准RSI和RSI动量
            if ef_macd_filter:
                macd_block = data['macd_hist'] < 0
                standard_entry = standard_entry & ~macd_block
                rsi_momentum_entry = rsi_momentum_entry & ~macd_block

            # 成交量趋势过滤
            if ef_vol_trend_max > 0 and 'volume' in data.columns:
                vol_5d = data['volume'].rolling(5).mean()
                vol_prev_5d = data['volume'].shift(5).rolling(5).mean()
                vol_trend = (vol_5d / vol_prev_5d).fillna(1.0)
                vol_trend_block = vol_trend > ef_vol_trend_max
                standard_entry = standard_entry & ~vol_trend_block
                rsi_momentum_entry = rsi_momentum_entry & ~vol_trend_block

            # 连涨天数过滤
            if ef_up_streak_max > 0:
                up_streak = pd.Series(0, index=data.index)
                for i in range(1, len(data)):
                    if data['close'].iloc[i] > data['close'].iloc[i - 1]:
                        up_streak.iloc[i] = up_streak.iloc[i - 1] + 1
                    else:
                        up_streak.iloc[i] = 0
                streak_block = up_streak > ef_up_streak_max
                standard_entry = standard_entry & ~streak_block
                rsi_momentum_entry = rsi_momentum_entry & ~streak_block

        entry_condition = standard_entry | divergence_entry | dual_channel_entry | w_bottom_entry | discount_zone_entry | sideways_entry | rsi_momentum_entry | bb_squeeze_entry | stoch_reversal_entry

        # Gap Fade — 跳空回补入场（大幅低开时均值回归买入）
        if bool(self.config.get('gap_fade_enabled', False)):
            _, gap_down = self._detect_gap(data)
            fade_threshold = float(self.config.get('gap_fade_threshold', 3.0))
            gap_fade_entry = (
                (gap_down.abs() >= fade_threshold) &
                (direction == 1) &
                (data['fast_rsi'] < 35) &
                lr_filter_condition
            )
            entry_condition = entry_condition | gap_fade_entry
            logger.info(f"[Gap Fade] 跳空回补入场: {gap_fade_entry.sum()}条")

        # Hurst Exponent策略选择器 — 根据市场状态过滤不适合的入场类型
        hurst_enabled = bool(self.config['hurst_enabled'])
        if hurst_enabled:
            hurst_window = int(self.config['hurst_window'])
            mean_revert_threshold = float(self.config['hurst_mean_revert_threshold'])
            trending_threshold = float(self.config['hurst_trending_threshold'])

            # 每隔20天重新计算Hurst指数
            hurst_vals = pd.Series(0.5, index=data.index)
            for i in range(hurst_window, len(data), 20):
                h = self._calculate_hurst_exponent(data['close'].iloc[:i], hurst_window)
                hurst_vals.iloc[i:min(i+20, len(data))] = h

            data['hurst_exponent'] = hurst_vals

            # H < 0.45: 均值回归 → 只允许震荡入场、底背离、W底
            # H > trending_threshold: 趋势跟踪 → 只允许标准入场、双通道、RSI动量
            # 中间: 随机游走 → 全部允许
            for i in range(len(data)):
                if not entry_condition.iloc[i]:
                    continue
                h = hurst_vals.iloc[i]
                if h < mean_revert_threshold:
                    if standard_entry.iloc[i] or dual_channel_entry.iloc[i] or rsi_momentum_entry.iloc[i]:
                        entry_condition.iloc[i] = False
                elif h > trending_threshold:
                    if sideways_entry.iloc[i] or discount_zone_entry.iloc[i]:
                        entry_condition.iloc[i] = False

        # Volume Quality Filter — 过滤成交量质量差的入场
        vq_enabled = bool(self.config['volume_quality_enabled'])
        if vq_enabled and 'volume' in data.columns:
            min_score = int(self.config['vq_min_score'])
            vq_scores = pd.Series(0.0, index=data.index)
            for i in range(len(data)):
                vq_scores.iloc[i] = self._calculate_volume_quality_score(data, i)
            for i in range(len(data)):
                if entry_condition.iloc[i] and vq_scores.iloc[i] < min_score:
                    if divergence_entry.iloc[i] or w_bottom_entry.iloc[i]:
                        continue
                    entry_condition.iloc[i] = False
            data['volume_quality_score'] = vq_scores
            logger.info(f"[VQ Filter] 评分过滤后入场数: {entry_condition.sum()}")

        # Slope-Pearson Downtrend Filter — 过滤确定性持续下跌趋势
        slope_dt_enabled = bool(self.config['slope_downtrend_filter_enabled'])
        if slope_dt_enabled:
            slope_dt_period = int(self.config['slope_downtrend_period'])
            slope_dt_threshold = float(self.config['slope_downtrend_threshold'])
            slope_dt_pearson = float(self.config['slope_downtrend_pearson'])
            if slope_dt_period >= 180 and 'ultra_long_slope' in data.columns:
                slope_col = 'ultra_long_slope'
                pearson_col = 'ultra_long_channel_pearson'
            elif 'very_long_slope' in data.columns:
                slope_col = 'very_long_slope'
                pearson_col = 'very_long_channel_pearson'
            else:
                slope_col = None
                pearson_col = None
            if slope_col and pearson_col:
                before_count = entry_condition.sum()
                for i in range(len(data)):
                    if entry_condition.iloc[i]:
                        slope_val = data[slope_col].iloc[i]
                        pearson_val = data[pearson_col].iloc[i]
                        if not pd.isna(slope_val) and not pd.isna(pearson_val):
                            if slope_val < slope_dt_threshold and pearson_val > slope_dt_pearson:
                                if divergence_entry.iloc[i] or w_bottom_entry.iloc[i]:
                                    continue
                                entry_condition.iloc[i] = False
                after_count = entry_condition.sum()
                logger.info(f"[Slope DT] period={slope_dt_period} slope<{slope_dt_threshold} pearson>{slope_dt_pearson}: {before_count}->{after_count}")

        # 整理内存碎片，消除后续列赋值的PerformanceWarning
        data = data.copy()

        # 底背离买入保护：标记底背离买入，用于后续退出逻辑
        data['divergence_entry'] = divergence_entry

        # W底买入保护：标记W底买入
        data['w_bottom_entry'] = w_bottom_entry

        # 震荡市场买入：标记震荡市场买入
        data['sideways_entry'] = sideways_entry

        # 折价区补充买入：标记折价区买入
        data['discount_zone_entry'] = discount_zone_entry

        # RSI动量加速买入：标记RSI动量买入
        data['rsi_momentum_entry'] = rsi_momentum_entry

        # BB Squeeze突破买入
        data['bb_squeeze_entry'] = bb_squeeze_entry

        # Stochastic超卖反转买入
        data['stoch_reversal_entry'] = stoch_reversal_entry

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

            # 主升浪期间的退出条件更严格（移除强制止盈，完全依赖ATR趋势判断）
            exit_condition = basic_exit_condition & (~comprehensive_protection)
        else:
            # 无主升浪保护时，使用基础退出条件
            exit_condition = basic_exit_condition

        # 初始化回调买入标记列
        data['chase_pullback_entry'] = False

        # 底背离买入和W底买入需要特殊的退出处理
        position, entry_flags, exit_flags, stop_loss_flags, profit_target_flags, sideways_exit_type, swing_exit_flags, swing_rebuy_reasons, entry_reasons, exit_reasons = self._build_position_series_with_divergence(
            entry_condition,
            exit_condition,
            divergence_entry,
            w_bottom_entry,  # W底买入标记
            sideways_entry,  # 震荡市场买入标记
            data['close'],
            stop_loss_pct,
            data  # 传入完整数据用于MA计算
        )

        data = data.copy()  # 整理内存碎片（消除PerformanceWarning）— 必须在新列赋值前
        data['buy_signal'] = position
        data['entry_signal'] = entry_flags
        data['exit_signal'] = exit_flags
        data['stop_loss_exit'] = stop_loss_flags
        data['profit_target_exit'] = profit_target_flags  # 添加止盈标记
        data['sideways_exit_type'] = sideways_exit_type  # 震荡退出类型：1=上轨, 2=止盈, 3=止损
        data['swing_exit_type'] = swing_exit_flags  # 高抛低吸：1=swing sell, 2=swing rebuy, 3=giveup
        data['swing_rebuy_reason'] = swing_rebuy_reasons  # 高抛低吸回买原因
        data['stop_loss_pct'] = stop_loss_pct if stop_loss_pct > 0 else np.nan
        data['entry_reason'] = pd.Series(entry_reasons, index=data.index)
        data['exit_reason'] = pd.Series(exit_reasons, index=data.index)
        
        # 调试：输出买入/卖出信号数量和对应日期
        entry_count = np.sum(entry_flags)
        exit_count = np.sum(exit_flags)
        logger.info(f"[信号统计] 买入信号数量: {entry_count}, 卖出信号数量: {exit_count}")
        
        # 输出所有买入和卖出信号的日期
        if 'date' in data.columns:
            entry_dates = data[entry_flags == 1]['date'].tolist()
            exit_dates = data[exit_flags == 1]['date'].tolist()
            logger.info(f"[买入信号日期] {entry_dates}")
            logger.info(f"[卖出信号日期] {exit_dates}")
            
            # 输出buy_signal在买入和卖出日期的值
            logger.info("[buy_signal状态检查]")
            for d in ['2023-07-28', '2023-07-31', '2024-12-24']:
                if d in data['date'].values:
                    idx = data[data['date'] == d].index[0]
                    signal = position[idx]
                    entry = entry_flags[idx]
                    exit_f = exit_flags[idx]
                    logger.info(f"  {d}: buy_signal={signal}, entry_flag={entry}, exit_flag={exit_f}")

        return data, None

    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """输出最新的策略信号"""
        if df is None or len(df) == 0:
            return {'signal': 'NO_DATA', 'reason': '数据不足'}

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        exit_ma_filter_enabled = bool(self.config['trend_exit_use_ma_filter'])
        lr_filter_enabled = bool(self.config['trend_lr_filter_enabled'])

        signal = 'HOLD'
        reasons: List[str] = []
        strength = 1

        stop_loss_pct = float(self.config['trend_stop_loss_pct'])

        # 【关键修复】判断是否为"新入场"：entry_signal=1 且 前一天未持仓(buy_signal=0)
        # 如果前一天已经持仓，则当前应为"持有"而非"买入"，避免重复发送买入提醒
        is_new_entry = (latest.get('entry_signal', 0) == 1 and previous.get('buy_signal', 0) == 0)

        if is_new_entry:
            signal = 'BUY'
            
            # 检查是否为底背离入场
            if latest.get('bullish_divergence_signal', False):
                reasons.append("检测到连续底背离信号（独立生效）")
                strength = 4  # 底背离信号强度较高
            # 检查是否为W底买点
            elif latest.get('w_bottom_signal', False):
                reasons.append("检测到W底形态（双底确认）")
                reasons.append("两个低点间隔>30天且价格接近")
                strength = 4  # W底信号强度较高
            # 检查是否为双通道买点
            elif latest.get('dual_channel_signal', False):
                ultra_pearson = latest.get('ultra_long_channel_pearson', 0)
                very_pearson = latest.get('very_long_channel_pearson', 0)
                long_pearson = latest.get('long_channel_pearson', 0)
                reasons.append(f"三重趋势确认回调买点(180日R={ultra_pearson:.2f}+120日R={very_pearson:.2f}+60日R={long_pearson:.2f})")
                reasons.append("价格回调至20日通道下轨")
                strength = 4
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
        logger.info(f"[回测统计] trades数量: {len(trades)}")
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
                # 获取买入原因，如果为空则使用默认值
                entry_reason = buy_row.get('entry_reason', '')
                if not entry_reason or entry_reason.strip() == '':
                    entry_reason = 'RSI趋势买入'
                    
                buy_points.append({
                    'date': trade['buy_date'],
                    'price': trade['buy_price'],
                    'reason': entry_reason,
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
        divergence_enabled = bool(self.config['trend_bullish_divergence_enabled'])
        if not divergence_enabled:
            return pd.Series(False, index=data.index)
            
        lookback = int(self.config['trend_divergence_lookback'])
        min_consecutive = int(self.config['trend_divergence_min_consecutive'])
        
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
                    
                    # 底背离质量过滤：要求从近期高点下跌超过min_drop_pct%
                    div_min_drop = float(self.config.get('trend_divergence_min_drop_pct', 0))
                    pass_quality_filter = True
                    if div_min_drop > 0 and signal_idx >= lookback:
                        recent_high = np.max(close[signal_idx - lookback:signal_idx])
                        if recent_high > 0:
                            drop_pct = (recent_high - close[signal_idx]) / recent_high * 100
                            if drop_pct < div_min_drop:
                                pass_quality_filter = False

                    if pass_slope_filter and pass_quality_filter:
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
        min_gain = self.config['trend_main_wave_min_gain'] / 100  # 降低到10%
        min_days = self.config['trend_main_wave_min_days']  # 降低到3天
        rsi_threshold = self.config['trend_main_wave_rsi_threshold']  # 提高到80
        volume_factor = self.config['trend_main_wave_volume_factor']  # 降低到1.2倍
        
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
            # 保存日期索引到列中
            data = data.sort_index()
            data['date'] = data.index
            data = data.reset_index(drop=True)

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
        # 强制禁用多时间框架(MTF)：避免信号滞后/漏买。
        # 注意：此处为“写死”逻辑，外部配置将被忽略。
        mtf_enabled = False
        if not mtf_enabled:
            # 如果禁用多时间框架，返回全部为True的序列（不限制）
            return pd.Series(True, index=data.index), {'status': 'disabled'}
            
        mtf_ratio = max(2, int(self.config['trend_mtf_ratio']))
        min_periods = max(20, int(self.config['trend_mtf_min_periods']))
        adaptive_mode = bool(self.config['trend_mtf_adaptive_mode'])
        trend_lookback = max(10, int(self.config['trend_mtf_trend_lookback']))
        early_entry = bool(self.config['trend_mtf_early_entry'])
        early_threshold = float(self.config['trend_mtf_early_threshold'])
        strict_mode = bool(self.config['trend_mtf_strict_mode'])
        
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
            fast_period = int(self.config['trend_rsi_fast_period'])
            slow_period = int(self.config['trend_rsi_slow_period'])
            atr_period = int(self.config['trend_atr_period'])
            atr_multiplier = float(self.config['trend_atr_multiplier'])
            use_close = bool(self.config['trend_use_close_for_extrema'])
            
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
                # 重要：避免未来函数（lookahead）
                # htf_bullish_bias[k] 是由第 k 个高周期K线(包含 ratio 根低周期K线)计算得到，
                # 只有在该高周期K线收盘(即 i % ratio == ratio-1)之后才“已知”。
                # 因此：
                # - 若当前低周期K线不是高周期收盘日，则只能使用上一个已完成高周期的bias
                # - 若是高周期收盘日，则可以使用当前高周期的bias
                current_htf_idx = i // mtf_ratio
                is_htf_close = (i % mtf_ratio) == (mtf_ratio - 1)
                latest_completed_htf_idx = current_htf_idx if is_htf_close else (current_htf_idx - 1)

                if 0 <= latest_completed_htf_idx < len(htf_bullish_bias):
                    bias_value = htf_bullish_bias.iloc[latest_completed_htf_idx]
                    mode_info = (
                        mode_sequence[latest_completed_htf_idx]
                        if latest_completed_htf_idx < len(mode_sequence)
                        else {'mode': 'standard', 'trend': 'neutral'}
                    )
                    if pd.isna(bias_value):
                        bias_value = True  # 默认不限制
                else:
                    bias_value = True  # 数据不足/尚无已完成高周期K线时不限制
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

    def _calculate_hurst_exponent(self, prices: pd.Series, window: int = 100) -> float:
        """计算Hurst指数 (R/S分析法)

        H < 0.5: 均值回归, H = 0.5: 随机游走, H > 0.5: 趋势性
        """
        if len(prices) < window:
            return 0.5
        prices_arr = prices.iloc[-window:].values
        returns = np.diff(np.log(prices_arr))
        if len(returns) < 10:
            return 0.5
        lags = range(2, min(20, len(returns)//2))
        rs_vals = []
        for lag in lags:
            rs = []
            for i in range(0, len(returns), lag):
                if i + lag > len(returns):
                    break
                subset = returns[i:i+lag]
                if len(subset) < 2:
                    continue
                mean_ret = np.mean(subset)
                cumdev = np.cumsum(subset - mean_ret)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(subset)
                if s > 0:
                    rs.append(r / s)
            if rs:
                rs_vals.append(np.mean(rs))
        if len(rs_vals) < 2:
            return 0.5
        log_lags = np.log(list(lags)[:len(rs_vals)])
        log_rs = np.log(rs_vals)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        return max(0.0, min(1.0, coeffs[0]))

    def _detect_gap(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """检测跳空百分比 (向上, 向下)"""
        gap_pct = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1) * 100).fillna(0)
        gap_up = gap_pct.clip(lower=0)
        gap_down = gap_pct.clip(upper=0)
        return gap_up, gap_down

    def _calculate_volume_quality_score(self, data: pd.DataFrame, idx: int) -> float:
        """计算成交量质量评分（0-100）: 一致性(40分) + 强度(60分)"""
        if idx < 10:
            return 50
        volume = data.get('volume', pd.Series([0]*len(data), index=data.index))
        close = data['close']
        window_start = max(0, idx - 9)
        window_close = close.iloc[window_start:idx+1]
        window_volume = volume.iloc[window_start:idx+1]
        consistency_score = 0
        for i in range(1, len(window_close)):
            price_up = window_close.iloc[i] > window_close.iloc[i-1]
            vol_up = window_volume.iloc[i] > window_volume.iloc[i-1]
            if (price_up and vol_up) or (not price_up and not vol_up):
                consistency_score += 1
        consistency_score = (consistency_score / (len(window_close) - 1)) * 40
        vol_ma = window_volume.mean()
        current_vol = volume.iloc[idx]
        if vol_ma > 0:
            vol_ratio = current_vol / vol_ma
            if vol_ratio >= 2.0:
                strength_score = 60
            elif vol_ratio >= 1.5:
                strength_score = 50 + (vol_ratio - 1.5) * 20
            elif vol_ratio >= 1.0:
                strength_score = 40 + (vol_ratio - 1.0) * 20
            else:
                strength_score = vol_ratio * 40
        else:
            strength_score = 0
        return min(100, max(0, consistency_score + strength_score))

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
                               stop_loss_pct: float,
                               data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """根据条件构造持仓序列"""
        n = len(entry_condition)
        position = np.zeros(n, dtype=int)
        entry_flags = np.zeros(n, dtype=int)
        exit_flags = np.zeros(n, dtype=int)
        stop_flags = np.zeros(n, dtype=int)
        in_position = False
        entry_price = None

        # 追高冷却期状态
        chase_cooldown_active = False
        chase_peak_price = 0
        chase_start_idx = 0
        chase_rise_vol_ratio = 0
        chase_cooldown_days = 30
        chase_pullback_pct = 4
        chase_hard_block = False

        for i in range(n):
            entry_active = bool(entry_condition.iloc[i]) if not pd.isna(entry_condition.iloc[i]) else False
            exit_active = bool(exit_condition.iloc[i]) if not pd.isna(exit_condition.iloc[i]) else False
            curr_price = price_series.iloc[i] if i < len(price_series) else np.nan

            # 追高冷却期逻辑
            avoid_extreme_chase = False
            chase_pullback_buy = False
            if data is not None and i < len(data):
                row = data.iloc[i]
                ma_120 = row.get('ma_120', np.nan)
                short_gain_10d = row.get('short_gain_10d', np.nan)

                price_vs_ma120 = ((curr_price / ma_120 - 1) * 100) if not np.isnan(ma_120) and not np.isnan(curr_price) and ma_120 > 0 else 0

                is_chase_condition = (price_vs_ma120 > 15) and (not np.isnan(short_gain_10d) and short_gain_10d > 15)

                if is_chase_condition and not chase_cooldown_active and not in_position:
                    chase_cooldown_active = True
                    chase_peak_price = curr_price if not np.isnan(curr_price) else 0
                    chase_start_idx = i
                    vol = row.get('volume', np.nan)
                    vol_ma20 = row.get('volume_ma20', np.nan)
                    if not np.isnan(vol) and not np.isnan(vol_ma20) and vol_ma20 > 0:
                        chase_rise_vol_ratio = vol / vol_ma20
                    else:
                        chase_rise_vol_ratio = 1.0
                    # 硬屏蔽：巨量(>2x)+10日涨>30% → 完全不允许回调买入
                    chase_hard_block = (chase_rise_vol_ratio > 2.0
                                        and not np.isnan(short_gain_10d) and short_gain_10d > 30)
                    avoid_extreme_chase = True
                elif chase_cooldown_active and not in_position:
                    if not np.isnan(curr_price) and curr_price > chase_peak_price:
                        chase_peak_price = curr_price
                    days_in_cooldown = i - chase_start_idx
                    if days_in_cooldown > chase_cooldown_days:
                        chase_cooldown_active = False
                        chase_hard_block = False
                    elif chase_peak_price > 0 and not np.isnan(curr_price):
                        if chase_hard_block:
                            avoid_extreme_chase = True
                        else:
                            drop_from_peak = (1 - curr_price / chase_peak_price) * 100
                            chase_cleared = not is_chase_condition
                            # 巨量暴涨过滤：追高时成交量>1.8x不执行回调买入
                            rise_vol_ok = chase_rise_vol_ratio <= 1.8
                            # 跌速检查：要求>=0.6%/天
                            speed_ok = True
                            if days_in_cooldown > 0:
                                drop_speed = drop_from_peak / days_in_cooldown
                                speed_ok = drop_speed >= 0.6
                            if drop_from_peak >= chase_pullback_pct and entry_active and rise_vol_ok and speed_ok:
                                chase_pullback_buy = True
                                chase_cooldown_active = False
                            else:
                                avoid_extreme_chase = True

            if not in_position and (entry_active and not avoid_extreme_chase) or (chase_pullback_buy and not in_position):
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
                                              w_bottom_entry: pd.Series,
                                              sideways_entry: pd.Series,
                                              price_series: pd.Series,
                                              stop_loss_pct: float,
                                              data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """根据条件构造持仓序列（支持底背离、W底和震荡市场买入保护）

        Args:
            entry_condition: 综合入场条件（标准入场 | 底背离 | W底 | 震荡）
            exit_condition: 退出条件
            divergence_entry: 底背离入场信号
            w_bottom_entry: W底入场信号
            sideways_entry: 震荡市场入场信号
            price_series: 价格序列
            stop_loss_pct: 止损百分比
            data: 完整数据

        Returns:
            position, entry_flags, exit_flags, stop_flags, profit_target_flags, sideways_exit_type
        """
        min_hold_days = int(self.config['trend_divergence_min_hold_days'])
        profit_target_pct = float(self.config['trend_divergence_profit_target'])
        ignore_rsi_exit = bool(self.config['trend_divergence_ignore_rsi_exit'])
        use_rsi_trend = bool(self.config['trend_divergence_use_rsi_trend'])
        rsi_decline_threshold = float(self.config['trend_divergence_rsi_decline_threshold'])
        
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
        sideways_exit_type = np.zeros(n, dtype=int)  # 震荡退出类型：1=上轨退出, 2=止盈, 3=止损
        entry_reasons = [''] * n  # 真实入场原因（在回测循环中直接记录）
        exit_reasons = [''] * n   # 真实出场原因（在回测循环中直接记录）
        in_position = False
        entry_price = None
        is_divergence_entry = False  # 标记当前持仓是否为底背离买入
        is_w_bottom_entry = False  # 标记当前持仓是否为W底买入
        is_sideways_entry = False  # 标记当前持仓是否为震荡市场买入
        # 震荡退出参数（从config读取，原先硬编码）
        _sw_exit_bb_upper = float(self.config.get('sideways_exit_bb_upper', 0.85))
        _sw_exit_rsi_upper = float(self.config.get('sideways_exit_rsi_upper', 70))
        _sw_exit_tp_pct = float(self.config.get('sideways_exit_tp_pct', 8.5))
        _sw_exit_sl_pct = float(self.config.get('sideways_exit_sl_pct', 5.0))
        w_bottom_price = None  # 记录W底的最低价格（用于止损）
        w_bottom_gap = None  # 记录W底的间隔天数（用于动态缓冲期）
        # W底退出模式：True=使用标准退出逻辑（与其他入场类型相同），False=使用W底专属退出
        _wb_std_exit = bool(self.config.get('w_bottom_use_standard_exit', False))
        # 自适应止损：熊市使用更紧的止损
        _adaptive_sl_enabled = bool(self.config.get('adaptive_stop_loss_enabled', False))
        _adaptive_sl_bear_pct = float(self.config.get('adaptive_stop_loss_bear_pct', 5.0))  # 熊市止损%
        _trade_stop_loss = stop_loss_pct  # 每笔交易的实际止损（可能被regime调整）
        # Hard Loss Cap — 硬性最大亏损上限
        hard_loss_cap_enabled = bool(self.config['hard_loss_cap_enabled'])
        hard_loss_cap_pct = float(self.config['hard_loss_cap_pct'])
        # W底缓冲期参数
        wb_buffer_stop_pct = float(self.config['wb_buffer_stop_pct'])
        wb_buffer_profit_pct = float(self.config['wb_buffer_profit_pct'])
        hold_days = 0  # 持仓天数
        entry_rsi = None  # 记录买入时的RSI值

        # 反弹卖出参数（bounce exit）：避免暴跌中卖出，等待反弹再卖
        bounce_exit_enabled = bool(self.config.get('bounce_exit_enabled', True))
        bounce_exit_drop_threshold = float(self.config.get('bounce_exit_drop_threshold', -3.0))  # 触发延迟的当日跌幅阈值%（-3%：只在真正暴跌时才延迟，小幅回调直接退出）
        bounce_exit_max_wait = int(self.config.get('bounce_exit_max_wait', 1))  # 最大等待天数
        bounce_exit_bounce_pct = float(self.config.get('bounce_exit_bounce_pct', 1.0))  # 反弹幅度要求%（vs卖出信号价）
        pending_exit = False  # 是否处于待卖出状态
        pending_exit_price = 0  # 触发卖出信号时的价格
        pending_exit_days = 0  # 等待天数

        # 止盈保护参数（浮盈超过trigger%后，回到入场价+level%就止损保本）
        trailing_stop_trigger = float(self.config['trailing_stop_trigger'])  # 0=关闭，浮盈X%后激活保本止损
        trailing_stop_level = float(self.config['trailing_stop_level'])  # 回到入场价就卖
        # 双层trailing stop: 更高利润时使用更紧的floor
        trailing_stop_trigger2 = float(self.config['trailing_stop_trigger2'])  # 0=关闭
        trailing_stop_level2 = float(self.config['trailing_stop_level2'])
        trailing_stop_confirm = int(self.config['trailing_stop_confirm'])  # 确认K线数, 0=立即卖出
        # 恐慌过滤：当日跌幅超过阈值时不触发trailing stop（认为是恐慌性下杀，可能V型反转）
        trailing_stop_panic_skip = float(self.config['trailing_stop_panic_skip'])  # 0=关闭, 如-5表示当日跌>5%时不卖
        # 平稳期突跌过滤: 前N天最大单日跌幅>阈值(平稳)→突跌可能是恐慌→给1天确认
        trailing_stop_calm_threshold = float(self.config['trailing_stop_calm_threshold'])  # 0=关闭, 如-3表示前5天最大日跌>-3%算平稳
        trailing_stop_calm_lookback = int(self.config['trailing_stop_calm_lookback'])  # 回看天数
        trailing_stop_active = False  # 当前是否已激活
        _ts_pending = False  # trailing stop是否在等待确认
        _ts_pending_days = 0  # 已等待确认的天数
        max_profit_in_trade = 0  # 当前交易中的最大浮盈%

        dynamic_profit_trigger = float(self.config.get('dynamic_profit_trigger', 0))  # 浮盈达X%后激活动态止盈,0=关闭
        dynamic_profit_drawback = float(self.config.get('dynamic_profit_drawback', 0))  # 从最高点回撤Y%就卖
        dynamic_profit_active = False  # 是否已激活

        # 成交量分布退出：在超买区间检测到机构派发时提前退出
        dist_exit_enabled = bool(self.config['dist_exit_enabled'])
        dist_exit_min_profit = float(self.config['dist_exit_min_profit'])
        dist_exit_lookback = int(self.config['dist_exit_lookback'])
        dist_exit_vol_threshold = float(self.config['dist_exit_vol_threshold'])
        dist_exit_count = int(self.config['dist_exit_count'])

        # 滞涨退出：浮盈达标后连续N天未创新高 → 动量耗尽信号
        stale_peak_enabled = bool(self.config['stale_peak_enabled'])
        stale_peak_min_profit = float(self.config['stale_peak_min_profit'])
        stale_peak_max_days = int(self.config['stale_peak_max_days'])
        _days_since_peak = 0

        # 放量阴线+均线偏离退出：高位出货信号（放量阴线+价格远离MA）
        dist_madev_exit_enabled = bool(self.config['dist_madev_exit_enabled'])
        dist_madev_exit_min_profit = float(self.config['dist_madev_exit_min_profit'])
        dist_madev_exit_vol_mult = float(self.config['dist_madev_exit_vol_mult'])
        dist_madev_exit_ma_period = int(self.config['dist_madev_exit_ma_period'])
        dist_madev_exit_dev_pct = float(self.config['dist_madev_exit_dev_pct'])

        # 多指标超买集群退出：利润在mid区间时，多个振荡指标同时超买则退出
        ob_cluster_exit_enabled = bool(self.config.get('ob_cluster_exit_enabled', False))
        ob_cluster_exit_min_profit = float(self.config.get('ob_cluster_exit_min_profit', 12))
        ob_cluster_exit_max_profit = float(self.config.get('ob_cluster_exit_max_profit', 22))
        ob_cluster_exit_min_count = int(self.config.get('ob_cluster_exit_min_count', 3))
        ob_cluster_exit_rsi_thresh = float(self.config.get('ob_cluster_exit_rsi_thresh', 70))
        ob_cluster_exit_stk_thresh = float(self.config.get('ob_cluster_exit_stk_thresh', 80))
        ob_cluster_exit_cci_thresh = float(self.config.get('ob_cluster_exit_cci_thresh', 200))
        ob_cluster_exit_mfi_thresh = float(self.config.get('ob_cluster_exit_mfi_thresh', 80))

        # 放量冲高回落退出：大阳线上影>实体+收盘下半区+放量
        vol_climax_exit_enabled = bool(self.config.get('vol_climax_exit_enabled', False))
        vol_climax_exit_min_profit = float(self.config.get('vol_climax_exit_min_profit', 8))
        vol_climax_exit_vol_mult = float(self.config.get('vol_climax_exit_vol_mult', 2.5))
        vol_climax_exit_require_new_high = bool(self.config.get('vol_climax_exit_require_new_high', True))

        # ROC动量衰竭退出：ROC正值但连续下降
        roc_fade_exit_enabled = bool(self.config.get('roc_fade_exit_enabled', False))
        roc_fade_exit_min_profit = float(self.config.get('roc_fade_exit_min_profit', 8))
        roc_fade_exit_declining_days = int(self.config.get('roc_fade_exit_declining_days', 3))
        roc_fade_exit_roc_floor = float(self.config.get('roc_fade_exit_roc_floor', 0))

        # 早期止损收紧：前N天使用更紧的止损，之后恢复正常止损
        early_stop_days = int(self.config['early_stop_days'])  # 0=关闭，>0=前N天收紧止损
        early_stop_loss_pct = float(self.config['early_stop_loss_pct'])  # 早期止损百分比

        # 负动量提前退出：亏损>X%且动量恶化时提前退出（不等止损线）
        neg_momentum_exit_enabled = bool(self.config.get('neg_momentum_exit_enabled', False))
        neg_momentum_loss_threshold = float(self.config.get('neg_momentum_loss_threshold', 3.0))  # 亏损达X%时检查动量
        neg_momentum_min_days = int(self.config.get('neg_momentum_min_days', 3))  # 最少持仓N天后才检查
        neg_momentum_rsi_declining_days = int(self.config.get('neg_momentum_rsi_declining_days', 2))  # RSI连续下降N天

        # 峰值相对追踪止盈：基于最高价回撤而非入场价（更好保护中间利润）
        peak_trailing_enabled = bool(self.config.get('peak_trailing_enabled', False))
        peak_trailing_trigger = float(self.config.get('peak_trailing_trigger', 10))  # 浮盈达X%后激活峰值追踪
        peak_trailing_pct = float(self.config.get('peak_trailing_pct', 5))  # 从最高价回撤X%时退出
        peak_trailing_min_floor = float(self.config.get('peak_trailing_min_floor', 0))  # 最低保护线（浮盈%），0=允许回到入场价

        # 亏损冷却期：止损退出后N天内不再入场（减少反复止损）
        loss_cooldown_days = int(self.config.get('loss_cooldown_days', 0))  # 0=关闭
        _last_loss_exit_idx = -9999  # 上次止损退出的bar index

        # 入场成交量确认：要求入场日成交量达到均量X倍（过滤弱信号）
        entry_vol_confirm_mult = float(self.config['entry_vol_confirm_mult'])  # 0=关闭, 如1.0=要求放量

        # MA趋势对齐过滤：要求短期MA > 长期MA（只在上升趋势入场）
        entry_ma_align_enabled = bool(self.config.get('entry_ma_align_enabled', False))
        entry_ma_align_short = int(self.config.get('entry_ma_align_short', 20))  # 短MA周期
        entry_ma_align_long = int(self.config.get('entry_ma_align_long', 60))  # 长MA周期
        # 预计算MA对齐所需的均线（避免在循环内重复计算）
        if entry_ma_align_enabled and data is not None:
            for _ma_p in [entry_ma_align_short, entry_ma_align_long]:
                _ma_col = f'ma_{_ma_p}'
                if _ma_col not in data.columns:
                    data[_ma_col] = data['close'].rolling(_ma_p).mean()

        # 大盘regime：用于入场过滤和/或自适应止损
        market_regime_enabled = bool(self.config.get('market_regime_enabled', False))  # 入场过滤开关
        market_regime_ma = int(self.config.get('market_regime_ma_period', 120))
        market_regime_index = str(self.config.get('market_regime_index_code', '000300'))  # 默认沪深300
        market_regime_buffer = float(self.config.get('market_regime_buffer_pct', 0))  # 缓冲区%
        _regime_signal = None
        # 加载regime信号：入场过滤或自适应止损任一启用时都需要
        _need_regime = market_regime_enabled or bool(self.config.get('adaptive_stop_loss_enabled', False))
        if _need_regime and data is not None and 'date' in data.columns:
            _regime_signal = self._load_index_regime(
                market_regime_index, market_regime_ma,
                start_date=str(data['date'].iloc[0])[:10] if len(data) > 0 else '2018-01-01',
                buffer_pct=market_regime_buffer
            )

        # 主升浪延长持仓：退出信号时浮盈>35%+MA120上升+持仓>25天 → 改用MA120退出线
        extended_hold_active = False
        extended_hold_trigger_profit = 0.0  # 触发时的浮盈%（用于计算回撤底线）
        extended_hold_max_profit = 0.0  # 延长持仓期间最高浮盈
        eh_profit_threshold = float(self.config['extended_hold_profit_threshold'])
        eh_drawdown_limit = float(self.config.get('extended_hold_drawdown', 5))  # 从触发浮盈回撤X%后退出（R7最优：5）
        eh_peak_trailing = float(self.config['extended_hold_peak_trailing'])  # 从最高浮盈回撤X%后退出
        eh_peak_activation_offset = float(self.config['extended_hold_peak_activation_offset'])  # 峰值回撤激活：浮盈超过触发浮盈+Xpp后启动
        eh_gain_protection_ratio = float(self.config['extended_hold_gain_protection_ratio'])  # 比例保护：保护已有增益的X%（0=关闭）
        eh_swing_enabled = bool(self.config.get('extended_hold_swing_enabled', True))  # EH期间做T开关（R7最优：开启）
        eh_swing_rsi_threshold = float(self.config.get('eh_swing_rsi_threshold', 50))  # EH做T卖出RSI阈值（R7最优：50）
        eh_swing_bb_threshold = float(self.config.get('eh_swing_bb_threshold', 0.50))  # EH做T卖出BB阈值（R7最优：0.50）
        eh_swing_min_gain_above_trigger = float(self.config.get('eh_swing_min_gain_above_trigger', 0))  # 浮盈超过触发值Xpp才允许卖出（R8最优：0=无门槛）
        eh_swing_volume_surge_block = float(self.config.get('eh_swing_volume_surge_block', 2.5))  # 放量突破不卖
        eh_swing_rebuy_rsi = float(self.config.get('eh_swing_rebuy_rsi', 45))  # EH做T回买RSI阈值
        eh_swing_rebuy_bb = float(self.config.get('eh_swing_rebuy_bb', 0.35))  # EH做T回买BB阈值
        eh_swing_rebuy_max_above = float(self.config.get('eh_swing_rebuy_max_above', 0))  # 允许回买价超过卖价的最大百分比（0=只允许低于卖价回买）
        eh_swing_max_wait_days = int(self.config.get('eh_swing_max_wait_days', 0))  # EH做T最长等待天数（R7最优：0=无限）
        eh_swing_force_rebuy_premium = float(self.config.get('eh_swing_force_rebuy_premium', 999))  # 股价涨超卖价X%时强制回买（R7最优：999=关闭）
        eh_swing_rebuy_pullback_pct = float(self.config.get('eh_swing_rebuy_pullback_pct', 0))  # 回调低吸：股价从T卖后高点回撤X%时买回（0=关闭）
        eh_swing_rebuy_min_drop = float(self.config.get('eh_swing_rebuy_min_drop', 0))  # 最小跌幅%：价格需低于卖价X%才允许低吸接回（0=无要求）
        eh_swing_rebuy_min_wait = int(self.config.get('eh_swing_rebuy_min_wait', 0))  # 最少等待天数：T-sell后至少等N天再接回（0=无要求）
        eh_swing_rebuy_stk = float(self.config.get('eh_swing_rebuy_stk', 0))  # StochK接回阈值：StK<X时触发接回（0=不用StK判断）
        eh_swing_peak_drawdown = float(self.config.get('eh_swing_peak_drawdown', 0))  # 从峰值回撤Xpp触发做T卖出（0=关闭，只用RSI/BB卖）
        eh_swing_score_threshold = int(self.config.get('eh_swing_score_threshold', 0))  # 多因子评分阈值（0=使用旧RSI+BB逻辑，>=1使用评分系统）
        # 放量阴线信号驱动T卖参数
        eh_swing_vol_signal_enabled = bool(self.config['eh_swing_vol_signal_enabled'])  # 放量阴线触发T卖（0=关闭）
        eh_swing_vol_signal_mult = float(self.config['eh_swing_vol_signal_mult'])  # 放量倍数阈值（volume > X * MA20）
        eh_swing_vol_signal_lookback = int(self.config['eh_swing_vol_signal_lookback'])  # 回看天数
        eh_swing_vol_signal_count = int(self.config['eh_swing_vol_signal_count'])  # 需要N根放量阴线
        # 超买收紧止盈参数（不增加交易，只在超买后收紧trailing stop）
        eh_overbought_trailing = float(self.config.get('eh_overbought_trailing', 0))  # 超买后收紧trailing到X%（0=关闭）
        # EH做T运行时状态
        _eh_swing_active = False  # EH做T等待回买中
        _eh_swing_sell_price = 0.0  # EH做T卖出价格
        _eh_swing_original_entry = 0.0  # EH做T前的原始入场价（用于计算底线）
        _eh_swing_floor_price = 0.0  # EH底线的绝对价格
        _eh_swing_sell_idx = 0  # EH做T卖出位置索引（用于等待天数计算）
        _eh_swing_peak_after_sell = 0.0  # T卖后的最高价（用于回调低吸判断）
        _eh_swing_used = False  # 当前EH周期是否已使用过做T（限制每个EH只做一次）
        _eh_swing_rebuy_idx = 0  # 上次T-rebuy的索引（冷却期控制）
        # T-sell时保存的原始交易状态（T-rebuy时恢复，做T不影响原始交易逻辑）
        _eh_swing_saved_entry_price = 0.0
        _eh_swing_saved_hold_days = 0
        _eh_swing_saved_pending_exit = False
        _eh_swing_saved_trailing_stop_active = False
        _eh_swing_saved_ts_pending = False
        _eh_swing_saved_ts_pending_days = 0
        _eh_swing_saved_dynamic_profit_active = False
        _eh_swing_saved_max_profit_in_trade = 0.0
        _eh_swing_saved_is_divergence_entry = False
        _eh_swing_saved_is_w_bottom_entry = False
        _eh_swing_saved_is_sideways_entry = False
        _eh_swing_saved_eh_trigger_profit = 0.0
        _eh_swing_saved_eh_max_profit = 0.0
        eh_swing_cooldown_days = int(self.config.get('eh_swing_cooldown_days', 10))  # T-rebuy后N天内不允许再次T-sell（R7最优：10）
        # 确认后卖出参数（等待确认再高抛，避免卖飞）
        eh_swing_confirm_days = int(self.config.get('eh_swing_confirm_days', 0))  # 确认等待天数（0=立即卖出，不等确认）
        eh_swing_confirm_drop_pct = float(self.config.get('eh_swing_confirm_drop_pct', 2.0))  # 从信号日回撤X%确认为高点
        _eh_swing_confirming = False  # 正在等待确认中
        _eh_swing_signal_idx = 0  # 超买信号触发日的索引
        _eh_swing_signal_price = 0.0  # 信号日的价格（用于判断回撤）
        _eh_overbought_seen = False  # 当前EH周期是否已检测到超买（粘性标记，用于收紧trailing）
        # 多模式T-sell参数（基于海量EH峰值数据分析）
        eh_swing_armed_mode = bool(self.config.get('eh_swing_armed_mode', True))  # 武装模式：信号后追踪峰值，回撤时卖出（R7最优：开启）
        eh_swing_trailing_drop_pct = float(self.config.get('eh_swing_trailing_drop_pct', 2.7))  # 武装峰值回撤X%触发卖出（R7最优：2.7）
        eh_swing_armed_max_days = int(self.config.get('eh_swing_armed_max_days', 20))  # 武装状态最大天数
        eh_swing_ob_stk_threshold = float(self.config.get('eh_swing_ob_stk_threshold', 70))  # StochK超买阈值（R7最优：70）
        eh_swing_ob_min_count = int(self.config.get('eh_swing_ob_min_count', 1))  # 超买集群最少指标数（R8最优：1 of 3）
        eh_swing_dev_ma20_pct = float(self.config.get('eh_swing_dev_ma20_pct', 0))  # MA20偏离%触发（0=关闭）
        eh_swing_dev_ma60_pct = float(self.config.get('eh_swing_dev_ma60_pct', 0))  # MA60偏离%触发（0=关闭）
        # ATR自适应武装回撤：替代固定百分比，根据波动率调整卖出敏感度
        eh_swing_atr_adaptive = bool(self.config.get('eh_swing_atr_adaptive', False))  # 0=使用固定%, 1=ATR自适应
        eh_swing_atr_mult = float(self.config.get('eh_swing_atr_mult', 1.5))  # ATR乘数：回撤>atr_pct*mult触发卖出
        # MFI作为第4个超买指标
        eh_swing_mfi_threshold = float(self.config.get('eh_swing_mfi_threshold', 0))  # MFI超买阈值（0=不使用, 80=使用）
        # 武装模式运行时状态
        _eh_swing_armed = False  # 武装模式激活中
        _eh_swing_armed_idx = 0  # 武装模式开始索引
        _eh_swing_armed_peak = 0.0  # 武装后最高价格

        # 主升浪再入场：高盈利退出后120天内，绕过ef_ma60_max过滤
        post_wave_reentry_countdown = 0  # >0时允许再入场
        post_wave_reentry_window = int(self.config['post_wave_reentry_window'])
        pw_profit_threshold = float(self.config['post_wave_profit_threshold'])
        pw_min_hold = int(self.config['post_wave_min_hold'])
        pw_price_confirm_pct = float(self.config['post_wave_price_confirm_pct'])  # 价格突破确认：股价>退场价×(1+X%)才回补
        _pw_last_trade_profit = 0.0  # 持仓中的实时利润，退出后保持最后值
        _pw_last_trade_hold = 0  # 持仓天数，退出后保持最后值
        _pw_exit_price = 0.0  # EH退出时的价格（用于价格突破确认）

        # 追高冷却期参数（从config读取，支持优化调参）
        chase_cooldown_active = False
        chase_peak_price = 0
        chase_start_idx = 0
        chase_rise_vol_ratio = 0  # 记录追高时的成交量倍数
        chase_mode = self.config.get('chase_mode', 'cooldown')  # 'block', 'cooldown', 'none'
        chase_cooldown_days = self.config.get('chase_cooldown_days', 30)
        # 多Profile OR逻辑：chase_profiles是条件列表，满足任一即允许回调买入
        # 每个profile: {pullback_pct, require_cleared, vol_min_ratio, rise_vol_max, min_drop_speed}
        chase_profiles = self.config.get('chase_profiles', None)
        if chase_profiles is None:
            # 兼容旧版单参数配置（默认值为优化最优参数）
            chase_profiles = [{
                'pullback_pct': self.config.get('chase_pullback_pct', 4),
                'require_cleared': self.config.get('chase_require_cleared', False),
                'vol_min_ratio': self.config.get('chase_vol_min_ratio', 0),
                'rise_vol_max': self.config.get('chase_rise_vol_max', 1.8),
                'min_drop_speed': self.config.get('chase_min_drop_speed', 0.6),
            }]
        # 硬屏蔽条件：满足时完全不允许回调买入（比cooldown更严格）
        chase_hard_block_vol = self.config.get('chase_hard_block_vol', 2.0)  # 巨量阈值
        chase_hard_block_gain = self.config.get('chase_hard_block_gain', 30)  # 10日涨幅阈值
        chase_hard_block = False  # 当前是否处于硬屏蔽状态
        chase_max_drop_pct = self.config.get('chase_max_drop_pct', 0)  # 急跌跌幅上限，0=不限

        # 高抛低吸参数
        swing_trade_enabled = bool(self.config['swing_trade_enabled'])
        swing_min_hold_days = int(self.config['swing_min_hold_days'])
        swing_min_profit_pct = float(self.config['swing_min_profit_pct'])
        swing_max_profit_pct = float(self.config['swing_max_profit_pct'])
        swing_aroon_threshold = float(self.config['swing_aroon_threshold'])
        swing_bb_sell_threshold = float(self.config['swing_bb_sell_threshold'])
        swing_rsi_sell_threshold = float(self.config['swing_rsi_sell_threshold'])
        swing_volume_surge_block = float(self.config['swing_volume_surge_block'])
        swing_bb_rebuy_threshold = float(self.config['swing_bb_rebuy_threshold'])
        swing_rsi_rebuy_threshold = float(self.config['swing_rsi_rebuy_threshold'])
        swing_stoch_k_rebuy_threshold = float(self.config['swing_stoch_k_rebuy_threshold'])
        swing_breakout_chase_pct = float(self.config['swing_breakout_chase_pct'])
        swing_next_day_up_rebuy = bool(self.config['swing_next_day_up_rebuy'])
        swing_max_wait_days = int(self.config['swing_max_wait_days'])
        swing_max_loss_from_sell_pct = float(self.config['swing_max_loss_from_sell_pct'])
        swing_trend_reversal_giveup = bool(self.config['swing_trend_reversal_giveup'])
        swing_breakout_max_gap_pct = float(self.config['swing_breakout_max_gap_pct'])
        swing_breakout_min_wait_days = int(self.config['swing_breakout_min_wait_days'])
        swing_volume_breakout_rebuy = bool(self.config['swing_volume_breakout_rebuy'])
        swing_volume_breakout_ratio = float(self.config['swing_volume_breakout_ratio'])
        # 高抛低吸运行时状态
        swing_state = 0  # 0=无, 1=等待回买
        swing_sell_price = 0.0
        swing_sell_idx = 0
        swing_original_entry_price = 0.0
        swing_lowest_price = 0.0  # 等待期间的最低价（用于智能回买）
        # swing做T状态保存（回买时恢复，与EH做T保持一致）
        swing_saved_entry_price = 0.0
        swing_saved_hold_days = 0
        swing_saved_pending_exit = False
        swing_saved_trailing_stop_active = False
        swing_saved_ts_pending = False
        swing_saved_ts_pending_days = 0
        swing_saved_dynamic_profit_active = False
        swing_saved_max_profit_in_trade = 0.0
        swing_saved_is_divergence_entry = False
        swing_saved_is_w_bottom_entry = False
        swing_saved_is_sideways_entry = False
        swing_exit_flags = np.zeros(n, dtype=int)  # 1=swing sell, 2=swing rebuy, 3=giveup
        swing_rebuy_reasons = [''] * n  # 记录回买原因
        swing_giveup_blocking = False  # 放弃后屏蔽买入，直到影子仓位退出
        # 影子仓位（giveup后概念上仍持有1股，走同样的退出逻辑）
        shadow_position_active = False
        shadow_entry_price = 0.0
        shadow_pending_exit = False
        shadow_pending_exit_price = 0.0
        shadow_pending_exit_days = 0

        for i in range(n):
            entry_active = bool(entry_condition.iloc[i]) if not pd.isna(entry_condition.iloc[i]) else False
            exit_active = bool(exit_condition.iloc[i]) if not pd.isna(exit_condition.iloc[i]) else False
            is_div_entry = bool(divergence_entry.iloc[i]) if not pd.isna(divergence_entry.iloc[i]) else False
            is_w_entry = bool(w_bottom_entry.iloc[i]) if not pd.isna(w_bottom_entry.iloc[i]) else False
            is_sw_entry = bool(sideways_entry.iloc[i]) if not pd.isna(sideways_entry.iloc[i]) else False
            curr_price = price_series.iloc[i] if i < len(price_series) else np.nan

            # 高抛放弃后的影子仓位处理：模拟原本仓位的退出逻辑
            if shadow_position_active and swing_giveup_blocking:
                shadow_should_exit = False

                # 检查止损
                if stop_loss_pct > 0 and shadow_entry_price > 0 and not pd.isna(curr_price):
                    threshold = shadow_entry_price * (1 - stop_loss_pct / 100.0)
                    if curr_price <= threshold:
                        shadow_should_exit = True

                # 处理待反弹卖出状态
                if not shadow_should_exit and shadow_pending_exit:
                    shadow_pending_exit_days += 1
                    prev_close = data['close'].iloc[i - 1] if data is not None and i > 0 else curr_price
                    day_change = (curr_price / prev_close - 1) * 100 if prev_close > 0 else 0
                    bounce_from_signal = (curr_price / shadow_pending_exit_price - 1) * 100 if shadow_pending_exit_price > 0 else 0

                    bounce_ok = (day_change > 0)
                    if bounce_exit_bounce_pct > 0:
                        bounce_ok = bounce_ok or (bounce_from_signal >= -bounce_exit_bounce_pct)
                    timeout = (shadow_pending_exit_days >= bounce_exit_max_wait)

                    if bounce_ok or timeout:
                        shadow_should_exit = True

                # 检查正常退出条件
                elif not shadow_should_exit and exit_active:
                    if bounce_exit_enabled and data is not None and i > 0:
                        prev_close = data['close'].iloc[i - 1]
                        day_change = (curr_price / prev_close - 1) * 100 if prev_close > 0 else 0

                        if day_change < bounce_exit_drop_threshold:
                            # 暴跌中，进入待卖出状态
                            shadow_pending_exit = True
                            shadow_pending_exit_price = curr_price
                            shadow_pending_exit_days = 0
                        else:
                            # 非暴跌，正常退出
                            shadow_should_exit = True
                    else:
                        # bounce_exit未启用，直接退出
                        shadow_should_exit = True

                # 影子仓位退出
                if shadow_should_exit:
                    shadow_position_active = False
                    shadow_entry_price = 0.0
                    shadow_pending_exit = False
                    shadow_pending_exit_price = 0.0
                    shadow_pending_exit_days = 0
                    # 重要：退出当天仍然阻止入场（与非swing场景一致）
                    # 下一个交易日才会解除阻止（swing_giveup_blocking在shadow不活跃时自动解除）
                    position[i] = 0
                    continue

            # 影子仓位已退出但blocking仍生效：解除屏蔽（从下一天开始允许入场）
            if swing_giveup_blocking and not shadow_position_active:
                swing_giveup_blocking = False

            # 追高冷却期逻辑
            avoid_extreme_chase = False
            chase_pullback_buy = False
            if chase_mode != 'none' and data is not None and i < len(data):
                row = data.iloc[i]
                ma_120 = row.get('ma_120', np.nan)
                short_gain_10d = row.get('short_gain_10d', np.nan)

                price_vs_ma120 = ((curr_price / ma_120 - 1) * 100) if not np.isnan(ma_120) and not np.isnan(curr_price) and ma_120 > 0 else 0

                is_chase_condition = (price_vs_ma120 > 15) and (not np.isnan(short_gain_10d) and short_gain_10d > 15)

                if is_chase_condition and not chase_cooldown_active and not in_position:
                    chase_cooldown_active = True
                    chase_peak_price = curr_price if not np.isnan(curr_price) else 0
                    chase_start_idx = i
                    # 记录追高时的成交量倍数
                    vol = row.get('volume', np.nan)
                    vol_ma20 = row.get('volume_ma20', np.nan)
                    if not np.isnan(vol) and not np.isnan(vol_ma20) and vol_ma20 > 0:
                        chase_rise_vol_ratio = vol / vol_ma20
                    else:
                        chase_rise_vol_ratio = 1.0
                    # 硬屏蔽判定：巨量+10日暴涨 → 完全不允许回调买入
                    chase_hard_block = (
                        chase_hard_block_vol > 0 and chase_hard_block_gain > 0
                        and chase_rise_vol_ratio > chase_hard_block_vol
                        and not np.isnan(short_gain_10d) and short_gain_10d > chase_hard_block_gain
                    )
                    avoid_extreme_chase = True
                elif chase_cooldown_active and not in_position:
                    if chase_mode == 'block':
                        # 纯阻断模式：条件期间一直阻断
                        if is_chase_condition:
                            avoid_extreme_chase = True
                        else:
                            chase_cooldown_active = False
                    else:
                        # cooldown模式：追踪高点，满足任一profile条件后允许回调买入
                        if not np.isnan(curr_price) and curr_price > chase_peak_price:
                            chase_peak_price = curr_price

                        days_in_cooldown = i - chase_start_idx
                        if days_in_cooldown > chase_cooldown_days:
                            chase_cooldown_active = False
                        elif chase_peak_price > 0 and not np.isnan(curr_price):
                            drop_from_peak = (1 - curr_price / chase_peak_price) * 100
                            chase_cleared = not is_chase_condition
                            # 计算回调时成交量比
                            vol = row.get('volume', np.nan)
                            vol_ma20 = row.get('volume_ma20', np.nan)
                            curr_vol_ratio = (vol / vol_ma20) if not np.isnan(vol) and not np.isnan(vol_ma20) and vol_ma20 > 0 else 1.0
                            drop_speed = (drop_from_peak / days_in_cooldown) if days_in_cooldown > 0 else 0

                            # 急跌跌幅上限：回调超过X%视为崩盘，不买入
                            if chase_max_drop_pct > 0 and drop_from_peak > chase_max_drop_pct:
                                avoid_extreme_chase = True
                            # 硬屏蔽：完全不允许回调买入，只能等冷却期结束
                            elif chase_hard_block:
                                avoid_extreme_chase = True
                            else:
                                # OR逻辑：任一profile满足即允许买入
                                any_profile_ok = False
                                for profile in chase_profiles:
                                    p_pb = profile.get('pullback_pct', 5)
                                    p_clr = profile.get('require_cleared', True)
                                    p_vol = profile.get('vol_min_ratio', 0.8)
                                    p_rvm = profile.get('rise_vol_max', 0)
                                    p_mds = profile.get('min_drop_speed', 0)

                                    if drop_from_peak < p_pb:
                                        continue
                                    if p_clr and not chase_cleared:
                                        continue
                                    if p_vol > 0 and curr_vol_ratio < p_vol:
                                        continue
                                    if p_rvm > 0 and chase_rise_vol_ratio > p_rvm:
                                        continue
                                    if p_mds > 0 and drop_speed < p_mds:
                                        continue
                                    any_profile_ok = True
                                    break

                                if any_profile_ok and entry_active:
                                    chase_pullback_buy = True
                                    chase_cooldown_active = False
                                else:
                                    avoid_extreme_chase = True

            # EH做T：等待回买状态处理（EH期间卖出后等回补）
            if _eh_swing_active and not in_position and data is not None:
                _ehs_rebuy = False
                _ehs_giveup = False
                _ehs_wait_days = i - _eh_swing_sell_idx
                _ehs_bb = data['bb_percent'].iloc[i] if 'bb_percent' in data.columns and not pd.isna(data['bb_percent'].iloc[i]) else np.nan
                _ehs_rsi = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns and not pd.isna(data['fast_rsi'].iloc[i]) else np.nan
                _ehs_stoch_k = data['stoch_k'].iloc[i] if 'stoch_k' in data.columns and not pd.isna(data['stoch_k'].iloc[i]) else np.nan
                _ehs_ma120 = data['ma_120'].iloc[i] if 'ma_120' in data.columns else np.nan

                # 追踪T卖后的最高价（用于回调低吸判断）
                if not np.isnan(curr_price) and curr_price > _eh_swing_peak_after_sell:
                    _eh_swing_peak_after_sell = curr_price

                # 回买路径1：超卖低吸（传统做T：RSI超卖 + 价格在允许范围内）
                _ehs_max_rebuy_price = _eh_swing_sell_price * (1 + eh_swing_rebuy_max_above / 100) if eh_swing_rebuy_max_above > 0 else _eh_swing_sell_price
                _ehs_price_ok = not np.isnan(curr_price) and _eh_swing_sell_price > 0 and curr_price <= _ehs_max_rebuy_price
                # 最小等待天数和最小跌幅前置条件
                _ehs_wait_ok = _ehs_wait_days >= eh_swing_rebuy_min_wait
                _ehs_drop_pct = (curr_price / _eh_swing_sell_price - 1) * 100 if _eh_swing_sell_price > 0 and not np.isnan(curr_price) else 0
                _ehs_drop_ok = eh_swing_rebuy_min_drop <= 0 or _ehs_drop_pct <= -eh_swing_rebuy_min_drop
                if _ehs_price_ok and _ehs_wait_ok and _ehs_drop_ok:
                    if (not np.isnan(_ehs_rsi) and _ehs_rsi < eh_swing_rebuy_rsi):
                        _ehs_rebuy = True
                    elif (not np.isnan(_ehs_bb) and _ehs_bb < eh_swing_rebuy_bb):
                        _ehs_rebuy = True
                    elif (not np.isnan(_ehs_stoch_k) and _ehs_stoch_k < swing_stoch_k_rebuy_threshold):
                        _ehs_rebuy = True
                    elif (eh_swing_rebuy_stk > 0 and not np.isnan(_ehs_stoch_k) and _ehs_stoch_k < eh_swing_rebuy_stk):
                        _ehs_rebuy = True

                # 回买路径2：回调接回（T飞后找机会接回：股价从T卖后高点回撤X%）
                if not _ehs_rebuy and eh_swing_rebuy_pullback_pct > 0 and _eh_swing_peak_after_sell > 0:
                    if not np.isnan(curr_price) and curr_price > _eh_swing_sell_price:
                        # 只在股价高于卖价时触发（真正卖飞的情况）
                        _ehs_pullback_threshold = _eh_swing_peak_after_sell * (1 - eh_swing_rebuy_pullback_pct / 100)
                        if curr_price <= _ehs_pullback_threshold:
                            _ehs_rebuy = True  # 回调接回

                # 回买路径3：强制回买（兜底：股价大幅高于卖价 → 无条件接回）
                if not _ehs_rebuy and eh_swing_force_rebuy_premium > 0 and not np.isnan(curr_price):
                    _ehs_force_price = _eh_swing_sell_price * (1 + eh_swing_force_rebuy_premium / 100)
                    if curr_price > _ehs_force_price:
                        _ehs_rebuy = True

                # 放弃条件
                if not _ehs_rebuy:
                    # 底线被击穿 或 MA120跌破 → 彻底退出
                    if not np.isnan(curr_price) and _eh_swing_floor_price > 0 and curr_price < _eh_swing_floor_price:
                        _ehs_giveup = True
                    if not np.isnan(_ehs_ma120) and _ehs_ma120 > 0 and not np.isnan(curr_price) and curr_price < _ehs_ma120:
                        _ehs_giveup = True
                    # 等待超时 → 放弃做T，让普通策略自然接管
                    if not _ehs_giveup and eh_swing_max_wait_days > 0 and _ehs_wait_days >= eh_swing_max_wait_days:
                        _ehs_giveup = True

                if _ehs_rebuy:
                    # EH做T回买成功 — 恢复原始交易状态（做T不影响原始买卖逻辑）
                    in_position = True
                    entry_flags[i] = 1
                    entry_price = _eh_swing_saved_entry_price  # 恢复原始买入价（止损等基于原始价格）
                    swing_exit_flags[i] = 2
                    _ehs_is_above_sell = curr_price > _eh_swing_sell_price if _eh_swing_sell_price > 0 else False
                    if _ehs_is_above_sell:
                        swing_rebuy_reasons[i] = 'EH做T-回调接回'
                        entry_reasons[i] = 'EH做T-回调接回'
                    else:
                        swing_rebuy_reasons[i] = 'EH做T-低吸'
                        entry_reasons[i] = 'EH做T-低吸'
                    # 恢复所有原始交易状态变量
                    is_divergence_entry = _eh_swing_saved_is_divergence_entry
                    is_w_bottom_entry = _eh_swing_saved_is_w_bottom_entry
                    is_sideways_entry = _eh_swing_saved_is_sideways_entry
                    hold_days = _eh_swing_saved_hold_days + (i - _eh_swing_sell_idx)  # 持仓天数连续计算
                    pending_exit = _eh_swing_saved_pending_exit
                    trailing_stop_active = _eh_swing_saved_trailing_stop_active
                    _ts_pending = _eh_swing_saved_ts_pending
                    _ts_pending_days = _eh_swing_saved_ts_pending_days
                    dynamic_profit_active = _eh_swing_saved_dynamic_profit_active
                    max_profit_in_trade = _eh_swing_saved_max_profit_in_trade
                    _eh_swing_active = False
                    _eh_swing_used = False  # 允许后续继续做T（冷却期控制防连续触发）
                    _eh_swing_rebuy_idx = i  # 记录回买位置用于冷却期计算
                    # 恢复EH状态（extended_hold_active仍为True）
                    extended_hold_trigger_profit = _eh_swing_saved_eh_trigger_profit
                    extended_hold_max_profit = _eh_swing_saved_eh_max_profit
                    position[i] = 1
                    continue
                elif _ehs_giveup:
                    # EH做T放弃，完全退出
                    _eh_swing_active = False
                    extended_hold_active = False
                    extended_hold_trigger_profit = 0.0
                    extended_hold_max_profit = 0.0
                    _eh_swing_sell_price = 0.0
                    _eh_swing_original_entry = 0.0
                    _eh_swing_floor_price = 0.0
                    _eh_swing_sell_idx = 0
                    position[i] = 0
                    continue
                else:
                    # 继续等待回买
                    position[i] = 0
                    continue

            # 高抛低吸：等待回买状态处理
            if swing_state == 1 and not in_position and data is not None:
                sw_days_waiting = i - swing_sell_idx
                sw_bb_pct = data['bb_percent'].iloc[i] if 'bb_percent' in data.columns and not pd.isna(data['bb_percent'].iloc[i]) else np.nan
                sw_rsi = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns and not pd.isna(data['fast_rsi'].iloc[i]) else np.nan
                sw_stoch_k = data['stoch_k'].iloc[i] if 'stoch_k' in data.columns and not pd.isna(data['stoch_k'].iloc[i]) else np.nan
                sw_macd_hist = data['macd_hist'].iloc[i] if 'macd_hist' in data.columns and not pd.isna(data['macd_hist'].iloc[i]) else np.nan
                sw_macd_hist_prev = data['macd_hist'].iloc[i-1] if i > 0 and 'macd_hist' in data.columns and not pd.isna(data['macd_hist'].iloc[i-1]) else np.nan
                sw_trend = data['trend_direction'].iloc[i] if 'trend_direction' in data.columns and not pd.isna(data['trend_direction'].iloc[i]) else 0

                # 更新等待期间的最低价
                if not np.isnan(curr_price):
                    if swing_lowest_price == 0 or curr_price < swing_lowest_price:
                        swing_lowest_price = curr_price

                sw_giveup = False
                sw_rebuy = False

                # 计算当前价格相对卖出价的变化
                price_vs_sell = (curr_price / swing_sell_price - 1) * 100 if not np.isnan(curr_price) and swing_sell_price > 0 else 0

                # ===== 低吸条件：价格 <= 卖出价 且 趋势未反转 =====
                # 关键修复：低吸买回必须检查趋势，避免在趋势反转后买入
                # （100%的低吸失败都是因为entry_condition=False时强制买入）
                can_low_rebuy = (price_vs_sell <= 0
                                 and sw_trend == 1  # 趋势仍向上
                                 and not exit_active)  # 没有退出信号
                sw_rebuy_reason = ''  # 记录回买原因
                if can_low_rebuy:
                    # 【条件1】RSI超卖（分析显示RSI<30胜率73.7%，是最佳回买指标）
                    if not sw_rebuy and not np.isnan(sw_rsi) and sw_rsi < swing_rsi_rebuy_threshold:
                        sw_rebuy = True
                        sw_rebuy_reason = '持仓做T-低吸'

                    # 【条件2】BB回到中低位
                    if not sw_rebuy and not np.isnan(sw_bb_pct) and sw_bb_pct < swing_bb_rebuy_threshold:
                        sw_rebuy = True
                        sw_rebuy_reason = '持仓做T-低吸'

                    # 【条件3】KDJ K线超卖
                    if not sw_rebuy and not np.isnan(sw_stoch_k) and sw_stoch_k < swing_stoch_k_rebuy_threshold:
                        sw_rebuy = True
                        sw_rebuy_reason = '持仓做T-低吸'

                    # 【条件4】MACD金叉（柱状图由负转正）
                    if not sw_rebuy and not np.isnan(sw_macd_hist) and not np.isnan(sw_macd_hist_prev):
                        if sw_macd_hist_prev < 0 and sw_macd_hist > 0:
                            sw_rebuy = True
                            sw_rebuy_reason = '持仓做T-低吸'

                    # 【条件5】极端下跌触发回买，博反弹
                    swing_rebuy_drop_pct = float(self.config['swing_rebuy_drop_pct'])
                    if not sw_rebuy and not np.isnan(curr_price) and swing_sell_price > 0:
                        drop_pct = (1 - curr_price / swing_sell_price) * 100
                        if drop_pct >= swing_rebuy_drop_pct:
                            sw_rebuy = True
                            sw_rebuy_reason = f'持仓做T-低吸(跌{drop_pct:.1f}%博反弹)'

                    # 【条件6】智能回买：价格已从最低点反弹2%以上，且当前仍低于卖出价
                    if not sw_rebuy and swing_lowest_price > 0 and not np.isnan(curr_price) and swing_sell_price > 0:
                        bounce_from_low = (curr_price / swing_lowest_price - 1) * 100
                        if bounce_from_low >= 2.0 and sw_days_waiting >= 2:
                            sw_rebuy = True
                            sw_rebuy_reason = '持仓做T-低吸'

                # ===== 放量突破买回：第二天涨 + 放量 = 真突破，立即买回 =====
                # 分析显示：放量(量比>1.8)的次日涨是真突破，应该买回
                # 而普通的次日涨82%会跌回来，不应该追
                if not sw_rebuy and swing_volume_breakout_rebuy and sw_days_waiting == 1:
                    if not np.isnan(curr_price) and swing_sell_price > 0 and price_vs_sell > 0:
                        # 检查今天是否放量
                        sw_vol = data['volume'].iloc[i] if 'volume' in data.columns else np.nan
                        sw_vol_ma = data['volume_ma20'].iloc[i] if 'volume_ma20' in data.columns else np.nan
                        if not np.isnan(sw_vol) and not np.isnan(sw_vol_ma) and sw_vol_ma > 0:
                            vol_ratio = sw_vol / sw_vol_ma
                            if vol_ratio >= swing_volume_breakout_ratio:
                                sw_rebuy = True  # 放量上涨 = 真突破
                                sw_rebuy_reason = '持仓做T-低吸'

                # ===== 普通追高条件（默认禁用） =====
                if not sw_rebuy and not np.isnan(curr_price) and swing_sell_price > 0:
                    if (price_vs_sell > swing_breakout_chase_pct
                        and price_vs_sell <= swing_breakout_max_gap_pct
                        and sw_days_waiting >= swing_breakout_min_wait_days):
                        sw_rebuy = True
                        sw_rebuy_reason = '持仓做T-低吸'

                # 只有回买条件不满足时，才检查放弃条件
                if not sw_rebuy:
                    if sw_days_waiting >= swing_max_wait_days:
                        sw_giveup = True
                    if not np.isnan(curr_price) and swing_original_entry_price > 0 and curr_price < swing_original_entry_price:
                        sw_giveup = True
                    if swing_trend_reversal_giveup and sw_trend == -1:
                        sw_giveup = True
                    if not np.isnan(curr_price) and swing_sell_price > 0:
                        drop_from_sell = (1 - curr_price / swing_sell_price) * 100
                        if drop_from_sell > swing_max_loss_from_sell_pct:
                            sw_giveup = True

                if sw_giveup:
                    swing_exit_flags[i] = 3
                    swing_state = 0
                    swing_sell_price = 0.0
                    swing_sell_idx = 0
                    swing_lowest_price = 0.0
                    # 放弃后启用影子仓位，模拟原本持仓的退出逻辑
                    swing_giveup_blocking = True
                    shadow_position_active = True
                    shadow_entry_price = swing_original_entry_price  # 保存原始入场价用于止损计算
                    shadow_pending_exit = False
                    shadow_pending_exit_price = 0.0
                    shadow_pending_exit_days = 0
                    swing_original_entry_price = 0.0
                    position[i] = 0
                    continue  # 不允许立即入场
                elif sw_rebuy:
                    # Swing做T回买 — 恢复原始交易状态（与EH做T保持一致）
                    in_position = True
                    entry_flags[i] = 1
                    entry_price = swing_saved_entry_price  # 恢复原始买入价
                    swing_exit_flags[i] = 2
                    swing_rebuy_reasons[i] = sw_rebuy_reason  # 记录回买原因
                    entry_reasons[i] = f'持仓做T-{sw_rebuy_reason}'
                    is_divergence_entry = swing_saved_is_divergence_entry
                    is_w_bottom_entry = swing_saved_is_w_bottom_entry
                    is_sideways_entry = swing_saved_is_sideways_entry
                    hold_days = swing_saved_hold_days + (i - swing_sell_idx)  # 持仓天数连续计算
                    pending_exit = swing_saved_pending_exit
                    pending_exit_days = 0
                    trailing_stop_active = swing_saved_trailing_stop_active
                    _ts_pending = swing_saved_ts_pending
                    _ts_pending_days = swing_saved_ts_pending_days
                    dynamic_profit_active = swing_saved_dynamic_profit_active
                    max_profit_in_trade = swing_saved_max_profit_in_trade
                    swing_state = 0
                    swing_sell_price = 0.0
                    swing_sell_idx = 0
                    swing_original_entry_price = 0.0
                    swing_lowest_price = 0.0
                    if data is not None and 'chase_pullback_entry' in data.columns:
                        data.iloc[i, data.columns.get_loc('chase_pullback_entry')] = True
                    position[i] = 1
                    continue
                else:
                    # 继续等待，不进入正常入场逻辑
                    position[i] = 0
                    continue

            # 高抛放弃后屏蔽买入（概念上还持有1股，等原本的卖出信号）
            if swing_giveup_blocking:
                position[i] = 0
                continue

            # 主升浪再入场：EH止盈退出后，股价突破退场价确认新高时回补
            if not in_position and post_wave_reentry_countdown > 0:
                post_wave_reentry_countdown -= 1
                if not entry_active and _pw_exit_price > 0 and not np.isnan(curr_price):
                    _pw_confirm_price = _pw_exit_price * (1 + pw_price_confirm_pct / 100)
                    if curr_price > _pw_confirm_price:
                        # 检查MA120仍在上升
                        _pw_ma120_still_rising = False
                        if data is not None and 'ma_120' in data.columns and i >= 40:
                            _pw_m = data['ma_120'].iloc[i]
                            _pw_ma120_still_rising = not np.isnan(_pw_m) and _pw_m > data['ma_120'].iloc[i - 40]
                        if _pw_ma120_still_rising:
                            entry_active = True
                            avoid_extreme_chase = False
                            post_wave_reentry_countdown = 0  # 回补后停止（后续由新的退出重新激活）

            # 入场前过滤检查（亏损冷却、成交量确认、MA对齐）
            _entry_filters_ok = True
            if not in_position and ((entry_active and not avoid_extreme_chase) or chase_pullback_buy):
                # 亏损冷却期检查
                if loss_cooldown_days > 0 and (i - _last_loss_exit_idx) <= loss_cooldown_days:
                    _entry_filters_ok = False
                # 入场成交量确认
                if _entry_filters_ok and entry_vol_confirm_mult > 0 and data is not None and 'volume' in data.columns and 'volume_ma20' in data.columns:
                    _ev = data['volume'].iloc[i]
                    _ev_ma = data['volume_ma20'].iloc[i]
                    if not np.isnan(_ev) and not np.isnan(_ev_ma) and _ev_ma > 0:
                        if _ev < entry_vol_confirm_mult * _ev_ma:
                            _entry_filters_ok = False
                # MA趋势对齐
                if _entry_filters_ok and entry_ma_align_enabled and data is not None:
                    _ma_s_col = f'ma_{entry_ma_align_short}'
                    _ma_l_col = f'ma_{entry_ma_align_long}'
                    if _ma_s_col in data.columns and _ma_l_col in data.columns:
                        _ma_s_val = data[_ma_s_col].iloc[i]
                        _ma_l_val = data[_ma_l_col].iloc[i]
                        if not np.isnan(_ma_s_val) and not np.isnan(_ma_l_val):
                            if _ma_s_val < _ma_l_val:
                                _entry_filters_ok = False
                # 大盘regime过滤：大盘趋势下行时阻止入场
                if _entry_filters_ok and market_regime_enabled and _regime_signal is not None and len(_regime_signal) > 0:
                    _date_str = str(data['date'].iloc[i])[:10]
                    if _date_str in _regime_signal.index:
                        if not _regime_signal[_date_str]:
                            _entry_filters_ok = False

            if _entry_filters_ok and not in_position and ((entry_active and not avoid_extreme_chase) or (chase_pullback_buy and not in_position)):
                in_position = True
                entry_flags[i] = 1
                entry_price = curr_price if not np.isnan(curr_price) else None
                is_divergence_entry = is_div_entry  # 记录是否为底背离买入
                is_w_bottom_entry = is_w_entry  # 记录是否为W底买入
                is_sideways_entry = is_sw_entry  # 记录是否为震荡市场买入
                # 记录真实入场原因
                if chase_pullback_buy:
                    entry_reasons[i] = '追高回调买入'
                elif is_sw_entry:
                    entry_reasons[i] = 'Aroon震荡入场'
                elif is_div_entry:
                    entry_reasons[i] = '底背离信号'
                elif is_w_entry:
                    entry_reasons[i] = 'W底形态'
                elif data is not None and 'bb_squeeze_entry' in data.columns and bool(data['bb_squeeze_entry'].iloc[i]):
                    entry_reasons[i] = 'BB Squeeze突破'
                elif data is not None and 'stoch_reversal_entry' in data.columns and bool(data['stoch_reversal_entry'].iloc[i]):
                    entry_reasons[i] = 'Stoch超卖反转'
                elif data is not None and 'rsi_momentum_entry' in data.columns and bool(data['rsi_momentum_entry'].iloc[i]):
                    entry_reasons[i] = 'RSI动量加速'
                elif data is not None and 'discount_zone_entry' in data.columns and bool(data['discount_zone_entry'].iloc[i]):
                    entry_reasons[i] = '折价区补仓'
                elif data is not None and 'dual_channel_signal' in data.columns and bool(data['dual_channel_signal'].iloc[i]):
                    entry_reasons[i] = '双通道信号'
                elif data is not None and 'gap_fade_signal' in data.columns and bool(data.get('gap_fade_signal', pd.Series(False)).iloc[i]):
                    entry_reasons[i] = '跳空回补'
                else:
                    # 标准RSI入场 - 区分金叉和多头延续
                    if data is not None and 'golden_cross' in data.columns and bool(data['golden_cross'].iloc[i]):
                        entry_reasons[i] = 'RSI金叉'
                    elif data is not None and 'rsi_relaxed_condition' in data.columns and bool(data['rsi_relaxed_condition'].iloc[i]):
                        entry_reasons[i] = 'RSI多头延续'
                    else:
                        entry_reasons[i] = 'RSI趋势买入'
                # 自适应止损：根据入场时大盘regime决定止损幅度
                _trade_stop_loss = stop_loss_pct  # 默认使用正常止损
                if _adaptive_sl_enabled and _regime_signal is not None and len(_regime_signal) > 0:
                    _date_str = str(data['date'].iloc[i])[:10]
                    if _date_str in _regime_signal.index and not _regime_signal[_date_str]:
                        _trade_stop_loss = _adaptive_sl_bear_pct  # 熊市用更紧止损
                # 标记回调买入
                if chase_pullback_buy and data is not None and 'chase_pullback_entry' in data.columns:
                    data.iloc[i, data.columns.get_loc('chase_pullback_entry')] = True
                hold_days = 0  # 重置持仓天数
                pending_exit = False  # 重置反弹卖出状态
                pending_exit_days = 0
                trailing_stop_active = False  # 重置止盈保护状态
                _ts_pending = False
                _ts_pending_days = 0
                dynamic_profit_active = False
                max_profit_in_trade = 0
                extended_hold_active = False  # 重置延长持仓
                extended_hold_trigger_profit = 0.0
                extended_hold_max_profit = 0.0
                _eh_swing_used = False  # 重置做T标记
                _eh_swing_confirming = False  # 重置确认状态
                _eh_swing_armed = False  # 重置武装模式
                _eh_overbought_seen = False  # 重置超买标记
                # post_wave_reentry_countdown 不重置：允许跨多笔交易持续生效

                # 调试W底买入
                if is_w_entry and data is not None and 'date' in data.columns:
                    buy_date = data['date'].iloc[i]
                    logger.info(f"[W底买入执行] {buy_date} 触发W底买入，价格={entry_price:.2f}")
                
                # 如果是W底买入，记录W底价格和间隔天数
                if is_w_entry and 'w_bottom_price' in data.columns:
                    w_bottom_price = data['w_bottom_price'].iloc[i] if not pd.isna(data['w_bottom_price'].iloc[i]) else None
                    w_bottom_gap = data['w_bottom_gap'].iloc[i] if 'w_bottom_gap' in data.columns and not pd.isna(data['w_bottom_gap'].iloc[i]) else None
                else:
                    w_bottom_price = None
                    w_bottom_gap = None
                
                # 记录买入时的RSI值（用于底背离买入的趋势判断）
                if is_div_entry and rsi_fast is not None and i < len(rsi_fast):
                    entry_rsi = rsi_fast.iloc[i] if not pd.isna(rsi_fast.iloc[i]) else None
                else:
                    entry_rsi = None

            # 刚退出后评估是否激活主升浪再入场窗口（检查前一天是否退出）
            if not in_position and i > 0 and exit_flags[i - 1] == 1 and post_wave_reentry_countdown == 0:
                if _pw_last_trade_profit > pw_profit_threshold and _pw_last_trade_hold >= pw_min_hold:
                    # 检查MA120是否上升
                    _pw_ma120_ok = False
                    if data is not None and 'ma_120' in data.columns and i >= 40:
                        _pw_ma120_v = data['ma_120'].iloc[i]
                        _pw_ma120_ok = not np.isnan(_pw_ma120_v) and _pw_ma120_v > data['ma_120'].iloc[i - 40]
                    if _pw_ma120_ok:
                        post_wave_reentry_countdown = post_wave_reentry_window
                        # 记录退场价（用于价格突破确认回补）
                        if _pw_exit_price <= 0 and data is not None:
                            _pw_exit_price = data['close'].iloc[i - 1]

            if in_position:
                hold_days += 1
                _pw_last_trade_profit = 0.0  # 初始值，每天更新
                _pw_last_trade_hold = hold_days

                # 更新当前交易最大浮盈
                if entry_price and not pd.isna(curr_price) and entry_price > 0:
                    curr_profit_pct = (curr_price / entry_price - 1) * 100
                    _pw_last_trade_profit = curr_profit_pct  # 追踪实时利润
                    if curr_profit_pct > max_profit_in_trade:
                        max_profit_in_trade = curr_profit_pct
                        _days_since_peak = 0  # 创新高，重置滞涨计数
                    else:
                        _days_since_peak += 1  # 未创新高，累计天数

                    # Hard Loss Cap — 硬性最大亏损上限（所有入场类型生效）
                    if hard_loss_cap_enabled and curr_profit_pct <= -hard_loss_cap_pct:
                        in_position = False
                        exit_flags[i] = 1
                        stop_flags[i] = 1
                        exit_reasons[i] = f'硬性止损上限({hard_loss_cap_pct:.0f}%)'
                        entry_price = None
                        is_divergence_entry = False
                        is_w_bottom_entry = False
                        is_sideways_entry = False
                        w_bottom_price = None
                        w_bottom_gap = None
                        hold_days = 0
                        trailing_stop_active = False
                        dynamic_profit_active = False
                        max_profit_in_trade = 0
                        pending_exit = False
                        position[i] = 0
                        continue

                    # 延长持仓每日安全检查：价格跌破MA120 或 利润回撤超限 或 从峰值回撤过多 → 退出
                    # 注意：对所有入场类型生效（包括底背离/W底/震荡）
                    if extended_hold_active:
                        if curr_profit_pct > extended_hold_max_profit:
                            extended_hold_max_profit = curr_profit_pct
                        _eh_ma120 = data['ma_120'].iloc[i] if data is not None and 'ma_120' in data.columns else np.nan
                        _eh_profit_floor = extended_hold_trigger_profit - eh_drawdown_limit
                        # 比例保护底线：保护已有增益的ratio%（随浮盈自动上升）
                        if eh_gain_protection_ratio > 0 and extended_hold_max_profit > extended_hold_trigger_profit:
                            _eh_gain = extended_hold_max_profit - extended_hold_trigger_profit
                            _eh_proportional_floor = extended_hold_trigger_profit + _eh_gain * eh_gain_protection_ratio
                            _eh_profit_floor = max(_eh_profit_floor, _eh_proportional_floor)
                        # 超买收紧检测（粘性标记：一旦超买，本EH周期内持续收紧trailing）
                        if eh_overbought_trailing > 0 and not _eh_overbought_seen and data is not None:
                            _eht_rsi = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns and not pd.isna(data['fast_rsi'].iloc[i]) else np.nan
                            _eht_bb = data['bb_percent'].iloc[i] if 'bb_percent' in data.columns and not pd.isna(data['bb_percent'].iloc[i]) else np.nan
                            if (not np.isnan(_eht_rsi) and _eht_rsi >= eh_swing_rsi_threshold
                                    and not np.isnan(_eht_bb) and _eht_bb >= eh_swing_bb_threshold):
                                _eh_overbought_seen = True

                        # 峰值回撤止损：浮盈超过触发浮盈+offset后激活（相对阈值）
                        _eh_peak_act_level = extended_hold_trigger_profit + eh_peak_activation_offset
                        _eh_trailing_val = eh_peak_trailing  # 默认20%

                        # 超买自适应：检测到超买后立即激活peak trailing并收紧
                        if _eh_overbought_seen and eh_overbought_trailing > 0:
                            _eh_peak_act_level = extended_hold_trigger_profit  # 立即激活（offset=0）
                            _eh_trailing_val = eh_overbought_trailing  # 收紧trailing

                        if extended_hold_max_profit > _eh_peak_act_level:
                            _eh_peak_floor = extended_hold_max_profit - _eh_trailing_val
                            _eh_effective_floor = max(_eh_profit_floor, _eh_peak_floor)
                        else:
                            _eh_effective_floor = _eh_profit_floor
                        if (not np.isnan(_eh_ma120) and _eh_ma120 > 0 and curr_price < _eh_ma120) or curr_profit_pct < _eh_effective_floor:
                            _pw_exit_price = curr_price  # 记录EH退出价格用于回补确认
                            extended_hold_active = False
                            in_position = False
                            exit_flags[i] = 1
                            if not np.isnan(_eh_ma120) and _eh_ma120 > 0 and curr_price < _eh_ma120:
                                exit_reasons[i] = '延长持仓-跌破MA120'
                            else:
                                exit_reasons[i] = f'延长持仓-利润回撤(floor={_eh_effective_floor:.1f}%)'
                            entry_price = None
                            hold_days = 0
                            trailing_stop_active = False
                            dynamic_profit_active = False
                            max_profit_in_trade = 0
                            pending_exit = False
                            position[i] = 0
                            continue

                        # EH做T高抛：在EH期间超买时卖出，等待回调再接回（支持多次做T，冷却期控制）
                        _ehs_cooldown_ok = (i - _eh_swing_rebuy_idx >= eh_swing_cooldown_days) if _eh_swing_rebuy_idx > 0 else True
                        if (eh_swing_enabled and not _eh_swing_active and not _eh_swing_used
                                and _ehs_cooldown_ok and swing_state == 0 and data is not None and entry_price and not np.isnan(curr_price)):
                            _ehs_profit_above_trigger = curr_profit_pct - extended_hold_trigger_profit
                            if _ehs_profit_above_trigger >= eh_swing_min_gain_above_trigger:
                                _ehs_rsi_val = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns and not pd.isna(data['fast_rsi'].iloc[i]) else np.nan
                                _ehs_bb_val = data['bb_percent'].iloc[i] if 'bb_percent' in data.columns and not pd.isna(data['bb_percent'].iloc[i]) else np.nan
                                _ehs_vol_val = data['volume'].iloc[i] if 'volume' in data.columns else np.nan
                                _ehs_vol_ma_val = data['volume_ma20'].iloc[i] if 'volume_ma20' in data.columns else np.nan
                                _ehs_vol_ratio = _ehs_vol_val / _ehs_vol_ma_val if (not np.isnan(_ehs_vol_val) and not np.isnan(_ehs_vol_ma_val) and _ehs_vol_ma_val > 0) else 0

                                _ehs_can_sell = False
                                _ehs_stk = data['stoch_k'].iloc[i] if 'stoch_k' in data.columns and not pd.isna(data['stoch_k'].iloc[i]) else np.nan
                                _ehs_dist_ma20 = data['dist_ma20'].iloc[i] if 'dist_ma20' in data.columns and not pd.isna(data['dist_ma20'].iloc[i]) else 0
                                _ehs_dist_ma60 = data['dist_ma60'].iloc[i] if 'dist_ma60' in data.columns and not pd.isna(data['dist_ma60'].iloc[i]) else 0

                                # ===== 武装模式：信号后追踪峰值，峰值回撤时卖出 =====
                                if eh_swing_armed_mode and _eh_swing_armed:
                                    # 追踪武装后最高价
                                    if curr_price > _eh_swing_armed_peak:
                                        _eh_swing_armed_peak = curr_price
                                    _armed_elapsed = i - _eh_swing_armed_idx
                                    # 卖出触发：价格从武装峰值回撤X%（支持ATR自适应）
                                    _armed_drop_threshold = eh_swing_trailing_drop_pct  # 默认固定值
                                    if eh_swing_atr_adaptive and data is not None and 'atr_pct' in data.columns:
                                        _atr_pct_val = data['atr_pct'].iloc[i]
                                        if not np.isnan(_atr_pct_val) and _atr_pct_val > 0:
                                            _armed_drop_threshold = _atr_pct_val * eh_swing_atr_mult
                                    if _eh_swing_armed_peak > 0 and _armed_drop_threshold > 0:
                                        _armed_drop_pct = (_eh_swing_armed_peak - curr_price) / _eh_swing_armed_peak * 100
                                        if _armed_drop_pct >= _armed_drop_threshold:
                                            _ehs_can_sell = True
                                            _eh_swing_armed = False
                                    # 解除武装：超过最大天数
                                    if not _ehs_can_sell and _armed_elapsed > eh_swing_armed_max_days:
                                        _eh_swing_armed = False
                                    # 解除武装：RSI回归正常区间（不再超买）
                                    if not _ehs_can_sell and not np.isnan(_ehs_rsi_val) and _ehs_rsi_val < 40:
                                        _eh_swing_armed = False

                                # ===== 多模式信号检测（不在武装状态时） =====
                                if not _ehs_can_sell and not _eh_swing_armed:

                                    # 确认卖出逻辑（兼容旧模式）
                                    if _eh_swing_confirming:
                                        _confirm_elapsed = i - _eh_swing_signal_idx
                                        if _confirm_elapsed >= 1 and _confirm_elapsed <= eh_swing_confirm_days:
                                            _confirm_drop = (curr_price / _eh_swing_signal_price - 1) * 100
                                            if _confirm_drop <= -eh_swing_confirm_drop_pct:
                                                _ehs_can_sell = True
                                                _eh_swing_confirming = False
                                        elif _confirm_elapsed > eh_swing_confirm_days:
                                            _eh_swing_confirming = False
                                    else:
                                        _ehs_signal_detected = False

                                        # --- 模式1: 超买集群（N of 4: RSI + BB + StochK + MFI） ---
                                        _ehs_ob_count = 0
                                        if not np.isnan(_ehs_rsi_val) and _ehs_rsi_val >= eh_swing_rsi_threshold:
                                            _ehs_ob_count += 1
                                        if not np.isnan(_ehs_bb_val) and _ehs_bb_val >= eh_swing_bb_threshold:
                                            _ehs_ob_count += 1
                                        if not np.isnan(_ehs_stk) and _ehs_stk >= eh_swing_ob_stk_threshold:
                                            _ehs_ob_count += 1
                                        if eh_swing_mfi_threshold > 0 and 'mfi_14' in data.columns:
                                            _ehs_mfi = data['mfi_14'].iloc[i]
                                            if not np.isnan(_ehs_mfi) and _ehs_mfi >= eh_swing_mfi_threshold:
                                                _ehs_ob_count += 1
                                        if _ehs_ob_count >= eh_swing_ob_min_count and _ehs_vol_ratio <= eh_swing_volume_surge_block:
                                            _ehs_signal_detected = True

                                        # --- 模式2: MA偏离（价格远离均线 → 回归风险） ---
                                        if not _ehs_signal_detected and eh_swing_dev_ma20_pct > 0:
                                            if _ehs_dist_ma20 >= eh_swing_dev_ma20_pct:
                                                _ehs_signal_detected = True
                                            elif eh_swing_dev_ma60_pct > 0 and _ehs_dist_ma60 >= eh_swing_dev_ma60_pct:
                                                _ehs_signal_detected = True

                                        # --- 模式3: 放量阴线（检测机构出货信号 → 做T） ---
                                        if not _ehs_signal_detected and eh_swing_vol_signal_enabled and 'open' in data.columns:
                                            _vs_count = 0
                                            for _vk in range(max(0, i - eh_swing_vol_signal_lookback + 1), i + 1):
                                                _vs_vol = data['volume'].iloc[_vk]
                                                _vs_vol_ma = data['volume_ma20'].iloc[_vk]
                                                _vs_close = data['close'].iloc[_vk]
                                                _vs_open = data['open'].iloc[_vk]
                                                if (not np.isnan(_vs_vol) and not np.isnan(_vs_vol_ma) and _vs_vol_ma > 0
                                                        and _vs_vol > _vs_vol_ma * eh_swing_vol_signal_mult
                                                        and _vs_close < _vs_open):
                                                    _vs_count += 1
                                            if _vs_count >= eh_swing_vol_signal_count:
                                                _ehs_signal_detected = True

                                        # --- 信号处理：武装模式 or 立即卖出 ---
                                        if _ehs_signal_detected:
                                            if eh_swing_armed_mode:
                                                # 进入武装模式，追踪峰值
                                                _eh_swing_armed = True
                                                _eh_swing_armed_idx = i
                                                _eh_swing_armed_peak = curr_price
                                            elif eh_swing_confirm_days > 0 and not _eh_swing_confirming:
                                                # 旧确认模式
                                                _eh_swing_confirming = True
                                                _eh_swing_signal_idx = i
                                                _eh_swing_signal_price = curr_price
                                            else:
                                                # 立即卖出
                                                _ehs_can_sell = True

                                # 触发条件2：从峰值回撤Xpp（安全网：非超买峰值的回撤保护）
                                if not _ehs_can_sell and not _eh_swing_confirming and not _eh_swing_armed and eh_swing_peak_drawdown > 0:
                                    _ehs_drawdown = extended_hold_max_profit - curr_profit_pct
                                    _ehs_peak_above_trigger = extended_hold_max_profit - extended_hold_trigger_profit
                                    if (_ehs_drawdown >= eh_swing_peak_drawdown
                                            and _ehs_peak_above_trigger >= eh_swing_min_gain_above_trigger):
                                        _ehs_can_sell = True

                                if _ehs_can_sell:
                                    # 计算EH底线的绝对价格
                                    _eh_swing_floor_price = entry_price * (1 + _eh_effective_floor / 100)
                                    _eh_swing_sell_price = curr_price
                                    _eh_swing_original_entry = entry_price
                                    _eh_swing_sell_idx = i
                                    _eh_swing_peak_after_sell = curr_price  # 初始化T卖后的最高价
                                    _eh_swing_active = True
                                    _eh_swing_used = True  # 标记已使用，本EH周期不再做T
                                    # 保存原始交易状态（做T不影响原始买卖逻辑）
                                    _eh_swing_saved_entry_price = entry_price
                                    _eh_swing_saved_hold_days = hold_days
                                    _eh_swing_saved_pending_exit = pending_exit
                                    _eh_swing_saved_trailing_stop_active = trailing_stop_active
                                    _eh_swing_saved_ts_pending = _ts_pending
                                    _eh_swing_saved_ts_pending_days = _ts_pending_days
                                    _eh_swing_saved_dynamic_profit_active = dynamic_profit_active
                                    _eh_swing_saved_max_profit_in_trade = max_profit_in_trade
                                    _eh_swing_saved_is_divergence_entry = is_divergence_entry
                                    _eh_swing_saved_is_w_bottom_entry = is_w_bottom_entry
                                    _eh_swing_saved_is_sideways_entry = is_sideways_entry
                                    _eh_swing_saved_eh_trigger_profit = extended_hold_trigger_profit
                                    _eh_swing_saved_eh_max_profit = extended_hold_max_profit
                                    # 卖出但保持extended_hold_active=True
                                    in_position = False
                                    exit_flags[i] = 1
                                    swing_exit_flags[i] = 1
                                    exit_reasons[i] = 'EH做T-高抛'
                                    entry_price = None
                                    hold_days = 0
                                    pending_exit = False
                                    trailing_stop_active = False
                                    dynamic_profit_active = False
                                    max_profit_in_trade = 0
                                    position[i] = 0
                                    continue

                    # 检查止盈保护（trailing stop）
                    if trailing_stop_trigger > 0 and not trailing_stop_active:
                        if max_profit_in_trade >= trailing_stop_trigger:
                            trailing_stop_active = True

                    if trailing_stop_active and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry:
                        # 双层trailing: 利润越高，floor越紧
                        _ts_effective_level = trailing_stop_level
                        if trailing_stop_trigger2 > 0 and max_profit_in_trade >= trailing_stop_trigger2:
                            _ts_effective_level = trailing_stop_level2
                        trailing_threshold = entry_price * (1 + _ts_effective_level / 100.0)
                        if curr_price <= trailing_threshold:
                            # 智能过滤: 判断是否需要延迟确认
                            _ts_need_confirm = False

                            # 过滤1: 恐慌跌幅过滤
                            if not _ts_need_confirm and trailing_stop_panic_skip < 0 and i > 0:
                                _prev_close = data['close'].iloc[i - 1]
                                if not np.isnan(_prev_close) and _prev_close > 0:
                                    _day_change_pct = (curr_price / _prev_close - 1) * 100
                                    if _day_change_pct <= trailing_stop_panic_skip:
                                        _ts_need_confirm = True

                            # 过滤2: 平稳期突跌过滤 (前N天最大日跌温和→今天突然暴跌=恐慌)
                            if not _ts_need_confirm and trailing_stop_calm_threshold < 0 and i > 1:
                                _calm_start = max(1, i - trailing_stop_calm_lookback)
                                _max_prior_drop = 0.0
                                for _ci in range(_calm_start, i):
                                    _c_prev = data['close'].iloc[_ci - 1]
                                    _c_curr = data['close'].iloc[_ci]
                                    if not np.isnan(_c_prev) and not np.isnan(_c_curr) and _c_prev > 0:
                                        _c_drop = (_c_curr / _c_prev - 1) * 100
                                        if _c_drop < _max_prior_drop:
                                            _max_prior_drop = _c_drop
                                # 前N天最大跌幅温和(>threshold) → 今天是突然下跌 → 需确认
                                if _max_prior_drop > trailing_stop_calm_threshold:
                                    _ts_need_confirm = True

                            if _ts_need_confirm and not _ts_pending:
                                # 首次触发智能过滤，延迟到下一日确认
                                _ts_pending = True
                                _ts_pending_days = 0
                            elif trailing_stop_confirm <= 0 and not _ts_need_confirm and not _ts_pending:
                                # 立即卖出（无确认）
                                in_position = False
                                exit_flags[i] = 1
                                if curr_profit_pct < 0:
                                    stop_flags[i] = 1
                                    _last_loss_exit_idx = i
                                    exit_reasons[i] = f'Trailing止损(level={_ts_effective_level:.1f}%)'
                                else:
                                    profit_target_flags[i] = 1
                                    exit_reasons[i] = f'Trailing止盈(level={_ts_effective_level:.1f}%)'
                                entry_price = None
                                hold_days = 0
                                trailing_stop_active = False
                                dynamic_profit_active = False
                                max_profit_in_trade = 0
                                pending_exit = False
                                _ts_pending = False
                                _ts_pending_days = 0
                                position[i] = 0
                                continue
                            else:
                                # 确认模式: 需要额外N天收在level以下才卖
                                if not _ts_pending:
                                    # 首次触发，开始计数（不算当天）
                                    _ts_pending = True
                                    _ts_pending_days = 0
                                else:
                                    _ts_pending_days += 1
                                if _ts_pending_days >= trailing_stop_confirm:
                                    # 确认完成，执行卖出
                                    in_position = False
                                    exit_flags[i] = 1
                                    if curr_profit_pct < 0:
                                        stop_flags[i] = 1
                                        _last_loss_exit_idx = i
                                        exit_reasons[i] = f'Trailing止损-确认({trailing_stop_confirm}日)'
                                    else:
                                        profit_target_flags[i] = 1
                                        exit_reasons[i] = f'Trailing止盈-确认({trailing_stop_confirm}日)'
                                    entry_price = None
                                    hold_days = 0
                                    trailing_stop_active = False
                                    dynamic_profit_active = False
                                    max_profit_in_trade = 0
                                    pending_exit = False
                                    _ts_pending = False
                                    _ts_pending_days = 0
                                    position[i] = 0
                                    continue
                        else:
                            # 价格回到level以上，取消确认
                            if _ts_pending:
                                _ts_pending = False
                                _ts_pending_days = 0

                    # 检查动态止盈（从最高点回撤X%就卖）
                    if dynamic_profit_trigger > 0 and not dynamic_profit_active:
                        if max_profit_in_trade >= dynamic_profit_trigger:
                            dynamic_profit_active = True

                    if dynamic_profit_active and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry:
                        drawback = max_profit_in_trade - curr_profit_pct
                        if drawback >= dynamic_profit_drawback:
                            in_position = False
                            exit_flags[i] = 1
                            profit_target_flags[i] = 1
                            exit_reasons[i] = f'动态止盈(峰值{max_profit_in_trade:.1f}%回撤{drawback:.1f}%)'
                            entry_price = None
                            hold_days = 0
                            trailing_stop_active = False
                            dynamic_profit_active = False
                            max_profit_in_trade = 0
                            pending_exit = False
                            position[i] = 0
                            continue

                    # 成交量分布退出：窗口内多次放量阴线=机构派发
                    if (dist_exit_enabled and not extended_hold_active and not exit_active
                            and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry
                            and curr_profit_pct >= dist_exit_min_profit
                            and data is not None and 'volume' in data.columns and 'volume_ma20' in data.columns):
                        _dist_days = 0
                        for _dk in range(max(0, i - dist_exit_lookback + 1), i + 1):
                            _dv = data['volume'].iloc[_dk]
                            _dv_ma = data['volume_ma20'].iloc[_dk]
                            _dc = data['close'].iloc[_dk]
                            _do = data['open'].iloc[_dk] if 'open' in data.columns else _dc
                            if (not np.isnan(_dv) and not np.isnan(_dv_ma) and _dv_ma > 0
                                    and _dv > _dv_ma * dist_exit_vol_threshold and _dc < _do):
                                _dist_days += 1
                        if _dist_days >= dist_exit_count:
                            in_position = False
                            exit_flags[i] = 1
                            exit_reasons[i] = f'放量阴线派发({_dist_days}次)'
                            entry_price = None
                            hold_days = 0
                            trailing_stop_active = False
                            dynamic_profit_active = False
                            max_profit_in_trade = 0
                            pending_exit = False
                            pending_exit_days = 0
                            position[i] = 0
                            continue

                    # 滞涨退出：浮盈达标后连续N天未创新高=动量耗尽
                    if (stale_peak_enabled and not extended_hold_active and not exit_active
                            and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry
                            and curr_profit_pct >= stale_peak_min_profit
                            and _days_since_peak >= stale_peak_max_days):
                        in_position = False
                        exit_flags[i] = 1
                        exit_reasons[i] = f'滞涨退出({_days_since_peak}日未创新高)'
                        entry_price = None
                        hold_days = 0
                        trailing_stop_active = False
                        dynamic_profit_active = False
                        max_profit_in_trade = 0
                        _days_since_peak = 0
                        pending_exit = False
                        pending_exit_days = 0
                        position[i] = 0
                        continue

                    # 放量阴线+均线偏离退出：高位出货信号
                    if (dist_madev_exit_enabled and not extended_hold_active and not exit_active
                            and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry
                            and curr_profit_pct >= dist_madev_exit_min_profit
                            and data is not None and 'volume' in data.columns
                            and 'volume_ma20' in data.columns and 'open' in data.columns):
                        _dm_vol = data['volume'].iloc[i]
                        _dm_vol_ma = data['volume_ma20'].iloc[i]
                        _dm_close = data['close'].iloc[i]
                        _dm_open = data['open'].iloc[i]
                        _dm_is_dist = (not np.isnan(_dm_vol) and not np.isnan(_dm_vol_ma)
                                       and _dm_vol_ma > 0
                                       and _dm_vol > _dm_vol_ma * dist_madev_exit_vol_mult
                                       and _dm_close < _dm_open)
                        if _dm_is_dist:
                            _dm_ma_period = dist_madev_exit_ma_period
                            if i >= _dm_ma_period - 1:
                                _dm_ma = np.mean(data['close'].iloc[i - _dm_ma_period + 1:i + 1].values)
                                if _dm_ma > 0:
                                    _dm_dev = (_dm_close - _dm_ma) / _dm_ma * 100
                                    if _dm_dev >= dist_madev_exit_dev_pct:
                                        in_position = False
                                        exit_flags[i] = 1
                                        exit_reasons[i] = f'放量阴线+偏离MA{_dm_ma_period}({_dm_dev:.1f}%)'
                                        entry_price = None
                                        hold_days = 0
                                        trailing_stop_active = False
                                        dynamic_profit_active = False
                                        max_profit_in_trade = 0
                                        pending_exit = False
                                        pending_exit_days = 0
                                        position[i] = 0
                                        continue

                    # 多指标超买集群退出：利润在10-22%区间，多个振荡指标同时超买
                    if (ob_cluster_exit_enabled and not extended_hold_active and not exit_active
                            and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry
                            and curr_profit_pct >= ob_cluster_exit_min_profit
                            and curr_profit_pct < ob_cluster_exit_max_profit
                            and data is not None):
                        _ob_count = 0
                        _ob_rsi = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns else np.nan
                        _ob_stk = data['stoch_k'].iloc[i] if 'stoch_k' in data.columns else np.nan
                        _ob_cci = data['cci_20'].iloc[i] if 'cci_20' in data.columns else np.nan
                        _ob_mfi = data['mfi_14'].iloc[i] if 'mfi_14' in data.columns else np.nan
                        if not np.isnan(_ob_rsi) and _ob_rsi >= ob_cluster_exit_rsi_thresh:
                            _ob_count += 1
                        if not np.isnan(_ob_stk) and _ob_stk >= ob_cluster_exit_stk_thresh:
                            _ob_count += 1
                        if not np.isnan(_ob_cci) and _ob_cci >= ob_cluster_exit_cci_thresh:
                            _ob_count += 1
                        if not np.isnan(_ob_mfi) and _ob_mfi >= ob_cluster_exit_mfi_thresh:
                            _ob_count += 1
                        if _ob_count >= ob_cluster_exit_min_count:
                            in_position = False
                            exit_flags[i] = 1
                            profit_target_flags[i] = 1
                            exit_reasons[i] = f'超买集群退出({_ob_count}指标超买)'
                            entry_price = None
                            hold_days = 0
                            trailing_stop_active = False
                            dynamic_profit_active = False
                            max_profit_in_trade = 0
                            pending_exit = False
                            pending_exit_days = 0
                            position[i] = 0
                            continue

                    # 放量冲高回落退出：上影线>实体+收盘下半区+放量=冲高回落
                    if (vol_climax_exit_enabled and not extended_hold_active and not exit_active
                            and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry
                            and entry_price and not np.isnan(curr_price)
                            and curr_profit_pct >= vol_climax_exit_min_profit
                            and data is not None and 'volume' in data.columns
                            and 'volume_ma20' in data.columns):
                        _vc_vol = data['volume'].iloc[i]
                        _vc_vol_ma = data['volume_ma20'].iloc[i]
                        _vc_high = data['high'].iloc[i]
                        _vc_low = data['low'].iloc[i]
                        _vc_open = data['open'].iloc[i]
                        _vc_close = data['close'].iloc[i]
                        _vc_range = _vc_high - _vc_low
                        if (not np.isnan(_vc_vol) and not np.isnan(_vc_vol_ma) and _vc_vol_ma > 0
                                and _vc_vol > _vc_vol_ma * vol_climax_exit_vol_mult
                                and _vc_range > 0):
                            _vc_body_top = max(_vc_open, _vc_close)
                            _vc_upper_shadow = _vc_high - _vc_body_top
                            _vc_body = abs(_vc_close - _vc_open)
                            # 上影线>实体 + 收盘在K线下半区
                            if _vc_upper_shadow > _vc_body and _vc_close < (_vc_high + _vc_low) / 2:
                                _vc_fire = True
                                if vol_climax_exit_require_new_high and i >= 5:
                                    _vc_fire = _vc_high >= data['high'].iloc[max(0, i - 5):i].max()
                                if _vc_fire:
                                    in_position = False
                                    exit_flags[i] = 1
                                    profit_target_flags[i] = 1
                                    exit_reasons[i] = '放量冲高回落'
                                    entry_price = None
                                    hold_days = 0
                                    trailing_stop_active = False
                                    dynamic_profit_active = False
                                    max_profit_in_trade = 0
                                    pending_exit = False
                                    pending_exit_days = 0
                                    position[i] = 0
                                    continue

                    # ROC动量衰竭退出：ROC正值但连续下降=加速度为负
                    if (roc_fade_exit_enabled and not extended_hold_active and not exit_active
                            and (not is_w_bottom_entry or _wb_std_exit) and not is_sideways_entry
                            and curr_profit_pct >= roc_fade_exit_min_profit
                            and data is not None and 'roc_10' in data.columns
                            and i >= roc_fade_exit_declining_days):
                        _rf_declining = True
                        for _rk in range(roc_fade_exit_declining_days):
                            _rk_idx = i - _rk
                            _rk_prev = _rk_idx - 1
                            if _rk_prev >= 0:
                                _rk_roc = data['roc_10'].iloc[_rk_idx]
                                _rk_roc_prev = data['roc_10'].iloc[_rk_prev]
                                if (pd.isna(_rk_roc) or pd.isna(_rk_roc_prev)
                                        or _rk_roc >= _rk_roc_prev
                                        or _rk_roc <= roc_fade_exit_roc_floor):
                                    _rf_declining = False
                                    break
                            else:
                                _rf_declining = False
                                break
                        if _rf_declining:
                            in_position = False
                            exit_flags[i] = 1
                            profit_target_flags[i] = 1
                            exit_reasons[i] = f'ROC动量衰竭({roc_fade_exit_declining_days}日连降)'
                            entry_price = None
                            hold_days = 0
                            trailing_stop_active = False
                            dynamic_profit_active = False
                            max_profit_in_trade = 0
                            pending_exit = False
                            pending_exit_days = 0
                            position[i] = 0
                            continue

                # W底买入的专属退出逻辑（动态缓冲期内：跌破第二个低点3%止损，涨超15%止盈）
                # 缓冲期规则：gap ≤ 45天 → 15天；gap > 45天 → gap/3
                # 当 w_bottom_use_standard_exit=True 时跳过此块，使用标准退出
                if is_w_bottom_entry and not _wb_std_exit and w_bottom_price and entry_price and not pd.isna(curr_price):
                    buffer_days = 15 if (not w_bottom_gap or w_bottom_gap <= 45) else int(w_bottom_gap // 3)
                    if hold_days <= buffer_days:
                        # 缓冲期内使用W底专属逻辑
                        # 止损：跌破第二个低点的3%（第二个低点是确认买入的关键支撑位）
                        # 【关键】使用当天最低价判断止损，而不是收盘价，这样更接近实际交易
                        curr_low = data['low'].iloc[i] if 'low' in data.columns else curr_price
                        stop_threshold = w_bottom_price * (1 - wb_buffer_stop_pct / 100.0)
                        if curr_low <= stop_threshold:
                            if data is not None and 'date' in data.columns:
                                sell_date = data['date'].iloc[i]
                                logger.info(f"[W底止损] {sell_date} 跌破止损线{stop_threshold:.2f}，当前价{curr_price:.2f}")
                            # 【关键】在退出行也标记w_bottom_price，用于显示准确的止损原因
                            if 'w_bottom_stop_exit' not in data.columns:
                                data['w_bottom_stop_exit'] = False
                            data.loc[data.index[i], 'w_bottom_stop_exit'] = True
                            in_position = False
                            exit_flags[i] = 1
                            stop_flags[i] = 1
                            exit_reasons[i] = f'W底止损(跌破支撑{wb_buffer_stop_pct:.0f}%)'
                            entry_price = None
                            is_w_bottom_entry = False
                            w_bottom_price = None
                            w_bottom_gap = None
                            hold_days = 0
                            position[i] = 0  # 【修复】在continue前设置position
                            continue

                        # 止盈：涨超wb_buffer_profit_pct%
                        profit_threshold = entry_price * (1 + wb_buffer_profit_pct / 100.0)
                        if curr_price >= profit_threshold:
                            if data is not None and 'date' in data.columns:
                                sell_date = data['date'].iloc[i]
                                logger.info(f"[W底止盈] {sell_date} 涨超15%止盈，当前价{curr_price:.2f}")
                            in_position = False
                            exit_flags[i] = 1
                            profit_target_flags[i] = 1
                            exit_reasons[i] = f'W底止盈({wb_buffer_profit_pct:.0f}%)'
                            entry_price = None
                            is_w_bottom_entry = False
                            w_bottom_price = None
                            w_bottom_gap = None
                            hold_days = 0
                            position[i] = 0  # 【修复】在continue前设置position
                            continue

                        # 【关键】缓冲期内未触发止损/止盈，继续持有，跳过后面的传统卖出逻辑
                        position[i] = 1  # 【修复】在continue前设置position
                        continue
                    else:
                        # 持有超过缓冲期，传统卖出逻辑接管
                        if exit_active:
                            in_position = False
                            exit_flags[i] = 1
                            exit_reasons[i] = 'W底-趋势转空退出'
                            entry_price = None
                            is_w_bottom_entry = False
                            w_bottom_price = None
                            w_bottom_gap = None
                            hold_days = 0
                        elif _trade_stop_loss > 0 and entry_price:
                            threshold = entry_price * (1 - _trade_stop_loss / 100.0)
                            if curr_price <= threshold:
                                in_position = False
                                exit_flags[i] = 1
                                stop_flags[i] = 1
                                exit_reasons[i] = f'W底-止损({_trade_stop_loss:.1f}%)'
                                entry_price = None
                                is_w_bottom_entry = False
                                w_bottom_price = None
                                w_bottom_gap = None
                                hold_days = 0

                # 底背离买入的特殊退出逻辑（不使用15%止盈，只用ATR+止损控制）
                elif is_divergence_entry and entry_price and not pd.isna(curr_price):
                    # 条件1：未达到最短持有天数，只有止损才退出
                    if hold_days < min_hold_days:
                        # 只有触发止损时才退出
                        if _trade_stop_loss > 0 and entry_price:
                            threshold = entry_price * (1 - _trade_stop_loss / 100.0)
                            if curr_price <= threshold:
                                in_position = False
                                exit_flags[i] = 1
                                stop_flags[i] = 1
                                exit_reasons[i] = f'底背离-早期止损({_trade_stop_loss:.1f}%)'
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
                                    exit_reasons[i] = f'底背离-RSI衰减退出(dRSI={rsi_change:.1f})'
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                                # 否则只检查止损
                                elif _trade_stop_loss > 0 and entry_price:
                                    threshold = entry_price * (1 - _trade_stop_loss / 100.0)
                                    if curr_price <= threshold:
                                        in_position = False
                                        exit_flags[i] = 1
                                        stop_flags[i] = 1
                                        exit_reasons[i] = f'底背离-止损({_trade_stop_loss:.1f}%)'
                                        entry_price = None
                                        is_divergence_entry = False
                                        entry_rsi = None
                                        hold_days = 0
                            else:
                                # RSI数据不可用，按正常逻辑
                                if exit_active:
                                    in_position = False
                                    exit_flags[i] = 1
                                    exit_reasons[i] = '底背离-趋势转空退出'
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                        # 完全忽略RSI退出
                        elif ignore_rsi_exit:
                            if _trade_stop_loss > 0 and entry_price:
                                threshold = entry_price * (1 - _trade_stop_loss / 100.0)
                                if curr_price <= threshold:
                                    in_position = False
                                    exit_flags[i] = 1
                                    stop_flags[i] = 1
                                    exit_reasons[i] = f'底背离-止损({_trade_stop_loss:.1f}%)'
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0
                        # 默认逻辑（按正常退出）
                        else:
                            if exit_active:
                                in_position = False
                                exit_flags[i] = 1
                                exit_reasons[i] = '底背离-趋势转空退出'
                                entry_price = None
                                is_divergence_entry = False
                                entry_rsi = None
                                hold_days = 0
                            elif _trade_stop_loss > 0 and entry_price:
                                threshold = entry_price * (1 - _trade_stop_loss / 100.0)
                                if curr_price <= threshold:
                                    in_position = False
                                    exit_flags[i] = 1
                                    stop_flags[i] = 1
                                    exit_reasons[i] = f'底背离-止损({_trade_stop_loss:.1f}%)'
                                    entry_price = None
                                    is_divergence_entry = False
                                    entry_rsi = None
                                    hold_days = 0

                # 震荡市场买入的特殊退出逻辑（固定止盈止损）
                elif is_sideways_entry and entry_price and not pd.isna(curr_price):
                    # 获取布林带和RSI数据
                    bb_percent = data['bb_percent'].iloc[i] if 'bb_percent' in data.columns and not pd.isna(data['bb_percent'].iloc[i]) else 0.5
                    fast_rsi = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns and not pd.isna(data['fast_rsi'].iloc[i]) else 50

                    # 出场条件1：触及布林带上轨 + RSI超买
                    if bb_percent >= _sw_exit_bb_upper and fast_rsi > _sw_exit_rsi_upper:
                        in_position = False
                        exit_flags[i] = 1
                        sideways_exit_type[i] = 1
                        exit_reasons[i] = '震荡-上轨+RSI超买退出'
                        entry_price = None
                        is_sideways_entry = False
                        hold_days = 0
                        if data is not None and 'date' in data.columns:
                            sell_date = data['date'].iloc[i]
                            logger.info(f"[Aroon震荡上轨退出] {sell_date} 触及上轨+RSI超买，卖出价格{curr_price:.2f}")
                        continue

                    # 出场条件2：固定止盈
                    profit_pct = (curr_price / entry_price - 1) * 100
                    if profit_pct >= _sw_exit_tp_pct:
                        in_position = False
                        exit_flags[i] = 1
                        profit_target_flags[i] = 1
                        sideways_exit_type[i] = 2
                        exit_reasons[i] = f'震荡-止盈({_sw_exit_tp_pct:.1f}%)'
                        entry_price = None
                        is_sideways_entry = False
                        hold_days = 0
                        if data is not None and 'date' in data.columns:
                            sell_date = data['date'].iloc[i]
                            logger.info(f"[Aroon震荡止盈] {sell_date} 达到8.5%止盈目标，卖出价格{curr_price:.2f}")
                        continue

                    # 出场条件3：固定止损
                    if profit_pct <= -_sw_exit_sl_pct:
                        in_position = False
                        exit_flags[i] = 1
                        stop_flags[i] = 1
                        sideways_exit_type[i] = 3
                        exit_reasons[i] = f'震荡-止损({_sw_exit_sl_pct:.1f}%)'
                        entry_price = None
                        is_sideways_entry = False
                        hold_days = 0
                        if data is not None and 'date' in data.columns:
                            sell_date = data['date'].iloc[i]
                            logger.info(f"[Aroon震荡止损] {sell_date} 触发1.0%止损，卖出价格{curr_price:.2f}")
                        continue

                    # 继续持有

                # 非底背离、非W底、非震荡市场买入，按正常逻辑处理
                elif not is_divergence_entry and not is_w_bottom_entry and not is_sideways_entry:
                    # 高抛低吸：检查是否满足swing sell条件
                    if (swing_trade_enabled and data is not None and not pending_exit
                            and swing_state == 0 and entry_price and not np.isnan(curr_price)):
                        sw_profit = (curr_price / entry_price - 1) * 100
                        sw_aroon = data['aroon_osc'].iloc[i] if 'aroon_osc' in data.columns and not pd.isna(data['aroon_osc'].iloc[i]) else np.nan
                        sw_bb_pct = data['bb_percent'].iloc[i] if 'bb_percent' in data.columns and not pd.isna(data['bb_percent'].iloc[i]) else np.nan
                        sw_rsi = data['fast_rsi'].iloc[i] if 'fast_rsi' in data.columns and not pd.isna(data['fast_rsi'].iloc[i]) else np.nan
                        sw_vol = data['volume'].iloc[i] if 'volume' in data.columns else np.nan
                        sw_vol_ma = data['volume_ma20'].iloc[i] if 'volume_ma20' in data.columns else np.nan
                        sw_trend = data['trend_direction'].iloc[i] if 'trend_direction' in data.columns and not pd.isna(data['trend_direction'].iloc[i]) else 0
                        sw_main_wave = bool(data['main_wave_signal'].iloc[i]) if 'main_wave_signal' in data.columns and not pd.isna(data['main_wave_signal'].iloc[i]) else False

                        sw_can_sell = True
                        # C1: 最少持仓天数
                        if hold_days < swing_min_hold_days:
                            sw_can_sell = False
                        # C2: 最少浮盈
                        if sw_profit < swing_min_profit_pct:
                            sw_can_sell = False
                        # C2b: 浮盈过高则不高抛（保护大牛股，让利润继续跑）
                        if sw_profit > swing_max_profit_pct:
                            sw_can_sell = False
                        # C3: 震荡市（aroon近零）
                        if np.isnan(sw_aroon) or abs(sw_aroon) >= swing_aroon_threshold:
                            sw_can_sell = False
                        # C4: 价格在BB高位（必须满足）
                        if np.isnan(sw_bb_pct) or sw_bb_pct < swing_bb_sell_threshold:
                            sw_can_sell = False
                        # C5: 严格条件 - RSI超买 且 涨幅够大（必须同时满足）
                        # 改用AND逻辑，避免在趋势中过早卖出
                        swing_sell_gain_threshold = float(self.config['swing_sell_gain_threshold'])
                        rsi_overbought = (not np.isnan(sw_rsi)) and sw_rsi >= swing_rsi_sell_threshold
                        gain_high = sw_profit >= swing_sell_gain_threshold
                        if not (rsi_overbought and gain_high):
                            sw_can_sell = False
                        # C6: 趋势未反转（只在趋势向上时做波段）
                        if exit_active:
                            sw_can_sell = False
                        # C7: 非放量突破
                        if (not np.isnan(sw_vol) and not np.isnan(sw_vol_ma)
                                and sw_vol_ma > 0 and sw_vol / sw_vol_ma > swing_volume_surge_block):
                            sw_can_sell = False
                        # C8: 非主升浪
                        if sw_main_wave:
                            sw_can_sell = False
                        # C9: SuperTrend仍看多
                        if sw_trend != 1:
                            sw_can_sell = False

                        if sw_can_sell:
                            swing_state = 1
                            swing_sell_price = curr_price
                            swing_sell_idx = i
                            swing_original_entry_price = entry_price
                            swing_lowest_price = 0.0  # 重置最低价追踪
                            # 保存交易状态（回买时恢复，与EH做T保持一致）
                            swing_saved_entry_price = entry_price
                            swing_saved_hold_days = hold_days
                            swing_saved_pending_exit = pending_exit
                            swing_saved_trailing_stop_active = trailing_stop_active
                            swing_saved_ts_pending = _ts_pending
                            swing_saved_ts_pending_days = _ts_pending_days
                            swing_saved_dynamic_profit_active = dynamic_profit_active
                            swing_saved_max_profit_in_trade = max_profit_in_trade
                            swing_saved_is_divergence_entry = is_divergence_entry
                            swing_saved_is_w_bottom_entry = is_w_bottom_entry
                            swing_saved_is_sideways_entry = is_sideways_entry
                            in_position = False
                            exit_flags[i] = 1
                            swing_exit_flags[i] = 1
                            exit_reasons[i] = '持仓做T-高抛'
                            entry_price = None
                            hold_days = 0
                            pending_exit = False
                            pending_exit_days = 0
                            trailing_stop_active = False
                            dynamic_profit_active = False
                            max_profit_in_trade = 0
                            position[i] = 0
                            continue

                    # 止损始终立即执行（不延迟）
                    if stop_loss_pct > 0 and entry_price and not pd.isna(curr_price):
                        # 早期止损收紧：前N天使用更紧的止损
                        _effective_sl = stop_loss_pct
                        if early_stop_days > 0 and hold_days <= early_stop_days:
                            _effective_sl = early_stop_loss_pct
                        threshold = entry_price * (1 - _effective_sl / 100.0)
                        if curr_price <= threshold:
                            in_position = False
                            exit_flags[i] = 1
                            stop_flags[i] = 1
                            _last_loss_exit_idx = i  # 记录止损退出位置（用于冷却期）
                            if _effective_sl != stop_loss_pct:
                                exit_reasons[i] = f'早期止损({_effective_sl:.1f}%,{hold_days}日内)'
                            else:
                                exit_reasons[i] = f'止损({_effective_sl:.1f}%)'
                            entry_price = None
                            hold_days = 0
                            pending_exit = False
                            pending_exit_days = 0
                            continue

                    # 负动量提前退出：亏损达标且动量恶化时提前退出
                    if (neg_momentum_exit_enabled and entry_price and not pd.isna(curr_price)
                            and hold_days >= neg_momentum_min_days and not extended_hold_active):
                        _nm_profit = (curr_price / entry_price - 1) * 100
                        if _nm_profit < -neg_momentum_loss_threshold:
                            # 检查RSI是否连续下降
                            _nm_rsi_declining = False
                            if data is not None and 'fast_rsi' in data.columns and i >= neg_momentum_rsi_declining_days:
                                _nm_rsi_declining = True
                                for _nm_k in range(neg_momentum_rsi_declining_days):
                                    _nm_idx = i - _nm_k
                                    _nm_idx_prev = _nm_idx - 1
                                    if _nm_idx_prev >= 0:
                                        _nm_rsi_curr = data['fast_rsi'].iloc[_nm_idx]
                                        _nm_rsi_prev = data['fast_rsi'].iloc[_nm_idx_prev]
                                        if pd.isna(_nm_rsi_curr) or pd.isna(_nm_rsi_prev) or _nm_rsi_curr >= _nm_rsi_prev:
                                            _nm_rsi_declining = False
                                            break
                            if _nm_rsi_declining:
                                in_position = False
                                exit_flags[i] = 1
                                stop_flags[i] = 1
                                exit_reasons[i] = '负动量提前退出'
                                entry_price = None
                                hold_days = 0
                                pending_exit = False
                                pending_exit_days = 0
                                continue

                    # 处理待反弹卖出状态
                    if pending_exit:
                        pending_exit_days += 1
                        # 检查是否满足反弹条件或超时
                        prev_close = data['close'].iloc[i - 1] if data is not None and i > 0 else curr_price
                        day_change = (curr_price / prev_close - 1) * 100 if prev_close > 0 else 0
                        bounce_from_signal = (curr_price / pending_exit_price - 1) * 100 if pending_exit_price > 0 else 0

                        # 反弹条件：当天收涨 或 价格回到信号价附近/之上 或 超时
                        bounce_ok = (day_change > 0)  # 阳线
                        if bounce_exit_bounce_pct > 0:
                            bounce_ok = bounce_ok or (bounce_from_signal >= -bounce_exit_bounce_pct)
                        timeout = (pending_exit_days >= bounce_exit_max_wait)

                        if bounce_ok or timeout:
                            in_position = False
                            exit_flags[i] = 1
                            if timeout:
                                exit_reasons[i] = f'反弹卖出-超时({pending_exit_days}日)'
                            else:
                                exit_reasons[i] = '反弹卖出-等待反弹后退出'
                            entry_price = None
                            hold_days = 0
                            pending_exit = False
                            pending_exit_days = 0
                        # else: 继续持有等待反弹

                    elif exit_active:
                        # 延长持仓：浮盈>35%且持仓>25天且MA120上升 → 改用MA120退出线
                        _eh_profit = ((curr_price / entry_price - 1) * 100) if entry_price and not np.isnan(curr_price) and entry_price > 0 else 0
                        _eh_ma120_rising = False
                        if data is not None and 'ma_120' in data.columns and i >= 40:
                            _eh_ma120_val = data['ma_120'].iloc[i]
                            _eh_ma120_rising = not np.isnan(_eh_ma120_val) and _eh_ma120_val > data['ma_120'].iloc[i - 40]
                        _eh_min_hold = int(self.config['extended_hold_min_days'])
                        _eh_profit_cap = float(self.config['extended_hold_profit_cap'])
                        if (not extended_hold_active and _eh_profit > eh_profit_threshold
                                and _eh_profit < _eh_profit_cap
                                and hold_days >= _eh_min_hold and _eh_ma120_rising):
                            extended_hold_active = True
                            extended_hold_trigger_profit = _eh_profit
                            extended_hold_max_profit = _eh_profit  # 从触发时开始追踪峰值
                            _eh_swing_used = False  # 新EH周期重置做T标记
                            _eh_swing_confirming = False  # 新EH周期重置确认状态
                            _eh_swing_armed = False  # 新EH周期重置武装模式
                            _eh_overbought_seen = False  # 新EH周期重置超买标记
                            if data is not None and 'date' in data.columns:
                                logger.debug(f"[EH_ACTIVATE] {data['date'].iloc[i]} profit={_eh_profit:.1f}% hold={hold_days}d")
                        elif extended_hold_active:
                            pass  # 由每日MA120检查处理退出
                        # 反弹卖出逻辑：检查是否在暴跌中
                        elif bounce_exit_enabled and data is not None and i > 0:
                            prev_close = data['close'].iloc[i - 1]
                            day_change = (curr_price / prev_close - 1) * 100 if prev_close > 0 else 0

                            if day_change < bounce_exit_drop_threshold:
                                # 暴跌中，检查多因子条件决定是否延迟
                                # 强制立即卖出的条件（基于深度分析）
                                force_immediate = False

                                # 因子1: 大赢家大跌 - 已经赚了很多，趋势反转
                                bounce_big_win_exit = float(self.config.get('bounce_big_win_exit', 0))
                                if bounce_big_win_exit > 0 and entry_price and curr_price > 0:
                                    curr_pnl = (curr_price / entry_price - 1) * 100
                                    if curr_pnl > bounce_big_win_exit and day_change < -5:
                                        force_immediate = True

                                # 因子2: BB高位大跌 - 可能是假突破回落
                                bounce_bb_immediate = float(self.config.get('bounce_bb_immediate', 0))
                                if bounce_bb_immediate > 0 and 'bb_percent' in data.columns:
                                    bb_val = data['bb_percent'].iloc[i] if not pd.isna(data['bb_percent'].iloc[i]) else 0.5
                                    if bb_val > bounce_bb_immediate:
                                        force_immediate = True

                                if force_immediate:
                                    # 强制立即卖出
                                    in_position = False
                                    exit_flags[i] = 1
                                    exit_reasons[i] = '趋势转空-暴跌强制退出'
                                    entry_price = None
                                    hold_days = 0
                                else:
                                    # 进入待卖出状态
                                    pending_exit = True
                                    pending_exit_price = curr_price
                                    pending_exit_days = 0
                            else:
                                # 非暴跌，正常卖出
                                in_position = False
                                exit_flags[i] = 1
                                exit_reasons[i] = '趋势转空退出'
                                entry_price = None
                                hold_days = 0
                        else:
                            in_position = False
                            exit_flags[i] = 1
                            exit_reasons[i] = '趋势转空退出'
                            entry_price = None
                            hold_days = 0

            position[i] = 1 if in_position else 0

        return position, entry_flags, exit_flags, stop_flags, profit_target_flags, sideways_exit_type, swing_exit_flags, swing_rebuy_reasons, entry_reasons, exit_reasons

    @staticmethod
    def _build_entry_reasons(data: pd.DataFrame, entry_flags: np.ndarray) -> pd.Series:
        reasons = [''] * len(data)
        mtf_disabled = False
        # 当策略层面关闭MTF时，避免原因里误写“高时间框架一致”
        if 'mtf_info' in data.columns:
            try:
                mtf_disabled = data['mtf_info'].astype(str).str.contains("'status': 'disabled'|\"status\": \"disabled\"|disabled", regex=True).any()
            except Exception:
                mtf_disabled = False
        for idx, flag in enumerate(entry_flags):
            if flag:
                row = data.iloc[idx]
                parts = []
                
                # 检查是否为高抛低吸回买
                swing_type = row.get('swing_exit_type', 0)
                if swing_type == 2:
                    parts.append("持仓做T-低吸")
                # 检查是否为追高回调买入
                elif row.get('chase_pullback_entry', False):
                    parts.append("追高回调买入")
                # 检查是否为底背离入场（独立生效）
                elif row.get('bullish_divergence_signal', False):
                    parts.append("底背离信号（独立生效）")
                # 检查是否为W底入场（独立生效）
                elif row.get('w_bottom_signal', False):
                    parts.append("W底形态（双底确认）")
                # 检查是否为震荡市场入场（独立生效）
                elif row.get('sideways_entry', False):
                    parts.append("Aroon震荡入场（BB下轨+RSI超卖）")
                # 标准RSI入场
                else:
                    if row.get('golden_cross', False):
                        parts.append("RSI金叉")
                    elif row.get('rsi_relaxed_condition', False):
                        parts.append("RSI多头延续")
                    else:
                        parts.append("RSI多头")
                    if row.get('is_heikin_bullish', False):
                        parts.append("Heikin Ashi 阳线")
                    if row.get('trend_direction', 0) == 1:
                        parts.append("ATR趋势多头")
                    if (not mtf_disabled) and row.get('mtf_bias', True):
                        parts.append("高时间框架一致")
                        
                # 确保至少有一个原因（不应该为空）
                if not parts:
                    parts.append('RSI趋势买入')
                reasons[idx] = ' + '.join(parts)
        return pd.Series(reasons, index=data.index)

    @staticmethod
    def _build_exit_reasons(data: pd.DataFrame, exit_flags: np.ndarray) -> pd.Series:
        reasons = [''] * len(data)
        for idx, flag in enumerate(exit_flags):
            if flag:
                parts = []

                # 优先检查高抛低吸退出
                swing_exit = data.iloc[idx].get('swing_exit_type', 0)
                if swing_exit == 1:
                    # 高抛卖出统一用"持仓做T-高抛"，因为卖出时不知道后续能否接回
                    parts.append("持仓做T-高抛")
                    reasons[idx] = ' + '.join(parts)
                    continue

                # 优先检查震荡退出
                sw_exit = data.iloc[idx].get('sideways_exit_type', 0)
                if sw_exit == 1:
                    parts.append("Aroon震荡上轨退出（BB上轨+RSI超买）")
                elif sw_exit == 2:
                    parts.append("Aroon震荡止盈8.5%")
                elif sw_exit == 3:
                    parts.append("Aroon震荡止损1.0%")
                # 检查止盈退出
                elif data.iloc[idx].get('profit_target_exit'):
                    parts.append("底背离止盈15%")
                else:
                    parts.append("ATR趋势转空")
                
                if data.iloc[idx].get('death_cross'):
                    parts.append("RSI死叉确认")
                if data.iloc[idx].get('exit_ma_filter_break'):
                    parts.append("EMA16>MA45下连续3日跌破MA16")
                if data.iloc[idx].get('stop_loss_exit'):
                    stop_val = data.iloc[idx].get('stop_loss_pct')
                    # 检查是否为W底止损（跌破第二个低点3%）
                    if data.iloc[idx].get('w_bottom_stop_exit'):
                        parts.append("跌破W底支撑3%")
                    elif stop_val and not pd.isna(stop_val):
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

    def _detect_w_bottom(self, data: pd.DataFrame) -> pd.Series:
        """
        检测W底形态买入信号
        
        逻辑：
        1. 找到一系列低点（close < pre-close && close < next-close）
        2. 低点必须是过去20日的最低点
        3. 选取2个低点，收盘价比较接近（差异<5%）
        4. 两个低点间隔大于30天
        5. 区间内所有收盘价都高于这两个低点
        6. 第二个低点的第二天作为买入信号（避免未来函数）
        
        Returns:
            pd.Series: W底买入信号
        """
        w_bottom_signals = pd.Series(False, index=data.index)
        
        if len(data) < 60:  # 至少需要60天数据
            return w_bottom_signals
        
        close = data['close'].values
        n = len(close)
        
        # 从配置读取参数
        lookback_period = int(self.config['trend_w_bottom_lookback'])
        min_gap_days = int(self.config['trend_w_bottom_min_gap'])
        price_tolerance = float(self.config['trend_w_bottom_price_tolerance'])
        
        # 第一步：找到所有局部低点
        local_lows = []
        for i in range(lookback_period, n - 1):
            # 检查是否是局部低点：close < pre-close && close < next-close
            if close[i] < close[i-1] and close[i] < close[i+1]:
                # 检查是否是过去20日最低点
                window_min = np.min(close[max(0, i-lookback_period+1):i+1])
                if close[i] <= window_min:
                    local_lows.append((i, close[i]))
        
        if len(local_lows) < 2:
            return w_bottom_signals
        
        # 第二步：寻找符合条件的W底
        for i in range(len(local_lows) - 1):
            idx1, price1 = local_lows[i]
            
            for j in range(i + 1, len(local_lows)):
                idx2, price2 = local_lows[j]
                
                # 检查间隔
                gap = idx2 - idx1
                if gap < min_gap_days:
                    continue
                
                # 【关键过滤】第二个低点不能明显低于第一个低点（防止下跌中继）
                # 允许第二个低点略低（5%容差内），但不能明显更低
                if price2 < price1 * 0.95:  # 第二个低点比第一个低超过5%，说明还在下跌
                    continue
                
                # 检查价格相似度
                lower_price = min(price1, price2)
                higher_price = max(price1, price2)
                if (higher_price - lower_price) / lower_price > price_tolerance:
                    continue
                
                # 检查区间内所有价格是否都高于这两个低点
                interval_prices = close[idx1:idx2+1]
                min_low = min(price1, price2)
                
                # 允许在低点当天等于最低价，但其他天必须高于
                valid_interval = True
                for k, price in enumerate(interval_prices):
                    actual_idx = idx1 + k
                    # 跳过两个低点本身
                    if actual_idx == idx1 or actual_idx == idx2:
                        continue
                    # 其他天必须高于最低点
                    if price <= min_low:
                        valid_interval = False
                        break
                
                if not valid_interval:
                    continue
                
                # 【关键过滤】振幅检查：从低点到中间高点的涨幅要>=30%
                # 这确保W底有足够明显的反弹力度，过滤掉振幅不足的弱反弹
                between_high = np.max(data['high'].iloc[idx1:idx2+1]) if 'high' in data.columns else np.max(interval_prices)
                w_bottom_low = min(price1, price2)
                amplitude = (between_high - w_bottom_low) / w_bottom_low * 100
                if amplitude < 30.0:
                    continue
                
                # 【关键过滤】避免缓慢下跌磨底 - 使用线性回归判断
                # 从中间高点到第二个低点，如果是明显的单边下跌趋势，则过滤
                # 找到中间高点的位置
                high_prices = data['high'].iloc[idx1:idx2+1].values if 'high' in data.columns else interval_prices
                max_high_idx_relative = np.argmax(high_prices)
                max_high_idx = idx1 + max_high_idx_relative
                
                # 如果高点到低点2的距离足够长（>30天），进行线性回归判断
                if idx2 - max_high_idx > 30:
                    # 提取从高点到低点2的收盘价
                    decline_segment = close[max_high_idx:idx2+1]
                    x = np.arange(len(decline_segment))
                    
                    # 线性回归
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, decline_segment)
                    r_squared = r_value ** 2
                    
                    # 判断标准：斜率为负且R²>0.7，说明是明显的单边下跌（缓慢阴跌）
                    # R²>0.7表示价格走势高度线性化，即持续单边下跌而非震荡
                    if slope < 0 and r_squared > 0.7:
                        continue
                
                # 旧的过滤条件保留：如果间隔>120天且两个低点价格非常接近（<3%），也过滤
                price_diff_pct = abs(price2 - price1) / price1 * 100
                if gap > 120 and price_diff_pct < 3.0:
                    continue
                
                # 【关键过滤】第二个低点的第二天必须收阳线
                # 这说明有资金开始介入，是更强的买入信号
                signal_idx = idx2 + 1  # 第二个低点的第二天
                if signal_idx >= n:
                    continue
                
                # 检查第二天是否收阳线，且阳线实体要有一定强度
                open_price = data['open'].iloc[signal_idx] if 'open' in data.columns else close[signal_idx - 1]
                close_price = close[signal_idx]
                is_bullish = close_price > open_price
                
                if not is_bullish:
                    # 第二天没有收阳线，跳过这个W底
                    continue
                
                # 【关键过滤】阳线实体强度：涨幅至少1%，避免弱势阳线
                body_strength = (close_price - open_price) / open_price * 100
                if body_strength < 1.0:
                    # 阳线太弱（涨幅<1%），跳过这个W底
                    continue
                
                # 找到了有效的W底，在第二天收盘买入（signal_idx就是买入日）
                w_bottom_signals.iloc[signal_idx] = True
                
                # 【重要】保存W底的第二个低点价格和间隔天数
                # 第二个低点是确认买入的关键支撑位，跌破第二个低点说明形态破坏
                # 间隔天数用于计算动态缓冲期（gap ≤ 45天 → 15天；gap > 45天 → gap/3）
                w_bottom_price = price2  # 使用第二个低点作为止损基准
                if 'w_bottom_price' not in data.columns:
                    data['w_bottom_price'] = np.nan
                if 'w_bottom_gap' not in data.columns:
                    data['w_bottom_gap'] = np.nan
                data.loc[data.index[signal_idx], 'w_bottom_price'] = w_bottom_price
                data.loc[data.index[signal_idx], 'w_bottom_gap'] = gap
                
                # 获取日期信息
                if 'date' in data.columns:
                    date1 = pd.to_datetime(data['date'].iloc[idx1]).strftime('%Y-%m-%d')
                    date2 = pd.to_datetime(data['date'].iloc[idx2]).strftime('%Y-%m-%d')
                    signal_date = pd.to_datetime(data['date'].iloc[signal_idx]).strftime('%Y-%m-%d')
                else:
                    date1 = str(idx1)
                    date2 = str(idx2)
                    signal_date = str(signal_idx)
                
                logger.info(f"[W底] 检测到W底形态: ({date1}, {date2}), "
                           f"低点价格=({price1:.2f}, {price2:.2f}), 止损基准={w_bottom_price:.2f}, "
                           f"间隔={gap}天, 振幅={amplitude:.1f}%, "
                           f"{signal_date}收阳线确认并买入(涨幅{body_strength:.2f}%), "
                           f"signal_idx={signal_idx}, w_bottom_signals[{signal_idx}]={w_bottom_signals.iloc[signal_idx]}")
                
                # 找到第一个有效的W底后，这个低点1就不再作为起点
                break

        return w_bottom_signals

    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """回测（重写父类方法，增加高抛低吸交易合并逻辑）"""
        # 调用父类的回测方法
        result = super().backtest(df, initial_capital)

        # 合并高抛低吸交易（将持仓周期内所有做T算作一笔交易）
        # 支持多次做T链：entry→sell1→rebuy1→sell2→rebuy2→...→final_exit 合并为一笔
        if 'swing_exit_type' in df.columns:
            trades = result.get('trades', [])
            if len(trades) > 1:
                swing_sell_dates = set(str(d)[:10] for d in df[df['swing_exit_type'] == 1]['date'].astype(str).tolist())
                swing_rebuy_dates = set(str(d)[:10] for d in df[df['swing_exit_type'] == 2]['date'].astype(str).tolist())

                if swing_sell_dates or swing_rebuy_dates:
                    merged_trades = []
                    i = 0
                    while i < len(trades):
                        trade = trades[i]
                        sell_date_str = str(trade.get('sell_date', ''))[:10]

                        # 检查是否是做T卖出（高抛）
                        if sell_date_str in swing_sell_dates:
                            # 开始合并链：追踪原始入场，累计做T利润，直到最终退出
                            original_buy_price = trade['buy_price']
                            original_buy_date = trade['buy_date']
                            total_commission = trade.get('commission', 0)
                            # 做T利润累计：每次卖出-买回的价差
                            swing_profit_sum = 0.0
                            last_swing_sell_price = trade['sell_price']
                            j = i + 1

                            # 沿着做T链向前走：rebuy→再sell→rebuy→...→final_exit
                            while j < len(trades):
                                next_trade = trades[j]
                                next_buy_str = str(next_trade.get('buy_date', ''))[:10]

                                if next_buy_str in swing_rebuy_dates:
                                    # 这是一次做T回买
                                    rebuy_price = next_trade['buy_price']
                                    # 做T利润 = 高抛价 - 低吸价（相对于原始入场价的比率）
                                    if original_buy_price > 0:
                                        swing_profit_sum += (last_swing_sell_price - rebuy_price) / original_buy_price
                                    total_commission += next_trade.get('commission', 0)

                                    next_sell_str = str(next_trade.get('sell_date', ''))[:10]
                                    if next_sell_str in swing_sell_dates:
                                        # 又一次做T卖出，继续链
                                        last_swing_sell_price = next_trade['sell_price']
                                        j += 1
                                        continue
                                    else:
                                        # 最终退出：链结束
                                        final_sell_price = next_trade['sell_price']
                                        final_sell_date = next_trade['sell_date']
                                        final_capital = next_trade.get('capital', 0)

                                        if original_buy_price > 0:
                                            # 总利润 = 最终退出利润 + 所有做T累计利润
                                            base_profit_rate = (final_sell_price / original_buy_price - 1)
                                            commission_rate = total_commission / (original_buy_price * 100)
                                            merged_profit_rate = base_profit_rate + swing_profit_sum - commission_rate
                                        else:
                                            merged_profit_rate = 0

                                        merged_trade = {
                                            'buy_date': original_buy_date,
                                            'buy_price': original_buy_price,
                                            'sell_date': final_sell_date,
                                            'sell_price': final_sell_price,
                                            'profit_rate': merged_profit_rate,
                                            'capital': final_capital,
                                            'commission': total_commission,
                                            'is_swing_merged': True,
                                            'swing_count': j - i,  # 做T次数
                                        }
                                        merged_trades.append(merged_trade)
                                        i = j + 1
                                        break
                                else:
                                    # 非做T回买（可能是做T放弃后的新交易）
                                    # 终止链：用最后一次做T卖出作为退出
                                    if original_buy_price > 0:
                                        base_profit_rate = (last_swing_sell_price / original_buy_price - 1)
                                        commission_rate = total_commission / (original_buy_price * 100)
                                        merged_profit_rate = base_profit_rate - commission_rate
                                    else:
                                        merged_profit_rate = 0

                                    merged_trade = {
                                        'buy_date': original_buy_date,
                                        'buy_price': original_buy_price,
                                        'sell_date': trade['sell_date'],
                                        'sell_price': last_swing_sell_price,
                                        'profit_rate': merged_profit_rate,
                                        'capital': trade.get('capital', 0),
                                        'commission': total_commission,
                                        'is_swing_merged': True,
                                        'swing_count': j - i - 1,
                                    }
                                    merged_trades.append(merged_trade)
                                    i = j  # 从下一笔非做T交易开始
                                    break
                            else:
                                # 做T卖出后没有更多交易（数据结束）
                                merged_trades.append(trade)
                                i = j
                            continue

                        merged_trades.append(trade)
                        i += 1

                    # 更新trades和相关统计
                    result['trades'] = merged_trades
                    if len(merged_trades) > 0:
                        winning = sum(1 for t in merged_trades if t['profit_rate'] > 0)
                        result['win_rate'] = winning / len(merged_trades) * 100

        # 计算盈亏比 (Profit Factor) - 用收益率之和
        trades = result.get('trades', [])
        win_trades = [t for t in trades if t['profit_rate'] > 0]
        lose_trades = [t for t in trades if t['profit_rate'] < 0]
        total_profit_pct = sum(t['profit_rate'] for t in win_trades)
        total_loss_pct = abs(sum(t['profit_rate'] for t in lose_trades))
        result['profit_factor'] = total_profit_pct / total_loss_pct if total_loss_pct > 0 else (999.0 if total_profit_pct > 0 else 0.0)
        # 保存汇总数据，供批量回测使用
        result['win_count'] = len(win_trades)
        result['lose_count'] = len(lose_trades)
        result['total_profit_pct'] = total_profit_pct
        result['total_loss_pct'] = total_loss_pct

        return result
