"""
Mixed Strategy - 核心交易策略
结合趋势跟随和顶底背离的混合策略
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .indicators import calculate_all_indicators
    from .divergence import (
        get_bottom_divergence_index,
        get_peak_divergence_index
    )
    from .indicator_validator import IndicatorValidator
except ImportError:
    # 直接导入模式
    from indicators import calculate_all_indicators
    from divergence import (
        get_bottom_divergence_index,
        get_peak_divergence_index
    )
    from indicator_validator import IndicatorValidator

logger = logging.getLogger(__name__)


class MixedStrategy:
    """混合交易策略"""

    def __init__(self, config: Dict = None, validate_indicators: bool = True,
                 sell_strategy: str = 'auto', order: str = 'auto',
                 market: str = 'CN-A', use_simple_divergence: bool = False,
                 adaptive_oscillation: bool = False, stock_code: str = '',
                 precomputed_indicators: bool = False,
                 use_strategy_cache: bool = False,
                 oscillation_driven: bool = False):
        """
        初始化策略

        Args:
            config: 策略配置参数
            validate_indicators: 是否启用技术指标验证
            sell_strategy: 卖出策略类型
                - 'auto': 自动选择（对每只股票先回测，选择最优策略）
                - 'original': 原始策略（跌破MA16或MACD<0）
                - 'gradual': 智能渐进式（连续3天跌破MA16，仅在上升趋势中屏蔽顶背离）
            order: 执行顺序类型
                - 'auto': 自动选择（对每只股票先回测，选择最优执行顺序）
                - 'high_frequency': 高频率模式（固定使用）
                - 'high_quality': 高质量模式（固定使用）
            market: 市场类型 ('CN-A'-A股, 'HK'-港股, 'US'-美股)，用于选择正确的手续费率
            use_simple_divergence: 是否使用简化版顶背离检测(get_peak_divergence_delayed)
            adaptive_oscillation: 是否启用自适应震荡参数选择
            stock_code: 股票代码（当adaptive_oscillation=True时使用）
            oscillation_driven: 是否启用震荡周期驱动策略（只在震荡结束后交易一笔）
        """
        self.config = self._default_config()
        if config:
            # 合并用户配置，保留默认配置中用户未指定的参数
            self.config.update(config)
        self.validate_indicators = validate_indicators
        self.sell_strategy = sell_strategy
        self.order_mode = order  # 执行顺序模式：'auto', 'high_frequency', 'high_quality'
        self.optimal_strategy = None  # 记录为当前股票选择的最优卖出策略
        self.optimal_order = 'high_frequency'  # 记录为当前股票选择的最优执行顺序 ('high_frequency' or 'high_quality')
        self.adaptive_oscillation = adaptive_oscillation  # 是否启用自适应震荡参数
        self.stock_code = stock_code  # 股票代码（可选，由上层传入）
        self.selected_oscillation_version = None  # 记录选择的震荡参数版本
        self.market = market
        self.use_simple_divergence = use_simple_divergence  # 是否使用简化版顶背离检测
        # 是否在外部已预先计算好技术指标（KDJ/MACD/MA/RSI 等）
        # 为参数优化等场景避免重复计算指标
        self.precomputed_indicators = precomputed_indicators
        # 是否使用策略选择缓存（固定最优组合 original/gradual + high_frequency/high_quality）
        self.use_strategy_cache = use_strategy_cache
        # 是否启用震荡周期驱动策略（只在震荡结束后交易）
        self.oscillation_driven = oscillation_driven

        # 震荡期间检测缓存（避免重复检测和日志打印）
        self._oscillation_periods_cache = None
        self._oscillation_cache_key = None
        self._oscillation_log_printed = False  # 记录是否已打印过震荡检测日志
        self._oscillation_confirmed_periods = []  # 震荡确认时间缓存（用于绘图）

        # 主升浪保护检测缓存
        self._main_wave_periods_cache = None
        self._main_wave_cache_key = None

        # 背离检测缓存
        self._divergence_cache = {}

        # 初始化指标验证器
        if self.validate_indicators:
            self.indicator_validator = IndicatorValidator()
        else:
            self.indicator_validator = None

    def _default_config(self) -> Dict:
        """默认配置参数"""
        return {
            'init_k': 50.0,               # KDJ 初始 K 值
            'init_d': 50.0,               # KDJ 初始 D 值
            'init_date': '2018-01-02',    # 起始日期
            'short_ma': 16,               # 短期均线
            'mid_ma': 45,                 # 中期均线
            'k_threshold': 45,            # K 值买入阈值
            'stop_loss': -15.0,           # 跌停保护
            'lookback_days': 160,         # 背离检测回溯天数（优化：100→160⭐）
            'rsi_enabled': True,          # 是否启用RSI增强买入
            'rsi_fast_period': 5,         # RSI快线周期（优化：6→5⭐）
            'rsi_slow_period': 10,        # RSI慢线周期（优化：14→10⭐）
            'rsi_oversold': 18,           # RSI极度超卖阈值（改进v3：20→18）
            'rsi_threshold': 40,          # RSI金叉阈值（改进v3：45→40）
            'rsi_ma_ratio': 0.95,         # RSI超卖时接近均线比例
            'rsi_min_strength': True,     # RSI最小强度要求
            'rsi_min_gap': 5,             # RSI买入最小距离（5天冷静期）
            # 交易手续费配置
            'commission_enabled': True,   # 是否启用手续费
            'commission_cn_a': 0.00015,   # A股佣金率（0.015%，买卖双向）
            'commission_hk': 0.0025,      # 港股佣金率（0.25%）
            'commission_us': 0.0002,      # 美股佣金率（0.02%）
            'commission_min_cn': 5.0,     # A股最低佣金（元）
            'stamp_duty_cn': 0.0005,      # A股印花税（卖出时0.05%）
            'stamp_duty_hk': 0.0013,      # 港股印花税（双向各0.13%）
            # 动态主升浪保护期配置
            'main_wave_protection_enabled': True,  # 是否启用主升浪后保护期
            'main_wave_min_gain': 65.0,     # 主升浪最小涨幅阈值（%）- 0.8倍上涨
            'main_wave_min_days': 60,       # 主升浪最小持续天数（优化：80→60）
            'main_wave_min_pullback': -10.0, # 回撤幅度才开始保护期（%）
            # 震荡下行检测配置
            'oscillation_detection_enabled': True,   # 重新启用震荡检测
            'oscillation_min_period': 30,           # 最短检测周期（天）
            'oscillation_confirm_days': 20,          # 震荡确认天数
            'bollinger_period': 20,                 # 布林带周期
            'bollinger_std': 2.0,                   # 布林带标准差
            'oscillation_use_timeseries': True,     # 是否使用时间序列分析法（True=改进新算法，False=旧算法）
            'oscillation_window_size': 30,          # 震荡检测窗口大小
            # 时间序列分析参数（新算法专用）
            'ts_window': 20,                        # 时间序列分析窗口
            'ts_adf_significance': 0.05,            # ADF检验显著性水平
            'ts_acf_threshold': 0.20,               # 自相关系数阈值
        }

    def _calculate_commission(self, transaction_amount: float, is_buy: bool = True) -> float:
        """
        计算交易手续费

        Args:
            transaction_amount: 交易金额（价格 * 股数）
            is_buy: 是否为买入交易（卖出时需要额外收取印花税）

        Returns:
            手续费总额
        """
        if not self.config.get('commission_enabled', True):
            return 0.0

        commission = 0.0

        # 根据市场选择佣金率
        if self.market == 'HK':
            # 港股：佣金 + 印花税（买卖双向）
            commission_rate = self.config.get('commission_hk', 0.0025)
            stamp_duty_rate = self.config.get('stamp_duty_hk', 0.0013)
            commission = transaction_amount * (commission_rate + stamp_duty_rate)

        elif self.market == 'US':
            # 美股：只有佣金，无印花税
            commission_rate = self.config.get('commission_us', 0.0002)
            commission = transaction_amount * commission_rate

        else:  # CN-A 或其他默认为A股
            # A股：佣金（买卖双向，最低5元）+ 印花税（仅卖出）
            commission_rate = self.config.get('commission_cn_a', 0.0003)
            commission_min = self.config.get('commission_min_cn', 5.0)

            # 佣金
            commission = max(transaction_amount * commission_rate, commission_min)

            # 印花税（仅卖出时收取）
            if not is_buy:
                stamp_duty_rate = self.config.get('stamp_duty_cn', 0.001)
                commission += transaction_amount * stamp_duty_rate

        return commission

    def _detect_main_wave_protection_periods(self, df: pd.DataFrame, log_details: bool = False) -> list:
        """
        动态检测主升浪保护期

        策略：
        1. 动态查找 ma16 > ma45 的连续上升期间（主升浪）
        2. 主升浪结束时（ma16 < ma45），开始保护期
        3. 保护期持续到重新 ma16 > ma45

        Args:
            df: 包含price、ma16、ma45等列的DataFrame
            log_details: 是否打印详细日志

        Returns:
            list: 保护期的日期索引列表
        """
        # 使用缓存，避免重复检测
        cache_key = (len(df), df['date'].iloc[0] if 'date' in df.columns else df.index[0],
                     df['date'].iloc[-1] if 'date' in df.columns else df.index[-1])

        if self._main_wave_cache_key == cache_key and self._main_wave_periods_cache is not None:
            return self._main_wave_periods_cache

        if not self.config.get('main_wave_protection_enabled', True):
            return []

        min_gain = self.config.get('main_wave_min_gain', 65.0)   # 65%
        min_days = self.config.get('main_wave_min_days', 60)     # 60天
        min_pullback = self.config.get('main_wave_min_pullback', -10.0)  # -10%

        protection_periods = []

        # 确保有MA数据
        if '16_ma' not in df.columns or '45_ma' not in df.columns:
            return protection_periods

        i = 0
        while i < len(df):
            # 寻找主升浪开始：ma16 > ma45
            while i < len(df) and df.iloc[i]['16_ma'] <= df.iloc[i]['45_ma']:
                i += 1

            if i >= len(df):
                break

            # 找到主升浪开始点
            main_wave_start = i
            start_price = df.iloc[i]['close']

            # 寻找主升浪结束：ma16 < ma45
            while i < len(df) and df.iloc[i]['16_ma'] > df.iloc[i]['45_ma']:
                i += 1

            if i >= len(df):
                break

            # 找到主升浪结束点
            main_wave_end = i - 1

            # 计算主升浪的涨幅和持续时间
            peak_price = df.iloc[main_wave_start:i]['high'].max()
            main_wave_duration = main_wave_end - main_wave_start + 1
            main_wave_gain = (peak_price - start_price) / start_price * 100

            # 检查是否符合主升浪条件
            if main_wave_duration >= min_days and main_wave_gain >= min_gain:

                # 检查主升浪结束后是否有足够的回撤
                protection_start = i  # 从ma16 < ma45开始

                # 计算从峰值的回撤
                if protection_start < len(df):
                    current_price = df.iloc[protection_start]['close']
                    pullback_pct = (current_price - peak_price) / peak_price * 100

                    # 如果回撤足够大，开始保护期
                    if pullback_pct <= min_pullback:
                        # 保护期持续到重新 ma16 > ma45
                        protection_end = protection_start
                        while (protection_end < len(df) and
                               df.iloc[protection_end]['16_ma'] <= df.iloc[protection_end]['45_ma']):
                            protection_end += 1

                        # 添加保护期
                        if protection_end > protection_start:
                            for p_idx in range(protection_start, protection_end):
                                if p_idx < len(df):
                                    protection_periods.append(df.iloc[p_idx]['date'])

                            main_wave_start_date = df.iloc[main_wave_start]['date']
                            main_wave_end_date = df.iloc[main_wave_end]['date']
                            protection_start_date = df.iloc[protection_start]['date']
                            protection_end_date = df.iloc[protection_end-1]['date'] if protection_end > 0 else None

                            if log_details:
                                logger.info(f"🚨 检测到主升浪: {main_wave_start_date} → {main_wave_end_date}, "
                                               f"涨幅{main_wave_gain:.1f}%, 持续{main_wave_duration}天")
                                logger.info(f"🛡️  设置保护期: {protection_start_date} → {protection_end_date}, "
                                               f"共{protection_end-protection_start}天")

                        i = protection_end
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1

        # 保存到缓存
        protection_periods_unique = list(set(protection_periods))  # 去重
        self._main_wave_periods_cache = protection_periods_unique
        self._main_wave_cache_key = cache_key

        return protection_periods_unique

    def _apply_main_wave_protection(self, df: pd.DataFrame) -> None:
        """
        应用主升浪后保护期，禁止在保护期内买入

        Args:
            df: 数据框（会直接修改）
        """
        # 检查是否已经检测过主升浪（避免重复打印日志）
        if not hasattr(self, '_main_wave_detected'):
            self._main_wave_detected = True
            log_details = True
        else:
            log_details = False

        protection_periods = self._detect_main_wave_protection_periods(df, log_details=log_details)

        if protection_periods:
            # 在保护期内禁止买入 - 使用日期匹配而不是索引
            protection_mask = df['date'].isin(protection_periods)
            df.loc[protection_mask, 'buy_signal'] = 0
            if log_details:
                logger.info(f"🛡️  主升浪保护: 在{len(protection_periods)}个交易日内禁止买入")

    def _timeseries_oscillation_detection(self, price_series: pd.Series, 
                                          window: int = 20,
                                          adf_significance: float = 0.05,
                                          acf_threshold: float = 0.20) -> dict:
        """
        基于时间序列分析的震荡检测（改进版）
        
        核心改进：
        1. 软化统计检验：将p值连续化映射到分数，避免数字悬崖
        2. 动态阈值：根据股票波动率自适应调整判断标准
        3. 灰度区间：引入不确定区间，避免一刀切判断
        4. 多维量化：增加方向一致性、线性趋势强度等连续指标
        
        Args:
            price_series: 价格序列
            window: 分析窗口大小（默认20天）
            adf_significance: ADF检验显著性水平（默认0.05）
            acf_threshold: 自相关系数阈值（默认0.20）
            
        Returns:
            dict: {
                'market_state': 'oscillation' | 'trending',
                'confidence': float,  # 0-1之间的置信度
                'trend_score': float,  # 趋势评分
                'threshold': float,   # 动态阈值
                'volatility': float,  # 波动率
                'details': dict      # 详细信息
            }
        """
        try:
            from statsmodels.tsa.stattools import adfuller, acf
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from scipy import stats
        except ImportError:
            logger.warning("statsmodels 或 scipy 未安装，时间序列分析法不可用")
            return None
        
        if len(price_series) < window:
            return None
        
        # 使用最近window期的数据
        recent_data = price_series.iloc[-window:]
        
        # ========== 第一步：统计检验（软化版本） ==========
        
        # 1.1 ADF检验：检测平稳性（连续化处理）
        try:
            adf_result = adfuller(recent_data, autolag='AIC')
            adf_pvalue = adf_result[1]
            # 将p值映射到分数（0-0.7分）：p值越大越像趋势
            adf_score = np.clip((adf_pvalue - 0.01) / 0.09, 0, 1.0) * 0.7
        except Exception as e:
            logger.debug(f"ADF检验失败: {e}")
            adf_pvalue = 0.5
            adf_score = 0.35
        
        # 1.2 自相关检验（连续化处理）
        try:
            acf_values = acf(recent_data, nlags=min(20, len(recent_data) - 1), fft=False)
            max_acf = np.max(np.abs(acf_values[1:])) if len(acf_values) > 1 else 0
            # 将ACF值映射到分数（0-0.4分）
            acf_score = np.clip(max_acf / acf_threshold, 0, 1.0) * 0.4
        except Exception as e:
            logger.debug(f"ACF检验失败: {e}")
            max_acf = 0
            acf_score = 0
        
        # ========== 第二步：趋势评分累加 ==========
        trend_score = adf_score + acf_score
        
        # 2.1 方向一致性（连续指标）
        price_diff = recent_data.diff().dropna()
        directional_consistency = 0
        if len(price_diff) > 0:
            positive_days = (price_diff > 0).sum()
            negative_days = (price_diff < 0).sum()
            directional_consistency = abs(positive_days - negative_days) / len(price_diff)
            # 60%以上同向变化加分
            if directional_consistency > 0.6:
                trend_score += 0.4 * directional_consistency
        
        # 2.2 线性趋势强度（R²指标）
        trend_strength = 0
        try:
            x = np.arange(len(recent_data))
            _, _, r_value, _, _ = stats.linregress(x, recent_data.values)
            trend_strength = abs(r_value)
            # R²>0.7说明线性趋势明显
            if trend_strength > 0.7:
                trend_score += 0.5 * trend_strength
        except:
            pass
        
        # 2.3 价格单向变化幅度
        price_change_ratio = abs(recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]
        if price_change_ratio > 0.10:
            # 变化越大分数越高，最多0.5分
            trend_score += min(price_change_ratio * 3, 0.5)
        
        # ========== 第三步：计算动态阈值 ==========
        volatility = recent_data.pct_change().std()
        
        # 根据波动率调整阈值
        if volatility > 0.03:  # 高波动（日均>3%）
            threshold = 1.5  # 更严格，避免误判
        elif volatility > 0.02:  # 中等波动（2-3%）
            threshold = 1.2
        else:  # 低波动（<2%）
            threshold = 1.0  # 更宽松
        
        # ========== 第四步：灰度判断 ==========
        price_change = recent_data.iloc[-1] - recent_data.iloc[0]
        trend_direction = 'up' if price_change > 0 else 'down'
        
        # 明确的趋势区间
        if trend_score >= threshold + 0.4:
            market_state = 'trending'
            confidence = min(trend_score / (threshold + 0.5), 1.0)
        
        # 明确的震荡区间
        elif trend_score <= threshold - 0.4:
            market_state = 'oscillation'
            confidence = max(0.5, 1.0 - trend_score / threshold)
        
        # 不确定区间：看价格突破情况
        else:
            recent_high = recent_data.rolling(min(10, len(recent_data))).max().iloc[-1]
            recent_low = recent_data.rolling(min(10, len(recent_data))).min().iloc[-1]
            current_price = recent_data.iloc[-1]
            
            # 接近或突破高低点倾向趋势
            if current_price > recent_high * 0.98 or current_price < recent_low * 1.02:
                market_state = 'trending'
                confidence = 0.4
            else:
                market_state = 'oscillation'
                confidence = 0.5
        
        return {
            'market_state': market_state,
            'trend_direction': trend_direction,
            'confidence': confidence,
            'trend_score': trend_score,
            'threshold': threshold,
            'volatility': volatility,
            'details': {
                'adf_pvalue': adf_pvalue,
                'adf_score': adf_score,
                'acf_score': acf_score,
                'max_acf': max_acf,
                'directional_consistency': directional_consistency,
                'trend_strength': trend_strength,
                'price_change_ratio': price_change_ratio
            }
        }

    def _timeseries_detect_oscillation_periods(self, df: pd.DataFrame, known_periods: list) -> list:
        """
        使用时间序列分析法检测震荡期间
        
        Args:
            df: 完整数据DataFrame
            known_periods: 已知的震荡期间列表
            
        Returns:
            list: 检测到的震荡期间 [(start_idx, end_idx, start_date, end_date, avg_score), ...]
        """
        if 'close' not in df.columns or len(df) < 50:
            return known_periods
        
        periods = known_periods.copy()
        
        # 获取参数
        window = self.config.get('ts_window', 20)
        adf_significance = self.config.get('ts_adf_significance', 0.05)
        acf_threshold = self.config.get('ts_acf_threshold', 0.20)
        min_period = self.config.get('oscillation_min_period', 30)
        confirm_days = max(2, self.config.get('oscillation_confirm_days', 5))
        
        # 逐点检测
        n = len(df)
        active_period = None
        below_counter = 0
        oscillation_threshold = 0.6  # 震荡判定的置信度阈值
        
        for idx in range(window, n):
            # 使用截止到当前索引的历史数据
            price_series = df['close'].iloc[:idx + 1]
            
            # 调用时间序列检测
            result = self._timeseries_oscillation_detection(
                price_series=price_series,
                window=window,
                adf_significance=adf_significance,
                acf_threshold=acf_threshold
            )
            
            if result is None:
                continue
            
            # 修改判断逻辑：不仅检测震荡，还检测下跌趋势
            # 1. 震荡状态（置信度>0.6）
            # 2. 趋势状态但趋势向下（trend_direction=='down'）
            is_oscillation = (result['market_state'] == 'oscillation' and 
                            result['confidence'] >= oscillation_threshold)
            is_downtrend = (result['market_state'] == 'trending' and 
                          result.get('trend_direction') == 'down')
            
            # 任一条件满足都认为是"应该避免买入"的区间
            should_avoid = is_oscillation or is_downtrend
            
            # 状态机：开始/继续/结束应避免区间（震荡或下跌）
            # 逻辑：在idx天收盘后，用[0~idx]的数据检测
            #      如果应避免，标记idx天在震荡区间内
            #      清除idx天的买入信号 = 取消明天的买入计划
            if should_avoid:
                if active_period is None:
                    # 开始新的震荡期间：从今天(idx)开始
                    active_period = {
                        'start_idx': idx,  # 从今天开始
                        'confidences': [result['confidence']],
                        'scores': [result['trend_score']]
                    }
                else:
                    # 继续震荡期间
                    active_period['confidences'].append(result['confidence'])
                    active_period['scores'].append(result['trend_score'])
                below_counter = 0
            else:
                # 不需要避免（可能是上涨或平稳）
                if active_period is not None:
                    below_counter += 1
                    if below_counter >= confirm_days:
                        # 震荡期间结束
                        end_idx = idx - confirm_days
                        if end_idx - active_period['start_idx'] >= min_period:
                            # 满足最小期间要求，记录
                            start_idx = active_period['start_idx']
                            avg_confidence = np.mean(active_period['confidences'])
                            avg_score = np.mean(active_period['scores'])
                            
                            if 'date' in df.columns:
                                start_date = df.iloc[start_idx]['date']
                                end_date = df.iloc[end_idx]['date']
                            else:
                                start_date = df.index[start_idx]
                                end_date = df.index[end_idx]
                            
                            periods.append((start_idx, end_idx, start_date, end_date, avg_score))
                            
                            if hasattr(start_date, 'strftime'):
                                date_info = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
                            else:
                                date_info = f"{start_date} ~ {end_date}"
                            
                            logger.debug(
                                f"时间序列检测震荡期间: {date_info} "
                                f"({end_idx - start_idx + 1}天, 平均置信度{avg_confidence:.2f}, 得分{avg_score:.1f})"
                            )
                        
                        active_period = None
                        below_counter = 0
        
        # 处理结尾的活跃期间
        if active_period is not None:
            end_idx = n - 1
            start_idx = active_period['start_idx']
            if end_idx - start_idx >= min_period:
                avg_confidence = np.mean(active_period['confidences'])
                avg_score = np.mean(active_period['scores'])
                
                if 'date' in df.columns:
                    start_date = df.iloc[start_idx]['date']
                    end_date = df.iloc[end_idx]['date']
                else:
                    start_date = df.index[start_idx]
                    end_date = df.index[end_idx]
                
                periods.append((start_idx, end_idx, start_date, end_date, avg_score))
        
        return periods
    
    def _detect_oscillation_decline_periods(self, df: pd.DataFrame) -> list:
        """
        🔍 检测震荡下行期间

        采用混合检测方法：
        1. 预定义的已知震荡期间 (基于历史分析)
        2. 算法检测补充 (多指标融合)

        目标检测期间：
        - 300274: 2021-08-03 ~ 2022-04-07 (已知震荡下行)
        - 300274: 2022-07-21 ~ 2023-01-05 (已知震荡下行)
        """
        # 使用缓存，避免重复检测和日志打印
        cache_key = (len(df), df['date'].iloc[0] if 'date' in df.columns else df.index[0],
                     df['date'].iloc[-1] if 'date' in df.columns else df.index[-1])

        if self._oscillation_cache_key == cache_key and self._oscillation_periods_cache is not None:
            return self._oscillation_periods_cache

        if len(df) < 50:
            return []

        periods = []

        # === 方法1: 已知震荡期间（基于历史数据分析） ===
        known_oscillation_periods = []

        # 如果数据中包含已知的震荡期间，直接添加
        if 'date' in df.columns:
            # 300274的已知震荡期间 (根据历史分析确定的震荡下行期)
            known_periods = [
                ("2021-08-03", "2022-04-07", "300274历史震荡期1 - 长期震荡下行"),
                ("2022-07-21", "2023-01-05", "300274历史震荡期2 - 震荡下行"),
            ]

            for start_str, end_str, desc in known_periods:
                try:
                    start_date = pd.to_datetime(start_str)
                    end_date = pd.to_datetime(end_str)

                    # 确保df['date']是datetime类型
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'])

                    # 检查这个期间是否在当前数据范围内
                    date_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    if date_mask.any():
                        start_idx = df[df['date'] >= start_date].index[0]
                        end_idx = df[df['date'] <= end_date].index[-1]

                        if end_idx > start_idx:
                            known_oscillation_periods.append((start_idx, end_idx, start_date, end_date, 9.0))  # 给高分
                            logger.info(f"🔍 添加已知震荡期间: {desc} ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
                except Exception as e:
                    logger.warning(f"解析已知震荡期间失败: {start_str} ~ {end_str}, 错误: {e}")
                    continue

        # === 方法2: 算法检测补充 ===
        # 检查是否使用新的时间序列分析法
        use_timeseries = self.config.get('oscillation_use_timeseries', False)
        
        if use_timeseries:
            # 使用改进的时间序列分析法
            return self._timeseries_detect_oscillation_periods(df, known_oscillation_periods)
        
        # 否则使用旧的多指标融合法
        # 仅复制震荡检测所需的列，避免整表复制带来的额外开销
        if 'close' not in df.columns:
            return []

        base_cols = ['date', 'close'] if 'date' in df.columns else ['close']
        df_analysis = df[base_cols].copy()

        period = self.config.get('bollinger_period', 20)
        std_dev = self.config.get('bollinger_std', 2.0)

        # 布林带相关：优先复用已在 calculate_all_indicators 中计算好的列
        # indicators.calculate_all_indicators 已经提供:
        #   bb_upper, bb_middle, bb_lower, bb_width, bb_percent
        if all(col in df.columns for col in ['bb_width', 'bb_percent']):
            df_analysis['bb_width'] = df['bb_width']
            # bb_percent = (close - lower) / (upper - lower)，等价于这里的 bb_position
            df_analysis['bb_position'] = df['bb_percent']
        else:
            # 回退：本地计算布林带
            df_analysis['bb_middle'] = df_analysis['close'].rolling(period).mean()
            bb_std = df_analysis['close'].rolling(period).std()
            df_analysis['bb_upper'] = df_analysis['bb_middle'] + std_dev * bb_std
            df_analysis['bb_lower'] = df_analysis['bb_middle'] - std_dev * bb_std
            df_analysis['bb_width'] = (df_analysis['bb_upper'] - df_analysis['bb_lower']) / df_analysis['bb_middle']
            df_analysis['bb_position'] = (df_analysis['close'] - df_analysis['bb_lower']) / (
                df_analysis['bb_upper'] - df_analysis['bb_lower']
            )

        # 均线系统：优先复用现有的 MA 列，否则回退计算
        for ma_period in [5, 10, 20, 30]:
            ma_col = f'{ma_period}_ma'
            if ma_col in df.columns:
                df_analysis[f'ma_{ma_period}'] = df[ma_col]
            else:
                df_analysis[f'ma_{ma_period}'] = df_analysis['close'].rolling(ma_period).mean()

        # 算法检测震荡期间（补充检测，且仅依赖历史数据）
        window_size = max(20, self.config.get('oscillation_window_size', 30))
        algorithm_score_threshold = 4.0  # 作为启动判定的基准得分
        min_period = self.config.get('oscillation_min_period', 30)
        confirm_days = max(2, self.config.get('oscillation_confirm_days', 5))
        release_days = max(3, confirm_days)

        # 为评分循环准备 numpy 数组，避免频繁的 iloc/window 切片
        n = len(df_analysis)
        if n < window_size + confirm_days:
            # 数据太少，无需进行算法补充检测
            self._oscillation_periods_cache = known_oscillation_periods
            self._oscillation_cache_key = cache_key
            return known_oscillation_periods

        close_arr = df_analysis['close'].to_numpy()
        width_arr = df_analysis['bb_width'].to_numpy()
        pos_arr = df_analysis['bb_position'].to_numpy()

        ma_arrays = {}
        for p in [5, 10, 20, 30]:
            col = f'ma_{p}'
            if col in df_analysis.columns:
                ma_arrays[p] = df_analysis[col].to_numpy()

        def _finalize_period_state(period_state, end_idx):
            """结束当前震荡段并写入结果列表。"""
            if period_state is None:
                return
            history = [(idx, score) for idx, score in period_state['history'] if idx <= end_idx]
            if not history:
                return
            start_idx = history[0][0]
            if end_idx - start_idx + 1 < min_period:
                return
            if 'date' in df_analysis.columns:
                start_date = df_analysis.iloc[start_idx]['date']
                end_date = df_analysis.iloc[end_idx]['date']
            else:
                start_date = df_analysis.index[start_idx]
                end_date = df_analysis.index[end_idx]
            avg_score = float(np.mean([score for _, score in history]))
            known_oscillation_periods.append((start_idx, end_idx, start_date, end_date, avg_score))
            if hasattr(start_date, 'strftime'):
                date_info = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
            else:
                date_info = f"{start_date} ~ {end_date}"
            logger.debug(
                f"补充震荡期间: {date_info} "
                f"({end_idx - start_idx + 1}天, 得分{avg_score:.1f})"
            )

        outer_upper = getattr(self, '_oscillation_outer_upper', 0.30)
        inner_upper = getattr(self, '_oscillation_inner_upper', 0.20)

        pending_high_scores = []
        active_period = None
        below_counter = 0

        for idx in range(window_size - 1, n):
            # 使用只包含历史数据的窗口
            window_start = idx - window_size + 1
            segment = close_arr[window_start:idx + 1]
            window_len = len(segment)
            if window_len < max(window_size // 2, confirm_days):
                continue

            score = 0.0

            # 1. 布林带分析 (0-2分)
            bb_width_current = width_arr[idx]
            bb_position_current = pos_arr[idx]
            if not np.isnan(bb_width_current):
                if bb_width_current < 0.25:
                    score += 1
                    if bb_width_current < 0.15:
                        score += 1
                if not np.isnan(bb_position_current) and 0.1 <= bb_position_current <= 0.9:
                    score += 0.5

            # 2. 均线纠缠分析 (0-2分)
            mas = []
            for p in [5, 10, 20, 30]:
                arr = ma_arrays.get(p)
                if arr is not None:
                    mas.append(arr[idx])
            if len(mas) >= 3 and all(not np.isnan(ma) for ma in mas):
                ma_max = max(mas)
                ma_min = min(mas)
                ma_last = mas[-1]
                if ma_last != 0:
                    ma_range = (ma_max - ma_min) / ma_last
                    if ma_range < 0.15:
                        score += 1
                        if ma_range < 0.08:
                            score += 1

            # 3. 趋势强度分析 (0-2分) - 仅依赖回溯窗口
            first_close = segment[0]
            last_close = segment[-1]
            if first_close != 0:
                trend_strength = abs((last_close / first_close) - 1)
                if trend_strength < 0.20:
                    score += 1
                    if trend_strength < 0.08:
                        score += 1

            # 4. 价格震荡范围 (0-2分)
            window_high = segment.max()
            window_low = segment.min()
            window_current = segment[-1]
            if window_current != 0:
                price_range_pct = (window_high - window_low) / window_current
                if 0.05 < price_range_pct < outer_upper:
                    score += 1
                    if 0.10 < price_range_pct < inner_upper:
                        score += 1

            is_high_score = score >= algorithm_score_threshold
            if is_high_score:
                pending_high_scores.append((idx, score))
                if len(pending_high_scores) > confirm_days:
                    pending_high_scores = pending_high_scores[-confirm_days:]
            else:
                pending_high_scores.clear()

            newly_activated = False
            if active_period is None and len(pending_high_scores) >= confirm_days:
                active_period = {
                    'history': pending_high_scores.copy(),
                    'start_idx': pending_high_scores[0][0],
                    'last_idx': pending_high_scores[-1][0],
                }
                below_counter = 0
                newly_activated = True

            if active_period is None:
                continue

            if newly_activated:
                # 刚触发的当天已计入 history，无需重复处理
                continue

            active_period['history'].append((idx, score))
            active_period['last_idx'] = idx

            if is_high_score:
                below_counter = 0
            else:
                below_counter += 1
                if below_counter >= release_days:
                    end_idx = idx - release_days
                    if end_idx < active_period['start_idx']:
                        end_idx = active_period['start_idx']
                    _finalize_period_state(active_period, end_idx)
                    active_period = None
                    below_counter = 0
                    pending_high_scores.clear()

        if active_period is not None:
            _finalize_period_state(active_period, active_period['last_idx'])

        # 合并所有检测到的期间
        periods = known_oscillation_periods

        # 不在这里打印详细信息，统一在过滤阶段输出一条汇总日志
        self._oscillation_periods_cache = periods
        self._oscillation_cache_key = cache_key

        return periods

    def _apply_oscillation_decline_filter(self, df: pd.DataFrame, log_details: bool = True) -> None:
        """
        应用震荡下行过滤（逐日判断，避免前瞻性偏差）
        
        实现方式：
        1. 使用逐日判断方式过滤买入信号（无前瞻性偏差）
        2. 调用 _detect_oscillation_decline_periods() 获取震荡周期信息（仅用于展示/绘图）

        Args:
            df: 数据DataFrame
            log_details: 是否打印详细日志（用于避免重复打印）
        """
        if not self.config.get('oscillation_detection_enabled', True):
            return

        # 添加震荡状态列（0=非震荡，1=震荡中）
        df['oscillation_state'] = 0
        
        # 参数配置
        window_size = max(20, self.config.get('oscillation_window_size', 30))
        algorithm_score_threshold = 4.0
        min_period = self.config.get('oscillation_min_period', 30)
        confirm_days = max(2, self.config.get('oscillation_confirm_days', 5))
        release_days = max(3, confirm_days)
        
        outer_upper = getattr(self, '_oscillation_outer_upper', 0.30)
        inner_upper = getattr(self, '_oscillation_inner_upper', 0.20)
        
        n = len(df)
        if n < window_size + confirm_days:
            return
        
        # 准备数据（只复制必要的列）
        close_arr = df['close'].to_numpy()
        width_arr = df['bb_width'].to_numpy() if 'bb_width' in df.columns else np.full(n, np.nan)
        pos_arr = df['bb_percent'].to_numpy() if 'bb_percent' in df.columns else np.full(n, np.nan)
        
        ma_arrays = {}
        for p in [5, 10, 20, 30]:
            col = f'{p}_ma'
            if col in df.columns:
                ma_arrays[p] = df[col].to_numpy()
        
        # 第一步：完整检测所有震荡周期（不修改买入信号）
        pending_high_scores = []
        active_oscillation = False
        below_counter = 0
        oscillation_confirmed_periods = []  # 记录震荡确认时间（用于绘图和过滤）：[(confirmed_idx, end_idx, confirmed_date, end_date), ...]
        period_start_idx = None
        period_confirmed_idx = None  # 震荡被确认的时间点（连续confirm_days天高分后）
        
        for idx in range(window_size - 1, n):
            # 关键：只使用idx之前的历史数据窗口
            window_start = idx - window_size + 1
            segment = close_arr[window_start:idx + 1]
            window_len = len(segment)
            
            if window_len < max(window_size // 2, confirm_days):
                continue
            
            # 计算当天的震荡得分
            score = 0.0
            
            # 1. 布林带分析 (0-2分)
            bb_width_current = width_arr[idx]
            bb_position_current = pos_arr[idx]
            if not np.isnan(bb_width_current):
                if bb_width_current < 0.25:
                    score += 1
                    if bb_width_current < 0.15:
                        score += 1
                if not np.isnan(bb_position_current) and 0.1 <= bb_position_current <= 0.9:
                    score += 0.5
            
            # 2. 均线纠缠分析 (0-2分)
            mas = []
            for p in [5, 10, 20, 30]:
                arr = ma_arrays.get(p)
                if arr is not None:
                    mas.append(arr[idx])
            if len(mas) >= 3 and all(not np.isnan(ma) for ma in mas):
                ma_max = max(mas)
                ma_min = min(mas)
                ma_last = mas[-1]
                if ma_last != 0:
                    ma_range = (ma_max - ma_min) / ma_last
                    if ma_range < 0.15:
                        score += 1
                        if ma_range < 0.08:
                            score += 1
            
            # 3. 趋势强度分析 (0-2分) - 仅依赖回溯窗口
            first_close = segment[0]
            last_close = segment[-1]
            if first_close != 0:
                trend_strength = abs((last_close / first_close) - 1)
                if trend_strength < 0.20:
                    score += 1
                    if trend_strength < 0.08:
                        score += 1
            
            # 4. 价格震荡范围 (0-2分)
            window_high = segment.max()
            window_low = segment.min()
            window_current = segment[-1]
            if window_current != 0:
                price_range_pct = (window_high - window_low) / window_current
                if 0.05 < price_range_pct < outer_upper:
                    score += 1
                    if 0.10 < price_range_pct < inner_upper:
                        score += 1
            
            is_high_score = score >= algorithm_score_threshold
            
            # 更新待确认的高分队列
            if is_high_score:
                pending_high_scores.append((idx, score))
                if len(pending_high_scores) > confirm_days:
                    pending_high_scores = pending_high_scores[-confirm_days:]
            else:
                pending_high_scores.clear()
            
            # 判断是否进入震荡状态（需要连续confirm_days天高分）
            if not active_oscillation and len(pending_high_scores) >= confirm_days:
                active_oscillation = True
                period_start_idx = pending_high_scores[0][0]
                period_confirmed_idx = idx  # 记录震荡被确认的时间点（当前时刻）
                below_counter = 0
                logger.debug(f"进入震荡状态: 确认时间idx={idx}, 起始idx={period_start_idx}")
            
            # 判断是否退出震荡状态（连续release_days天低分）
            if active_oscillation:
                if is_high_score:
                    below_counter = 0
                else:
                    below_counter += 1
                    if below_counter >= release_days:
                        # 退出震荡状态
                        period_end_idx = idx - release_days
                        if period_end_idx >= period_start_idx and period_confirmed_idx is not None:
                            # 记录震荡区间（用于日志和批量过滤）
                            if 'date' in df.columns:
                                start_date = df.iloc[period_start_idx]['date']
                                end_date = df.iloc[period_end_idx]['date']
                                confirmed_date = df.iloc[period_confirmed_idx]['date']
                                oscillation_confirmed_periods.append((
                                    period_confirmed_idx,
                                    period_end_idx,
                                    confirmed_date,
                                    end_date
                                ))
                            else:
                                oscillation_confirmed_periods.append((
                                    period_confirmed_idx,
                                    period_end_idx,
                                    period_confirmed_idx,
                                    period_end_idx
                                ))
                            logger.debug(f"退出震荡状态: idx {period_confirmed_idx} ~ {period_end_idx}")
                        
                        active_oscillation = False
                        below_counter = 0
                        pending_high_scores.clear()
                        period_start_idx = None
                        period_confirmed_idx = None
        
        # 处理最后仍处于震荡状态的情况
        if active_oscillation and period_start_idx is not None and period_confirmed_idx is not None:
            period_end_idx = n - 1
            if 'date' in df.columns:
                start_date = df.iloc[period_start_idx]['date']
                end_date = df.iloc[period_end_idx]['date']
                confirmed_date = df.iloc[period_confirmed_idx]['date']
                oscillation_confirmed_periods.append((
                    period_confirmed_idx,
                    period_end_idx,
                    confirmed_date,
                    end_date
                ))
            else:
                oscillation_confirmed_periods.append((
                    period_confirmed_idx,
                    period_end_idx,
                    period_confirmed_idx,
                    period_end_idx
                ))
        
        # 第二步：批量标记震荡状态和过滤买入信号
        filtered_count = 0
        oscillation_periods_for_log = []
        
        for confirmed_idx, end_idx, confirmed_date, end_date in oscillation_confirmed_periods:
            # 批量标记震荡状态（从确认日到结束日）
            df.iloc[confirmed_idx:end_idx + 1, df.columns.get_loc('oscillation_state')] = 1
            
            # 批量过滤震荡期间的买入信号
            for idx in range(confirmed_idx, end_idx + 1):
                if df.iloc[idx]['buy_signal'] == 1:
                    df.iloc[idx, df.columns.get_loc('buy_signal')] = 0
                    filtered_count += 1
            
            # 记录日志信息
            if 'date' in df.columns and hasattr(confirmed_date, 'strftime'):
                date_info = f"{confirmed_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
            else:
                date_info = f"{confirmed_date} ~ {end_date}"
            oscillation_periods_for_log.append(date_info)
        
        # 日志输出
        should_log = log_details and not self._oscillation_log_printed
        if should_log and oscillation_periods_for_log:
            self._oscillation_log_printed = True
            periods_str = ", ".join(oscillation_periods_for_log)
            logger.info(
                f"震荡下行过滤（逐日判断，无前瞻性偏差）:\n"
                f"[{periods_str}]\n"
                f"共过滤 {filtered_count} 个买入信号"
            )
        
        # 保存震荡确认时间到缓存（供绘图使用）
        self._oscillation_confirmed_periods = oscillation_confirmed_periods
        
        # 检测震荡周期（用于展示/绘图，不用于过滤信号）
        # 这个函数返回的周期信息可以被 plotter.py 等模块使用
        _ = self._detect_oscillation_decline_periods(df)

    def get_oscillation_confirmed_periods(self) -> List[Tuple]:
        """
        返回震荡确认时间列表（逐日判断的结果，用于绘图）
        
        Returns:
            list: [(confirmed_idx, end_idx, confirmed_date, end_date), ...]
                  confirmed_idx/confirmed_date: 震荡被确认的时间点
                  end_idx/end_date: 震荡结束的时间点
        """
        return self._oscillation_confirmed_periods

    def _apply_oscillation_driven_trading_filter(self, df: pd.DataFrame) -> None:
        """
        应用震荡周期驱动的交易策略：
        - 只在震荡周期结束后开启交易窗口
        - 每个窗口只允许完成一笔交易（一买一卖）
        - 完成交易后关闭窗口，等待下一个震荡周期结束
        
        Args:
            df: 数据DataFrame（需要已经生成buy_signal）
        """
        if not self._oscillation_confirmed_periods:
            logger.info("⚠️ 震荡驱动策略：未检测到震荡周期，禁止所有交易")
            df['buy_signal'] = 0
            return
        
        # 提取所有震荡结束时间点（交易窗口开启时刻）
        oscillation_end_dates = []
        for confirmed_idx, end_idx, confirmed_date, end_date in self._oscillation_confirmed_periods:
            oscillation_end_dates.append((end_idx, end_date))
        
        # 按时间排序
        oscillation_end_dates.sort(key=lambda x: x[0])
        
        logger.info(f"🔄 震荡驱动策略：检测到 {len(oscillation_end_dates)} 个震荡周期结束点")
        
        # 初始状态：所有信号都清零
        df['buy_signal'] = 0
        
        # 标记交易窗口状态
        df['trading_window'] = 0  # 0=禁止交易，1=允许交易
        
        # 遍历每个震荡结束点，开启交易窗口
        total_trades = 0
        for window_idx, (end_idx, end_date) in enumerate(oscillation_end_dates):
            # 交易窗口从震荡结束的下一天开始
            window_start_idx = end_idx + 1
            
            # 确定下一个震荡周期结束点（作为窗口的理论上限）
            if window_idx + 1 < len(oscillation_end_dates):
                next_end_idx = oscillation_end_dates[window_idx + 1][0]
            else:
                next_end_idx = len(df) - 1
            
            # 在这个窗口内查找第一笔完整交易
            if window_start_idx >= len(df):
                continue
            
            # 寻找买入信号（使用原始策略的买入信号）
            buy_found = False
            buy_idx = None
            
            # 重新生成这个窗口内的原始买入信号
            for idx in range(window_start_idx, min(next_end_idx + 1, len(df))):
                # 检查是否满足买入条件（close >= MA16）
                if df.iloc[idx]['close'] >= df.iloc[idx][f"{self.config['short_ma']}_ma"]:
                    buy_idx = idx
                    buy_found = True
                    break
            
            if not buy_found:
                continue
            
            # 标记买入
            df.iloc[buy_idx, df.columns.get_loc('buy_signal')] = 1
            df.iloc[buy_idx, df.columns.get_loc('trading_window')] = 1
            
            # 寻找卖出信号（从买入后开始）
            sell_found = False
            for idx in range(buy_idx + 1, min(next_end_idx + 1, len(df))):
                # 检查是否满足卖出条件（close < MA16 或 MACD < 0）
                if (df.iloc[idx]['close'] < df.iloc[idx][f"{self.config['short_ma']}_ma"] or 
                    df.iloc[idx]['macd'] < 0):
                    # 完成一笔交易，关闭窗口
                    df.iloc[idx, df.columns.get_loc('trading_window')] = 0
                    sell_found = True
                    total_trades += 1
                    
                    if 'date' in df.columns:
                        buy_date = df.iloc[buy_idx]['date']
                        sell_date = df.iloc[idx]['date']
                        logger.info(
                            f"  窗口{window_idx + 1}: 完成交易 "
                            f"买入={buy_date} 卖出={sell_date}"
                        )
                    break
                else:
                    # 持续持仓
                    df.iloc[idx, df.columns.get_loc('trading_window')] = 1
            
            # 如果没有找到卖出点，这笔交易将持续到窗口结束或数据末尾
            if not sell_found:
                if 'date' in df.columns:
                    buy_date = df.iloc[buy_idx]['date']
                    logger.info(
                        f"  窗口{window_idx + 1}: 买入后未卖出 "
                        f"买入={buy_date} (持仓至窗口结束)"
                    )
        
        logger.info(f"✅ 震荡驱动策略：完成 {total_trades} 笔交易")

    def get_detected_oscillation_periods(self, df: Optional[pd.DataFrame] = None) -> List[Tuple]:
        """
        返回最近一次检测到的震荡区间列表

        Args:
            df: 可选的 DataFrame，仅在确实需要重新检测时传入；
                一般情况下直接使用缓存即可

        Returns:
            list: 形如 (start_idx, end_idx, start_date, end_date, avg_score) 的元组列表
        """
        try:
            if df is not None:
                return list(self._detect_oscillation_decline_periods(df))
        except Exception:
            logger.warning("获取震荡检测结果失败，将返回缓存结果", exc_info=True)
            return []

        if self._oscillation_periods_cache is None:
            return []

        return list(self._oscillation_periods_cache)

    def analyze(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        执行策略分析

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            (添加了买卖信号的 DataFrame, 指标验证报告)
            如果不启用验证，返回 (DataFrame, None)
        """
        if df is None or len(df) == 0:
            logger.warning("数据为空，无法分析")
            return None, None

        # 设置默认震荡参数（如果不启用自适应）
        if not self.adaptive_oscillation:
            # 使用原始的默认参数（与commit d4c0caa一致）
            self._oscillation_outer_upper = 0.60
            self._oscillation_inner_upper = 0.30

        # 1. 跌停保护检查（只对A股执行，港股无涨跌停限制）
        # 使用与 data_fetcher.py 相同的港股识别逻辑
        is_hk_stock = False
        stock_code = None

        # 尝试从多个位置获取股票代码
        if 'ts_code' in df.columns and len(df) > 0:
            stock_code = str(df['ts_code'].iloc[0])
        elif 'code' in df.columns and len(df) > 0:
            stock_code = str(df['code'].iloc[0])
        elif hasattr(self, 'stock_code'):
            stock_code = str(self.stock_code)

        # 港股检测逻辑（与 data_fetcher.py 保持一致）
        if stock_code:
            # 去掉可能的市场后缀（如 .SH, .SZ）提取纯代码
            code_clean = stock_code.split('.')[0]

            # 检测是否为港股
            if code_clean.isdigit():
                code_len = len(code_clean)
                # 港股代码: 1-5位数字（可能去掉前导0）
                # A股代码: 6位数字
                if code_len <= 5:
                    is_hk_stock = True
            # 带后缀的港股代码
            elif '.HK' in stock_code.upper():
                is_hk_stock = True

        # 更新实例内的股票代码（供后续策略缓存使用）
        if stock_code and not self.stock_code:
            self.stock_code = stock_code

        logger.info(f"[跌停检查] 股票代码: {stock_code}, is_hk_stock={is_hk_stock}")

        # 只对A股执行跌停检查
        if not is_hk_stock:
            # 仅基于 close 计算最近30天的跌幅，无需整表复制
            p_change_temp = df['close'].pct_change() * 100

            # 只检查最近30天的数据
            if len(df) > 30:
                recent_idx = df.index[-30:]
            else:
                recent_idx = df.index

            recent_p_change = p_change_temp.loc[recent_idx]

            # 是否存在跌停风险
            mask = recent_p_change <= self.config['stop_loss']
            if mask.any():
                # 找出最近的跌停日期（保持与原逻辑一致）
                latest_idx = recent_p_change[mask].index[-1]
                latest_date = df.loc[latest_idx, 'date'] if 'date' in df.columns else latest_idx
                latest_drop_val = recent_p_change.loc[latest_idx]

                logger.warning(
                    f"近期存在跌停风险: {latest_date}, "
                    f"跌幅: {latest_drop_val:.2f}%, 跳过该股票"
                )
                return None, None

        # 2. 计算所有技术指标（如未预先计算）
        if not getattr(self, 'precomputed_indicators', False):
            try:
                df = calculate_all_indicators(
                    df,
                    self.config['init_k'],
                    self.config['init_d'],
                    self.config['init_date'],
                    rsi_fast_period=self.config.get('rsi_fast_period', 5),
                    rsi_slow_period=self.config.get('rsi_slow_period', 10)
                )
            except Exception as e:
                logger.error(f"计算技术指标失败: {e}")
                return None, None

        if len(df) == 0:
            logger.warning("指标计算后数据为空")
            return None, None

        # 2.5 验证技术指标
        indicator_report = None
        if self.validate_indicators and self.indicator_validator:
            passed, indicator_report = self.indicator_validator.validate(df)
            if not passed:
                logger.warning("技术指标验证失败")
                # 不中断流程，只记录警告

        # 3. 检测顶底背离
        # 注意：顶背离检测已改为延迟确认版本（无未来函数）
        # 返回的 top_index 是确认日（即极值点的下一天），实现了次日卖出
        df['top'] = 0
        df['bottom'] = 0

        # 初始化背离索引
        top_index = []
        bottom_index = []

        try:
            # 顶部背离（仅使用基于DIFF的检测方法）
            # 返回的是延迟一天的确认日期（无未来函数）
            # use_simple=True 使用简化版 get_peak_divergence_delayed
            top_index = get_peak_divergence_index(df, self.config['lookback_days'], use_simple=self.use_simple_divergence)
            if len(top_index) > 0:
                df.loc[top_index, 'top'] = 1

            # 底部背离
            bottom_index = get_bottom_divergence_index(df, self.config['lookback_days'])
            if len(bottom_index) > 0:
                df.loc[bottom_index, 'bottom'] = 1

        except Exception as e:
            logger.warning(f"背离检测失败: {e}")
            # 确保变量已定义
            top_index = []
            bottom_index = []

        # 4. 生成买入信号
        df['buy_signal'] = 0

        # 基础买入条件：收盘价 >= 16日均线
        # 这个条件生成原始的持有/卖出信号，后续的卖出策略会根据策略类型进行调整
        buy_index = df[df['close'] >= df[f"{self.config['short_ma']}_ma"]].index
        df.loc[buy_index, 'buy_signal'] = 1

        # 5. 自适应策略和执行顺序选择
        # 步骤0: 选择震荡参数 (V1 vs V4)
        # 步骤1: 选择卖出策略 (original vs gradual)
        # 步骤2: 选择执行顺序 (high_frequency vs high_quality)

        # 如果启用了策略缓存，且 sell_strategy/order 均为 auto，则优先尝试读取缓存
        used_cached_strategy = False
        stock_code_for_cache = self.stock_code or stock_code
        if (
            self.use_strategy_cache
            and stock_code_for_cache
            and self.sell_strategy == 'auto'
            and self.order_mode == 'auto'
        ):
            try:
                from . import strategy_cache
                cached = strategy_cache.get_strategy_choice(stock_code_for_cache)
                if cached:
                    cached_sell = cached.get('sell_strategy')
                    cached_order = cached.get('order')
                    if cached_sell in ('original', 'gradual'):
                        self.optimal_strategy = cached_sell
                    if cached_order in ('high_frequency', 'high_quality'):
                        self.optimal_order = cached_order
                    if self.optimal_strategy and self.optimal_order:
                        used_cached_strategy = True
                        logger.info(
                            f"使用策略缓存: 股票 {stock_code_for_cache}, "
                            f"卖出策略={self.optimal_strategy}, 执行顺序={self.optimal_order}"
                        )
            except Exception as e:
                logger.warning(f"读取策略缓存失败: {e}")

        # === 步骤0: 选择震荡参数（如果启用） ===
        if not self.adaptive_oscillation:
            # 设置默认震荡参数（如果不启用自适应）
            # 使用原始的默认参数（与commit d4c0caa一致）
            self._oscillation_outer_upper = 0.60
            self._oscillation_inner_upper = 0.30
        else:
            logger.info("自适应模式：正在评估最优震荡参数...")

            # 测试 V1 参数 (5%-60%, 10%-30%) - 原始默认参数
            self._oscillation_outer_upper = 0.60
            self._oscillation_inner_upper = 0.30
            df_test_v1 = df.copy()
            # 应用完整的过滤器组合（使用默认的original和high_frequency，只关注震荡参数影响）
            self._apply_combination(df_test_v1, 'original', 'high_frequency', bottom_index, top_index)
            backtest_v1 = self._quick_backtest(df_test_v1)

            # 测试 V4 参数 (5%-80%, 10%-50%) - 更宽松的参数
            self._oscillation_outer_upper = 0.80
            self._oscillation_inner_upper = 0.50
            df_test_v4 = df.copy()
            self._apply_combination(df_test_v4, 'original', 'high_frequency', bottom_index, top_index)
            backtest_v4 = self._quick_backtest(df_test_v4)

            # 选择最优参数
            if backtest_v1 is not None and backtest_v4 is not None:
                initial_capital = 10000.0
                capital_v1 = backtest_v1['capital']
                capital_v4 = backtest_v4['capital']
                trades_v1 = backtest_v1['trades']
                trades_v4 = backtest_v4['trades']
                return_v1 = (capital_v1 - initial_capital) / initial_capital * 100
                return_v4 = (capital_v4 - initial_capital) / initial_capital * 100

                # 规则1: V4收益率 > V1收益率
                if capital_v4 > capital_v1:
                    diff = return_v4 - return_v1
                    self._oscillation_outer_upper = 0.80
                    self._oscillation_inner_upper = 0.50
                    self.selected_oscillation_version = 'V4'
                    logger.info(
                        "在 V1 参数 和 V4 参数 之间选择了 V4，原因是："
                        f"V4 收益更高 +{diff:.1f}%（V1: {return_v1:.1f}%/{trades_v1}笔，"
                        f"V4: {return_v4:.1f}%/{trades_v4}笔）"
                    )
                # 规则2: V1交易次数比V4多50%以上（过度交易）
                elif trades_v4 > 0 and trades_v1 > trades_v4 * 1.5:
                    increase_pct = ((trades_v1 - trades_v4) / trades_v4 * 100)
                    self._oscillation_outer_upper = 0.80
                    self._oscillation_inner_upper = 0.50
                    self.selected_oscillation_version = 'V4'
                    logger.info(
                        "在 V1 参数 和 V4 参数 之间选择了 V4，原因是："
                        f"V1 交易过于频繁（V1: {trades_v1}笔，V4: {trades_v4}笔，"
                        f"交易增长约 {increase_pct:.0f}%）"
                    )
                # 规则3: 默认V1
                else:
                    self._oscillation_outer_upper = 0.60
                    self._oscillation_inner_upper = 0.30
                    self.selected_oscillation_version = 'V1'
                    logger.info(
                        "在 V1 参数 和 V4 参数 之间选择了 V1，原因是："
                        f"V4 相比 V1 提升不明显（V1: {return_v1:.1f}%/{trades_v1}笔，"
                        f"V4: {return_v4:.1f}%/{trades_v4}笔）"
                    )
            else:
                # 回测失败，使用默认V1
                self._oscillation_outer_upper = 0.60
                self._oscillation_inner_upper = 0.30
                self.selected_oscillation_version = 'V1'
                logger.warning("震荡参数回测失败，使用默认V1参数")

        # === 步骤1: 选择卖出策略 ===
        if self.sell_strategy == 'auto' and not used_cached_strategy:
            logger.info(
                "自适应模式：正在评估最优卖出策略 "
                "（在原始策略 original 与渐进式策略 gradual 之间自动选择）..."
            )

            # 使用high_frequency顺序测试两种卖出策略
            df_test_orig = df.copy()
            self._apply_combination(df_test_orig, 'original', 'high_frequency', bottom_index, top_index)
            backtest_orig = self._quick_backtest(df_test_orig)

            df_test_grad = df.copy()
            self._apply_combination(df_test_grad, 'gradual', 'high_frequency', bottom_index, top_index)
            backtest_grad = self._quick_backtest(df_test_grad)

            # 选择最优卖出策略（保留原逻辑：gradual需要提升30%以上）
            if backtest_grad is not None and backtest_orig is not None:
                initial_capital = 10000.0

                # 计算收益率（用于显示）
                return_orig = (backtest_orig['capital'] - initial_capital) / initial_capital * 100
                return_grad = (backtest_grad['capital'] - initial_capital) / initial_capital * 100

                # 判断是否选择渐进式策略
                if backtest_grad['capital'] > initial_capital and backtest_grad['capital'] > backtest_orig['capital']:
                    # 最终资金为正，且渐进式更优
                    improvement = (backtest_grad['capital'] - backtest_orig['capital']) / backtest_orig['capital']
                    if improvement > 0.30:  # 提升超过30%
                        self.optimal_strategy = 'gradual'
                        logger.info(
                            "在 原始策略(original) 和 渐进式策略(gradual) 之间选择了 渐进式策略，原因是："
                            f"资金提升{improvement*100:.1f}%（原始: ¥{backtest_orig['capital']:,.2f}"
                            f"[{return_orig:.1f}%] → 渐进式: ¥{backtest_grad['capital']:,.2f}"
                            f"[{return_grad:.1f}%]）"
                        )
                    else:
                        self.optimal_strategy = 'original'
                        logger.info(
                            "在 原始策略(original) 和 渐进式策略(gradual) 之间选择了 原始策略，原因是："
                            f"渐进式提升不足30%（提升{improvement*100:.1f}%："
                            f"原始 ¥{backtest_orig['capital']:,.2f} vs 渐进式 ¥{backtest_grad['capital']:,.2f}）"
                        )
                else:
                    # 其他情况默认使用原始策略
                    self.optimal_strategy = 'original'
                    logger.info(
                        "在 原始策略(original) 和 渐进式策略(gradual) 之间选择了 原始策略，原因是："
                        f"渐进式最终资金不优于原始（原始: ¥{backtest_orig['capital']:,.2f}"
                        f"[{return_orig:.1f}%]，渐进式: ¥{backtest_grad['capital']:,.2f}"
                        f"[{return_grad:.1f}%]）"
                    )
            else:
                self.optimal_strategy = 'original'
                logger.warning("回测失败，默认使用原始策略")
        else:
            # 使用指定的策略或已从缓存加载的策略
            if not self.optimal_strategy:
                self.optimal_strategy = self.sell_strategy

        # === 步骤2: 选择执行顺序 ===
        if self.order_mode == 'auto' and not used_cached_strategy:
            # 自适应选择执行顺序：在选定的策略基础上，比较high_frequency和high_quality
            logger.info(f"自适应模式：已确定卖出策略为 {self.optimal_strategy}")

            df_test_new = df.copy()
            self._apply_combination(df_test_new, self.optimal_strategy, 'high_frequency', bottom_index, top_index)
            backtest_new = self._quick_backtest(df_test_new)

            df_test_old = df.copy()
            self._apply_combination(df_test_old, self.optimal_strategy, 'high_quality', bottom_index, top_index)
            backtest_old = self._quick_backtest(df_test_old)

            # 选择最优执行顺序
            if backtest_new is not None and backtest_old is not None:
                # 前置条件：只有当NEW收益比OLD高出20%以上时，才进行智能评分
                # 否则直接使用NEW（避免不必要的评分计算）
                capital_new = backtest_new['capital']
                capital_old = backtest_old['capital']

                if capital_new > capital_old * 1.2:
                    # HIGH_FREQUENCY显著更优（>20%），直接使用HIGH_FREQUENCY
                    self.optimal_order = 'high_frequency'
                    improvement = (capital_new - capital_old) / capital_old * 100
                    logger.info(
                        "在 高频率模式(high_frequency) 和 高质量模式(high_quality) 之间选择了 高频率模式，原因是："
                        f"高频率模式收益显著更高（+{improvement:.1f}%）"
                    )
                else:
                    # HIGH_FREQUENCY优势不显著或HIGH_QUALITY更优，进行智能评分比较
                    self.optimal_order = self._select_better_order(backtest_new, backtest_old)
            else:
                self.optimal_order = 'high_frequency'
                logger.warning("执行顺序回测失败，默认使用高频率模式")
        else:
            # 使用指定的执行顺序（固定模式）或已从缓存加载的执行顺序
            if not self.optimal_order:
                self.optimal_order = self.order_mode
            logger.info(f"使用固定执行顺序: {self.optimal_order}")

        # 6. 在自适应模式下，将最新选择写入策略缓存
        if stock_code_for_cache and (self.sell_strategy == 'auto' or self.order_mode == 'auto'):
            try:
                from . import strategy_cache
                strategy_cache.save_strategy_choice(
                    stock_code_for_cache,
                    self.optimal_strategy,
                    self.optimal_order,
                )
            except Exception as e:
                logger.warning(f"写入策略缓存失败: {e}")

        # 7. 应用选定的策略和执行顺序组合
        self._apply_combination(df, self.optimal_strategy, self.optimal_order, bottom_index, top_index)

        # 8. 计算持仓状态
        df['position'] = df['buy_signal'].shift(1)
        df['position'] = df['position'].ffill()
        df.loc[:self.config['init_date'], 'position'] = 0

        return df, indicator_report

    def _calculate_block_index(self, df: pd.DataFrame, top_index: list) -> tuple:
        """
        计算顶部背离期间的买入阻止区域

        Args:
            df: 数据框
            top_index: 顶部背离日期列表

        Returns:
            (阻止买入的日期列表, DIFF顶背离失效的日期列表)
        """
        block_index = set()
        diff_invalidation_dates = set()  # 记录DIFF突破失效的日期
        close_up_index = df[df['close'] > df['close'].shift(1)].index

        for peak_date in top_index:
            tmpdf = df.loc[peak_date:]

            for date in tmpdf.index:
                # 阻止买入，直到：
                # 1. MACD <= 0，或
                # 2. DIFF 突破顶部背离点的 DIFF 值且收盘价上涨
                if tmpdf.loc[date, 'macd'] <= 0.0:
                    break
                elif (tmpdf.loc[date, 'diff'] > df.loc[peak_date, 'diff'] and
                      date in close_up_index):
                    # DIFF顶背离失效！记录这个日期
                    diff_invalidation_dates.add(date)
                    logger.debug(f"DIFF顶背离失效: {date}, DIFF突破 {df.loc[peak_date, 'diff']:.3f} -> {tmpdf.loc[date, 'diff']:.3f}")
                    break
                else:
                    block_index.add(date)

        return list(block_index), list(diff_invalidation_dates)

    def _select_better_order(self, backtest_new: dict, backtest_old: dict) -> str:
        """
        选择更优的执行顺序

        综合考虑3个比率（权重相同）：
        1. 收益比 = capital_new / capital_old
        2. 交易次数比 = trades_new / trades_old
        3. 收益增长率 = (capital_new - capital_old) / capital_old

        Args:
            backtest_new: 高频率模式的回测结果
            backtest_old: 高质量模式的回测结果

        Returns:
            'high_frequency' or 'high_quality'
        """
        initial_capital = 10000.0

        # 提取指标
        capital_new = backtest_new['capital']
        capital_old = backtest_old['capital']
        trades_new = backtest_new['trades']
        trades_old = backtest_old['trades']
        avg_return_new = backtest_new['avg_return']
        avg_return_old = backtest_old['avg_return']
        win_rate_new = backtest_new['win_rate']
        win_rate_old = backtest_old['win_rate']

        # 计算收益率
        return_new = (capital_new - initial_capital) / initial_capital * 100
        return_old = (capital_old - initial_capital) / initial_capital * 100

        # 输出详细对比信息
        logger.info(f"  高频率模式: ¥{capital_new:,.2f} (收益{return_new:.1f}%) "
                   f"交易{trades_new}次 平均{avg_return_new:.2f}% 胜率{win_rate_new:.1f}%")
        logger.info(f"  高质量模式: ¥{capital_old:,.2f} (收益{return_old:.1f}%) "
                   f"交易{trades_old}次 平均{avg_return_old:.2f}% 胜率{win_rate_old:.1f}%")

        # 计算3个关键比率
        # 1. 收益比 (capital_new / capital_old)
        capital_ratio = capital_new / capital_old if capital_old > 0 else 1.0

        # 2. 交易次数比 (trades_new / trades_old)
        trades_ratio = trades_new / trades_old if trades_old > 0 else 1.0

        # 3. 收益增长率 (新增收益 / 原始收益)
        # 如果 capital_new > capital_old，计算新增收益占原收益的比例
        if capital_new > capital_old:
            return_growth_ratio = (capital_new - capital_old) / capital_old if capital_old > 0 else 0
        else:
            # 如果 capital_new < capital_old，收益增长为负
            return_growth_ratio = (capital_new - capital_old) / capital_old if capital_old > 0 else 0

        # 输出三个比率
        logger.info(f"  📊 收益比: {capital_ratio:.4f}, 交易次数比: {trades_ratio:.4f}, 收益增长率: {return_growth_ratio:.4f}")

        # 综合判断逻辑：
        # 如果交易次数增长远高于收益增长，说明新增交易质量低，选择高质量模式
        # 判断标准：交易次数比 > 收益比，且 交易次数增长 >> 收益增长

        if trades_ratio > capital_ratio:
            # 交易次数增长比收益增长快
            # 计算"效率损失"：交易增长了X%，但收益只增长了Y%
            efficiency_gap = (trades_ratio - 1) - (capital_ratio - 1)

            # 如果交易次数增长显著高于收益增长（效率损失>10%），选择高质量模式
            if efficiency_gap > 0.1:  # 10%的效率损失阈值
                logger.info(
                    "在 高频率模式(high_frequency) 和 高质量模式(high_quality) 之间选择了 高质量模式，原因是："
                    f"交易次数增长{(trades_ratio-1)*100:.1f}%，但收益仅增长{(capital_ratio-1)*100:.1f}%，"
                    f"效率损失约 {efficiency_gap*100:.1f}%"
                )
                return 'high_quality'

        # 否则，选择收益更高的模式
        if capital_old > capital_new:
            improvement = (capital_old - capital_new) / capital_new * 100
            logger.info(
                "在 高频率模式(high_frequency) 和 高质量模式(high_quality) 之间选择了 高质量模式，原因是："
                f"高质量模式最终资金更高（+{improvement:.1f}%）"
            )
            return 'high_quality'
        else:
            logger.info(
                "在 高频率模式(high_frequency) 和 高质量模式(high_quality) 之间选择了 高频率模式，原因是："
                "高频率模式收益更高或至少不低于高质量模式"
            )
            return 'high_frequency'

    def _apply_combination(self, df: pd.DataFrame, strategy: str, order: str, bottom_index: list, top_index: list) -> None:
        """
        应用指定的策略和执行顺序组合

        Args:
            df: 数据框（会直接修改）
            strategy: 卖出策略类型 ('original', 'gradual')
            order: 执行顺序类型 ('high_frequency', 'high_quality')
            bottom_index: 底部背离索引列表
            top_index: 顶部背离索引列表
        """
        # 首先确保买卖信号已正确初始化
        # 注意：买入信号已在analyze()中基于"收盘价 >= 16日均线"生成

        # 0. 应用主升浪后保护期（优先级最高）
        self._apply_main_wave_protection(df)

        # 1. 应用震荡下行检测过滤
        self._apply_oscillation_decline_filter(df)
        
        # 1.5. 如果启用震荡驱动策略，使用完全不同的交易逻辑
        if self.oscillation_driven:
            self._apply_oscillation_driven_trading_filter(df)
            return  # 震荡驱动策略不需要后续的常规过滤逻辑

        if order == 'high_frequency':
            # NEW顺序：卖出策略 → 底部背离 → 顶部背离阻止 → RSI
            # 底部背离和RSI不会被错误阻止，产生更多买入机会

            # 1. 应用卖出策略
            self._apply_sell_strategy(df, strategy, top_index)

            # 2. 底部背离次日强制买入（但要尊重主升浪保护期）
            bottom_shift_index = df[df['bottom'].shift(1) == 1].index
            # 获取当前保护期，确保不覆盖保护期的限制
            protection_periods = self._detect_main_wave_protection_periods(df)
            if protection_periods:
                protection_mask = df['date'].isin(protection_periods)
                # 只在非保护期内执行底部背离买入
                valid_bottom_index = [idx for idx in bottom_shift_index if not protection_mask.loc[idx]]
                if valid_bottom_index:
                    df.loc[valid_bottom_index, 'buy_signal'] = 1
                # 记录被保护期阻止的底部背离次数
                blocked_count = len(bottom_shift_index) - len(valid_bottom_index)
                if blocked_count > 0:
                    logger.info(f"🛡️ 主升浪保护阻止了{blocked_count}次底部背离买入")
            else:
                # 没有保护期，正常执行底部背离买入
                df.loc[bottom_shift_index, 'buy_signal'] = 1

            # 3. 顶部背离期间阻止买入（根据策略类型决定是否启用)
            if strategy == 'original':
                # 原始策略：使用顶部背离阻止买入
                block_index, diff_invalidation_dates = self._calculate_block_index(df, top_index)
                df.loc[block_index, 'buy_signal'] = 0

                # 标记DIFF顶背离失效的日期
                df['diff_invalidation'] = 0
                if len(diff_invalidation_dates) > 0:
                    df.loc[diff_invalidation_dates, 'diff_invalidation'] = 1
            elif strategy == 'gradual':
                # 渐进式策略：禁用顶部背离阻止买入，保持原有信号
                df['diff_invalidation'] = 0

            # 4. RSI增强买入（可选，但要尊重主升浪保护期）
            if self.config.get('rsi_enabled', False):
                # 获取保护期，传递给RSI函数
                protection_periods = self._detect_main_wave_protection_periods(df)
                self._apply_rsi_enhancements(df, protection_periods)

        else:  # order == 'high_quality'
            # OLD顺序：底部背离 → RSI → 卖出策略 → 顶部背离阻止
            # 底部背离和RSI会被顶部背离阻止，过滤低质量交易，单笔收益更高

            # 1. 底部背离次日强制买入（在卖出策略之前，但要尊重主升浪保护期）
            bottom_shift_index = df[df['bottom'].shift(1) == 1].index
            # 获取当前保护期，确保不覆盖保护期的限制
            protection_periods = self._detect_main_wave_protection_periods(df)
            if protection_periods:
                protection_mask = df['date'].isin(protection_periods)
                # 只在非保护期内执行底部背离买入
                valid_bottom_index = [idx for idx in bottom_shift_index if not protection_mask.loc[idx]]
                if valid_bottom_index:
                    df.loc[valid_bottom_index, 'buy_signal'] = 1
                # 记录被保护期阻止的底部背离次数
                blocked_count = len(bottom_shift_index) - len(valid_bottom_index)
                if blocked_count > 0:
                    logger.info(f"🛡️ 主升浪保护阻止了{blocked_count}次底部背离买入")
            else:
                # 没有保护期，正常执行底部背离买入
                df.loc[bottom_shift_index, 'buy_signal'] = 1

            # 2. RSI增强买入（在卖出策略之前，但要尊重主升浪保护期）
            if self.config.get('rsi_enabled', False):
                # 获取保护期，传递给RSI函数
                protection_periods = self._detect_main_wave_protection_periods(df)
                self._apply_rsi_enhancements(df, protection_periods)

            # 3. 应用卖出策略
            self._apply_sell_strategy(df, strategy, top_index)

            # 4. 顶部背离期间阻止买入（根据策略类型决定是否启用）
            if strategy == 'original':
                # 原始策略：使用顶部背离阻止买入（会阻止上面的底部背离和RSI买入）
                block_index, diff_invalidation_dates = self._calculate_block_index(df, top_index)
                df.loc[block_index, 'buy_signal'] = 0

                # 标记DIFF顶背离失效的日期
                df['diff_invalidation'] = 0
                if len(diff_invalidation_dates) > 0:
                    df.loc[diff_invalidation_dates, 'diff_invalidation'] = 1
            elif strategy == 'gradual':
                # 渐进式策略：禁用顶部背离阻止买入，保持原有信号
                df['diff_invalidation'] = 0

    def _apply_sell_strategy(self, df: pd.DataFrame, strategy: str, top_index: list) -> None:
        """
        应用指定的卖出策略

        Args:
            df: 数据框（会直接修改）
            strategy: 策略类型 ('original', 'gradual')
            top_index: 顶部背离索引列表
        """
        sell_index = set()

        if strategy == 'original':
            # 原始策略：跌破MA16或MACD<0立即卖出
            sell_index1 = df[df['close'] < df[f"{self.config['short_ma']}_ma"]].index
            sell_index2 = df[df['macd'] < 0].index
            sell_index = set(sell_index1).union(set(sell_index2))

        elif strategy == 'gradual':
            # 渐进式止损：区分上升趋势和下跌趋势
            # 1. 下跌趋势（ma16 < ma45）：使用传统止损（跌破MA16立即止损）
            # 2. 上升/横盘趋势（ma16 >= ma45）：使用渐进式止损（连续3天跌破MA16才止损）

            # 第一步：恢复所有因跌破MA16而被设置为0的信号为1（假设持有状态）
            # 这样做是为了取消基础买入条件中"跌破MA16立即卖出"的逻辑
            below_ma_index = df[df['close'] < df[f"{self.config['short_ma']}_ma"]].index
            df.loc[below_ma_index, 'buy_signal'] = 1  # 先恢复为持有状态

            # 第二步：根据趋势应用不同的止损策略
            for i in range(2, len(df)):
                idx = df.index[i]
                # 检查当前趋势
                ma16 = df.loc[idx, '16_ma']
                ma45 = df.loc[idx, '45_ma']
                below_ma_0 = df.loc[idx, 'close'] < df.loc[idx, f"{self.config['short_ma']}_ma"]

                if ma16 < ma45:
                    # 下跌趋势：使用传统止损（跌破MA16立即止损）
                    if below_ma_0:
                        sell_index.add(idx)
                else:
                    # 上升/横盘趋势：使用渐进式止损（连续3天跌破MA16才止损）
                    if i >= 2:  # 确保有足够的历史数据
                        idx_1 = df.index[i-1]
                        idx_2 = df.index[i-2]

                        below_ma_1 = df.loc[idx_1, 'close'] < df.loc[idx_1, f"{self.config['short_ma']}_ma"]
                        below_ma_2 = df.loc[idx_2, 'close'] < df.loc[idx_2, f"{self.config['short_ma']}_ma"]

                        # 严格要求：只有连续3天都跌破才卖出
                        if below_ma_0 and below_ma_1 and below_ma_2:
                            sell_index.add(idx)

        # 顶部背离强制卖出（区分策略）
        # 注意：top_index 已经是延迟一天的确认日，即顶背离次日卖出
        if strategy == 'original':
            # 原始策略：所有顶部背离都强制卖出
            sell_index = sell_index.union(set(top_index))
        elif strategy == 'gradual':
            # 渐进式策略：智能顶背离处理
            # 只有在明显上升趋势（MA16 > MA45）时才完全屏蔽顶背离
            # 在下跌/横盘趋势中，恢复顶背离的保护作用
            filtered_top_index = []
            for idx in top_index:
                if idx in df.index:
                    # 检查当前是否处于明显上升趋势
                    ma16 = df.loc[idx, '16_ma']
                    ma45 = df.loc[idx, '45_ma']

                    if ma16 <= ma45:
                        # 非上升趋势（下跌/横盘），执行顶背离卖出
                        filtered_top_index.append(idx)
                    # else: 上升趋势中，屏蔽顶背离卖出

            sell_index = sell_index.union(set(filtered_top_index))

        # 应用卖出信号
        df.loc[list(sell_index), 'buy_signal'] = 0

    def _quick_backtest(self, df: pd.DataFrame) -> dict:
        """
        快速回测，返回详细的交易指标（用于自适应策略选择）

        Args:
            df: 包含buy_signal的数据框

        Returns:
            dict: {
                'capital': 最终资金,
                'trades': 交易次数,
                'avg_return': 平均单笔收益率(%),
                'win_rate': 胜率(%)
            }
            如果回测失败返回None
        """
        try:
            capital = 10000.0
            holding = False
            buy_price = 0
            shares = 0
            trades = []  # 记录每笔交易的收益率

            for i in range(len(df)):
                row = df.iloc[i]
                curr_signal = row['buy_signal']
                curr_price = row['close']  # 使用收盘价

                # 买入
                if curr_signal == 1 and not holding:
                    buy_price = curr_price
                    shares = capital / buy_price  # 计算可买股数
                    transaction_amount = shares * buy_price

                    # 扣除买入手续费
                    buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                    capital -= buy_commission

                    holding = True

                # 卖出
                elif curr_signal == 0 and holding:
                    sell_price = curr_price
                    transaction_amount = shares * sell_price

                    # 扣除卖出手续费
                    sell_commission = self._calculate_commission(transaction_amount, is_buy=False)

                    # 计算本次交易收益率
                    trade_return = (sell_price - buy_price) / buy_price * 100
                    trades.append(trade_return)

                    # 更新资金：卖出所得 - 手续费
                    capital = transaction_amount - sell_commission
                    holding = False
                    shares = 0

            # 最后还持仓，用收盘价计算
            if holding:
                sell_price = df.iloc[-1]['close']
                transaction_amount = shares * sell_price
                sell_commission = self._calculate_commission(transaction_amount, is_buy=False)

                # 计算本次交易收益率
                trade_return = (sell_price - buy_price) / buy_price * 100
                trades.append(trade_return)

                capital = transaction_amount - sell_commission

            # 计算交易质量指标
            if len(trades) > 0:
                avg_return = sum(trades) / len(trades)
                win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
            else:
                avg_return = 0
                win_rate = 0

            return {
                'capital': capital,
                'trades': len(trades),
                'avg_return': avg_return,
                'win_rate': win_rate
            }

        except Exception as e:
            logger.error(f"快速回测失败: {e}")
            return None

    def _apply_rsi_enhancements(self, df: pd.DataFrame, protection_periods: list = None) -> None:
        """
        应用RSI增强买入逻辑（改进v3：大幅降低交易频率，避免手续费侵蚀）

        改进点：
        1. RSI超卖阈值从20降低到18（更极端，只抓最好的机会）
        2. RSI金叉要求更低位（<40而非<45，极低位）
        3. 增加交易间隔限制（至少间隔5天，避免频繁交易）
        4. 更严格的确认条件（MACD、价格反弹、涨幅）
        5. 尊重震荡区间：震荡期间不添加RSI买入信号

        规则1: RSI极度超卖 (<18) + 接近均线 + 不在跌停 = 抄底买入
        规则2: RSI短期金叉 (6日突破14日) + RSI < 40 + (MACD>0.1 或 涨幅>3%) = 反转买入

        Args:
            df: 数据框（会直接修改）
            protection_periods: 主升浪保护期列表（可选）
        """
        rsi_oversold = self.config.get('rsi_oversold', 18)
        rsi_threshold = self.config.get('rsi_threshold', 40)
        rsi_ma_ratio = self.config.get('rsi_ma_ratio', 0.95)
        rsi_min_strength = self.config.get('rsi_min_strength', True)
        rsi_min_gap = self.config.get('rsi_min_gap', 5)  # 最小间隔天数

        # 初始化RSI买入类型标记列
        df['rsi_buy_type'] = ''

        n = len(df)
        if n < 2:
            return

        # 使用 numpy 数组以减少逐行 DataFrame 访问开销
        rsi_arr = df['rsi'].to_numpy()
        rsi6_arr = df['rsi_6'].to_numpy()
        close_arr = df['close'].to_numpy()
        ma16_arr = df[f"{self.config['short_ma']}_ma"].to_numpy()
        pchg_arr = df['p_change'].to_numpy()
        macd_arr = df['macd'].to_numpy()
        date_arr = df['date'].to_numpy() if 'date' in df.columns else df.index.to_numpy()

        buy_signal_arr = df['buy_signal'].to_numpy()
        rsi_type_arr = df['rsi_buy_type'].to_numpy()

        # 将保护期转换为集合以加速 membership 判断
        protection_set = set(protection_periods) if protection_periods else None

        # 获取震荡状态数组（如果存在）
        oscillation_arr = df['oscillation_state'].to_numpy() if 'oscillation_state' in df.columns else None

        last_rsi_buy_idx = -999  # 上次RSI买入的位置

        for i in range(1, n):
            # 检查是否距离上次RSI买入太近
            if i - last_rsi_buy_idx < rsi_min_gap:
                continue  # 跳过，避免频繁交易

            rsi = rsi_arr[i]
            rsi_6 = rsi6_arr[i]
            prev_rsi_6 = rsi6_arr[i-1]
            prev_rsi = rsi_arr[i-1]
            close = close_arr[i]
            ma_16 = ma16_arr[i]
            p_change = pchg_arr[i]
            macd = macd_arr[i]
            current_date = date_arr[i]

            # 规则1：RSI极度超卖（< 18），接近均线，且不在跌停
            if rsi < rsi_oversold and close >= ma_16 * rsi_ma_ratio and p_change > -8:
                # 检查是否在震荡区间
                if oscillation_arr is not None and oscillation_arr[i] == 1:
                    logger.debug(f"RSI超卖买入被震荡过滤阻止: {current_date}, RSI={rsi:.1f}")
                    continue
                
                if protection_set is None or current_date not in protection_set:
                    buy_signal_arr[i] = 1
                    rsi_type_arr[i] = 'RSI超卖'
                    last_rsi_buy_idx = i
                    logger.debug(f"RSI超卖买入: {current_date}, RSI={rsi:.1f}")
                else:
                    logger.debug(f"RSI超卖买入被主升浪保护期阻止: {current_date}")

            # 规则2：RSI短期金叉，且在极低位，并满足更严格的确认条件
            elif rsi_6 > rsi and prev_rsi_6 <= prev_rsi and rsi < rsi_threshold:
                confirmed = False

                if rsi_min_strength:
                    # 条件A：MACD > 0.1（趋势明显向上，不是弱势多头）
                    if macd > 0.1:
                        confirmed = True
                        logger.debug(f"RSI金叉+强MACD: {current_date}")

                    # 条件B：价格强势反弹 > 3%（改进：从2%提高到3%）
                    elif p_change > 3:
                        confirmed = True
                        logger.debug(f"RSI金叉+强反弹: {current_date}, 涨幅={p_change:.1f}%")

                    # 条件C：接近均线且RSI < 35（极低位金叉，从40降到35）
                    elif close >= ma_16 * 0.98 and rsi < 35:
                        confirmed = True
                        logger.debug(f"RSI极低位金叉: {current_date}, RSI={rsi:.1f}")
                else:
                    if close >= ma_16 * 0.98:
                        confirmed = True

                if confirmed:
                    # 检查是否在震荡区间
                    if oscillation_arr is not None and oscillation_arr[i] == 1:
                        logger.debug(f"RSI金叉买入被震荡过滤阻止: {current_date}, RSI_6={rsi_6:.1f}")
                        continue
                    
                    if protection_set is None or current_date not in protection_set:
                        buy_signal_arr[i] = 1
                        rsi_type_arr[i] = 'RSI金叉'
                        last_rsi_buy_idx = i
                        logger.debug(f"RSI金叉买入: {current_date}, RSI_6={rsi_6:.1f}, RSI={rsi:.1f}")
                    else:
                        logger.debug(f"RSI金叉买入被主升浪保护期阻止: {current_date}")

        # 回写修改后的列
        df['buy_signal'] = buy_signal_arr
        df['rsi_buy_type'] = rsi_type_arr

    def _apply_signals_order(self, df: pd.DataFrame, order: str, bottom_index: list, top_index: list) -> None:
        """
        应用指定执行顺序的信号处理逻辑

        Args:
            df: 数据框（会直接修改）
            order: 执行顺序类型 ('high_frequency' or 'high_quality')
            bottom_index: 底部背离索引列表
            top_index: 顶部背离索引列表

        注意: OLD顺序需要在应用卖出策略**之前**执行底部背离和RSI！
        所以这个方法需要同时处理卖出信号的应用。
        """
        logger.debug(f"应用{order}执行顺序, 底部背离数: {len(bottom_index)}, 顶部背离数: {len(top_index)}")

        if order == 'high_frequency':
            # NEW顺序：卖出策略 → 底部背离 → 顶部背离阻止 → RSI
            # 这样底部背离和RSI不会被错误阻止，产生更多买入机会

            # 注意：卖出策略已在调用此方法之前应用

            # 1. 底部背离次日强制买入
            bottom_shift_index = df[df['bottom'].shift(1) == 1].index
            if len(bottom_shift_index) > 0:
                logger.debug(f"NEW顺序: 底部背离次日买入 {len(bottom_shift_index)} 次")
            df.loc[bottom_shift_index, 'buy_signal'] = 1

            # 2. 顶部背离期间阻止买入
            block_index, diff_invalidation_dates = self._calculate_block_index(df, top_index)
            if len(block_index) > 0:
                logger.debug(f"NEW顺序: 顶部背离阻止买入 {len(block_index)} 次")
            df.loc[block_index, 'buy_signal'] = 0

            # 标记DIFF顶背离失效的日期
            df['diff_invalidation'] = 0
            if len(diff_invalidation_dates) > 0:
                df.loc[diff_invalidation_dates, 'diff_invalidation'] = 1

            # 3. RSI增强买入（可选）
            rsi_buy_count_before = len(df[df['buy_signal'] == 1])
            if self.config.get('rsi_enabled', False):
                self._apply_rsi_enhancements(df)
                rsi_buy_count_after = len(df[df['buy_signal'] == 1])
                if rsi_buy_count_after > rsi_buy_count_before:
                    logger.debug(f"NEW顺序: RSI增加买入 {rsi_buy_count_after - rsi_buy_count_before} 次")

        else:  # order == 'high_quality'
            # OLD顺序：底部背离 → RSI → 卖出策略 → 顶部背离阻止
            # 部分底部背离和RSI信号会被阻止，过滤了低质量交易，单笔收益更高

            # 注意：卖出策略已在调用此方法之前应用，但我们需要重新应用信号才能体现OLD顺序的效果
            # 问题是：我们无法"撤销"已应用的卖出策略...

            # 解决方案：OLD顺序需要在卖出策略应用之前插入底部背离和RSI信号
            # 但这要求重新设计调用顺序...

            # 暂时的解决方案：在已有卖出策略的基础上，用OLD顺序应用信号
            # 1. 底部背离次日强制买入
            bottom_shift_index = df[df['bottom'].shift(1) == 1].index
            if len(bottom_shift_index) > 0:
                logger.debug(f"OLD顺序: 底部背离次日买入 {len(bottom_shift_index)} 次")
            df.loc[bottom_shift_index, 'buy_signal'] = 1

            # 2. RSI增强买入（可选）
            rsi_buy_count_before = len(df[df['buy_signal'] == 1])
            if self.config.get('rsi_enabled', False):
                self._apply_rsi_enhancements(df)
                rsi_buy_count_after = len(df[df['buy_signal'] == 1])
                if rsi_buy_count_after > rsi_buy_count_before:
                    logger.debug(f"OLD顺序: RSI增加买入 {rsi_buy_count_after - rsi_buy_count_before} 次")

            # 3. 顶部背离期间阻止买入（会阻止上面的底部背离和RSI买入）
            block_index, diff_invalidation_dates = self._calculate_block_index(df, top_index)
            buy_count_before_block = len(df[df['buy_signal'] == 1])
            df.loc[block_index, 'buy_signal'] = 0
            buy_count_after_block = len(df[df['buy_signal'] == 1])
            if buy_count_before_block > buy_count_after_block:
                logger.debug(f"OLD顺序: 顶部背离阻止了 {buy_count_before_block - buy_count_after_block} 次买入")

            # 标记DIFF顶背离失效的日期
            df['diff_invalidation'] = 0
            if len(diff_invalidation_dates) > 0:
                df.loc[diff_invalidation_dates, 'diff_invalidation'] = 1

    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """
        获取最新的交易信号

        Args:
            df: 分析后的数据框

        Returns:
            包含信号信息的字典
        """
        if df is None or len(df) < 2:
            return {'signal': 'NO_DATA', 'reason': '数据不足'}

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # 判断信号类型
        signal = 'HOLD'
        reason = []
        strength = 0  # 信号强度 (0-5)

        # 买入信号判断
        if latest['buy_signal'] == 1.0 and previous['buy_signal'] != 1.0:
            signal = 'BUY'

            # 分析买入理由和强度
            # 1. 检查是否突破或站稳16日均线（核心买入条件）
            if latest['close'] >= latest[f"{self.config['short_ma']}_ma"]:
                if previous['close'] < previous[f"{self.config['short_ma']}_ma"]:
                    # 刚突破
                    reason.append(f"突破{self.config['short_ma']}日均线")
                else:
                    # 站稳均线上方
                    reason.append(f"站稳{self.config['short_ma']}日均线")
                strength += 1

            # 2. K值超卖
            if latest['k'] < self.config['k_threshold']:
                reason.append(f"K值超卖({latest['k']:.1f})")
                strength += 1

            # 3. 中期趋势
            if latest[f"{self.config['mid_ma']}_ma"] > previous[f"{self.config['mid_ma']}_ma"]:
                reason.append("中期趋势向上")
                strength += 1

            # 4. 底部背离（强信号）
            if latest['bottom'] == 1.0 or previous['bottom'] == 1.0:
                reason.append("底部背离信号")
                strength += 2

            # 5. MACD多头
            if latest['macd'] > 0:
                reason.append("MACD多头")
                strength += 1

        # 卖出信号判断
        elif latest['buy_signal'] == 0.0 and previous['buy_signal'] == 1.0:
            signal = 'SELL'

            if latest['close'] < latest[f"{self.config['short_ma']}_ma"]:
                reason.append(f"跌破{self.config['short_ma']}日均线")
                strength += 1

            if latest['macd'] < 0:
                reason.append("MACD转负")
                strength += 1

            if latest['top'] == 1.0:
                reason.append("顶部背离信号")
                strength += 2

        # 持有状态分析
        elif latest['buy_signal'] == 1.0:
            signal = 'HOLD_BUY'
            reason.append("维持持仓")

        return {
            'signal': signal,
            'strength': min(strength, 5),
            'reason': ', '.join(reason) if reason else '无明确信号',
            'price': latest['close'],
            'date': latest['date'],
            'k': latest['k'],
            'd': latest['d'],
            'macd': latest['macd'],
            'diff': latest['diff'],
            'ma_16': latest[f"{self.config['short_ma']}_ma"],
            'ma_45': latest[f"{self.config['mid_ma']}_ma"],
        }

    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        回测策略（含手续费）

        交易逻辑：
        - 看到买入信号(buy_signal=1)后，当日收盘买入
        - 看到卖出信号(buy_signal=0)后，当日收盘卖出
        - 收益 = (卖出收盘价 - 买入收盘价) / 买入收盘价
        - 扣除手续费：
          * A股：买入佣金(0.015%,最低5元) + 卖出佣金(0.015%,最低5元) + 卖出印花税(0.05%)
          * 港股：买卖双向佣金(0.25%) + 买卖双向印花税(0.13%)
          * 美股：买卖双向佣金(0.02%)

        Args:
            df: 分析后的数据框
            initial_capital: 初始资金

        Returns:
            回测结果字典
        """
        if df is None or 'buy_signal' not in df.columns:
            return None

        # 创建一个新列记录每日的资金（含手续费）
        n = len(df)
        if n == 0:
            return None

        # 为加速，使用 numpy 数组而不是逐行 DataFrame 访问
        close_arr = df['close'].to_numpy()
        signal_arr = df['buy_signal'].to_numpy()
        if 'date' in df.columns:
            # 统一转换为字符串，便于打印和比较（YYYY-MM-DD）
            date_arr = df['date'].astype(str).to_numpy()
        else:
            # 若不存在 date 列，则使用索引字符串作为日期占位
            date_arr = df.index.astype(str).to_numpy()

        capital = initial_capital
        capital_list = np.empty(n, dtype=float)
        trades = []
        holding = False
        buy_price = 0
        buy_date = None
        shares = 0  # 持有股数
        buy_commission = 0.0  # 当前持仓对应的买入手续费
        total_commission = 0.0  # 累计手续费

        # 同步维护一套「不含手续费」的资金轨迹，用于计算 gross_return
        # 为保持与历史实现一致，这里复用 _quick_backtest 中的初始资金 10000.0
        gross_capital = 10000.0
        gross_holding = False
        gross_buy_price = 0.0
        gross_shares = 0.0

        for i in range(n):
            curr_signal = signal_arr[i]
            curr_price = close_arr[i]
            curr_date = date_arr[i]

            # === 含手续费资金轨迹 ===
            # 当日有买入信号，当日收盘买入
            if curr_signal == 1 and not holding:
                buy_price = curr_price
                buy_date = curr_date

                # 计算可买股数
                shares = capital / buy_price
                transaction_amount = shares * buy_price

                # 计算并扣除买入手续费
                buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                capital -= buy_commission
                total_commission += buy_commission

                holding = True
                logger.debug(f"买入: {buy_date}, 价格: {buy_price:.2f}, 股数: {shares:.2f}, 手续费: {buy_commission:.2f}")

            # 当日有卖出信号（buy_signal=0），当日收盘卖出
            elif curr_signal == 0 and holding:
                sell_price = curr_price
                transaction_amount = shares * sell_price

                # 计算并扣除卖出手续费
                sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
                total_commission += sell_commission

                # 更新资金：卖出所得 - 手续费
                capital = transaction_amount - sell_commission

                # 计算收益率（基于初始买入资金）
                initial_investment = shares * buy_price
                profit = (sell_price - buy_price) * shares - (buy_commission + sell_commission)
                profit_rate = profit / initial_investment if initial_investment > 0 else 0

                trades.append({
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': curr_date,
                    'sell_price': sell_price,
                    'profit_rate': profit_rate,
                    'capital': capital,
                    'commission': buy_commission + sell_commission,  # 本次交易总手续费
                    'buy_commission': buy_commission,  # 买入手续费
                    'sell_commission': sell_commission,  # 卖出手续费
                })

                holding = False
                shares = 0
                buy_commission = 0.0
                logger.debug(f"卖出: {curr_date}, 价格: {sell_price:.2f}, 收益率: {profit_rate*100:.2f}%, 手续费: {sell_commission:.2f}")

            # === 不含手续费的「毛收益」资金轨迹 ===
            # 逻辑与 _quick_backtest 保持一致，但不再单独遍历 df
            if curr_signal == 1 and not gross_holding:
                # 使用当前 gross_capital 计算可买股数
                gross_buy_price = curr_price
                gross_shares = gross_capital / gross_buy_price
                gross_holding = True

            elif curr_signal == 0 and gross_holding:
                # 卖出全部持仓，不扣手续费
                sell_price_gross = curr_price
                transaction_amount_gross = gross_shares * sell_price_gross
                gross_capital = transaction_amount_gross
                gross_holding = False
                gross_shares = 0.0

            capital_list[i] = capital

        # 如果最后还持仓，用最后一天的收盘价计算
        if holding:
            sell_price = close_arr[-1]
            transaction_amount = shares * sell_price

            # 计算卖出手续费（用于净值计算）
            sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
            total_commission += sell_commission

            # 最终资金 = 卖出所得 - 手续费
            capital = transaction_amount - sell_commission

            # 计算收益率
            initial_investment = shares * buy_price
            profit = (sell_price - buy_price) * shares - (buy_commission + sell_commission)
            profit_rate = profit / initial_investment if initial_investment > 0 else 0

            trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': date_arr[-1],
                'sell_price': sell_price,
                'profit_rate': profit_rate,
                'capital': capital,
                'commission': buy_commission + sell_commission,  # 未平仓记录买入和卖出手续费
                'buy_commission': buy_commission,  # 记录买入手续费
                'sell_commission': sell_commission,  # 卖出手续费
            })

            # 更新最后一天的资金
            capital_list[-1] = capital

        # 不含手续费路径：如果仍在持仓，按最后一天收盘价平仓（与 _quick_backtest 一致）
        if gross_holding:
            last_row_gross = df.iloc[-1]
            sell_price_gross = last_row_gross['close']
            transaction_amount_gross = gross_shares * sell_price_gross
            gross_capital = transaction_amount_gross
            gross_holding = False
            gross_shares = 0.0

        # 写回资金曲线
        df['capital'] = capital_list

        # 计算指标
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # 毛收益率（不含手续费），保持与历史实现相同的基准计算方式
        gross_return = (gross_capital - initial_capital) / initial_capital * 100

        # 最大回撤
        capital_series = pd.Series(capital_list)
        running_max = capital_series.expanding().max()
        drawdown = (capital_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0

        # 计算夏普比率（基于交易收益）
        if len(trades) > 0:
            trade_returns = [t['profit_rate'] for t in trades]
            trade_returns_series = pd.Series(trade_returns)
            sharpe = trade_returns_series.mean() / trade_returns_series.std() * np.sqrt(252) if trade_returns_series.std() > 0 else 0
        else:
            sharpe = 0

        # 计算胜率
        winning_trades = sum(1 for t in trades if t['profit_rate'] > 0)
        win_rate = (winning_trades / len(trades) * 100) if len(trades) > 0 else 0

        # 构建资金曲线关键节点
        sample_interval = max(1, len(df) // 10)
        capital_curve = []

        for i in range(0, len(df), sample_interval):
            row = df.iloc[i]
            capital_curve.append({
                'date': row['date'],
                'capital': capital_list[i],
                'return_pct': (capital_list[i] / initial_capital - 1) * 100
            })

        # 确保包含最后一天
        if len(df) > 0 and (len(df) - 1) % sample_interval != 0:
            last_row = df.iloc[-1]
            capital_curve.append({
                'date': last_row['date'],
                'capital': capital_list[-1],
                'return_pct': (capital_list[-1] / initial_capital - 1) * 100
            })

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'gross_return': gross_return,  # 新增：毛收益率（不含手续费）
            'total_commission': total_commission,  # 新增：累计手续费
            'commission_rate': (total_commission / initial_capital * 100),  # 新增：手续费率
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trading_days': len(df),
            'capital_curve': capital_curve,
            'trades': trades,  # 保存交易明细
        }

    def backtest_legacy(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        旧版回测实现（包含内部调用 _quick_backtest 计算 gross_return）

        用于对比重构前后的结果是否一致，不在正常流程中使用。
        """
        if df is None or 'buy_signal' not in df.columns:
            return None

        capital = initial_capital
        capital_list = []
        trades = []
        holding = False
        buy_price = 0
        buy_date = None
        shares = 0
        buy_commission = 0.0
        total_commission = 0.0

        for i in range(len(df)):
            row = df.iloc[i]

            if row['buy_signal'] == 1 and not holding:
                buy_price = row['close']
                buy_date = row['date']
                shares = capital / buy_price
                transaction_amount = shares * buy_price
                buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                capital -= buy_commission
                total_commission += buy_commission
                holding = True

            elif row['buy_signal'] == 0 and holding:
                sell_price = row['close']
                transaction_amount = shares * sell_price
                sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
                total_commission += sell_commission
                capital = transaction_amount - sell_commission
                initial_investment = shares * buy_price
                profit = (sell_price - buy_price) * shares - (buy_commission + sell_commission)
                profit_rate = profit / initial_investment if initial_investment > 0 else 0

                trades.append({
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': row['date'],
                    'sell_price': sell_price,
                    'profit_rate': profit_rate,
                    'capital': capital,
                    'commission': buy_commission + sell_commission,
                    'buy_commission': buy_commission,
                    'sell_commission': sell_commission,
                })

                holding = False
                shares = 0
                buy_commission = 0.0

            capital_list.append(capital)

        if holding:
            last_row = df.iloc[-1]
            sell_price = last_row['close']
            transaction_amount = shares * sell_price

            sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
            total_commission += sell_commission

            capital = transaction_amount - sell_commission

            initial_investment = shares * buy_price
            profit = (sell_price - buy_price) * shares - (buy_commission + sell_commission)
            profit_rate = profit / initial_investment if initial_investment > 0 else 0

            trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': last_row['date'],
                'sell_price': sell_price,
                'profit_rate': profit_rate,
                'capital': capital,
                'commission': buy_commission + sell_commission,
                'buy_commission': buy_commission,
                'sell_commission': sell_commission,
            })

            capital_list[-1] = capital

        df = df.copy()
        df['capital'] = capital_list

        # 旧实现：通过 _quick_backtest 计算无手续费的毛收益率
        commission_backup = self.config.get('commission_enabled', True)
        self.config['commission_enabled'] = False
        try:
            backtest_no_fee = self._quick_backtest(df)
            if backtest_no_fee is not None and isinstance(backtest_no_fee, dict):
                gross_capital = backtest_no_fee['capital']
                gross_return = (gross_capital - initial_capital) / initial_capital * 100
            else:
                gross_return = 0.0
        finally:
            self.config['commission_enabled'] = commission_backup

        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        capital_series = pd.Series(capital_list)
        running_max = capital_series.expanding().max()
        drawdown = (capital_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0

        if len(trades) > 0:
            trade_returns = [t['profit_rate'] for t in trades]
            trade_returns_series = pd.Series(trade_returns)
            sharpe = trade_returns_series.mean() / trade_returns_series.std() * np.sqrt(252) if trade_returns_series.std() > 0 else 0
        else:
            sharpe = 0

        winning_trades = sum(1 for t in trades if t['profit_rate'] > 0)
        win_rate = (winning_trades / len(trades) * 100) if len(trades) > 0 else 0

        sample_interval = max(1, len(df) // 10)
        capital_curve = []

        for i in range(0, len(df), sample_interval):
            row = df.iloc[i]
            capital_curve.append({
                'date': row['date'],
                'capital': capital_list[i],
                'return_pct': (capital_list[i] / initial_capital - 1) * 100
            })

        if len(df) > 0 and (len(df) - 1) % sample_interval != 0:
            last_row = df.iloc[-1]
            capital_curve.append({
                'date': last_row['date'],
                'capital': capital_list[-1],
                'return_pct': (capital_list[-1] / initial_capital - 1) * 100
            })

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'gross_return': gross_return,
            'total_commission': total_commission,
            'commission_rate': (total_commission / initial_capital * 100),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trading_days': len(df),
            'capital_curve': capital_curve,
            'trades': trades,
        }

    def get_trading_signals(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        获取历史买卖点信号

        Args:
            df: 分析后的数据框
            initial_capital: 初始资金（用于计算每次交易后的资金余额）

        Returns:
            包含买卖点列表的字典
        """
        if df is None or 'buy_signal' not in df.columns:
            return {'buy_points': [], 'sell_points': [], 'initial_capital': initial_capital, 'total_trades': 0}

        # 首先执行backtest获取交易明细
        backtest_result = self.backtest(df, initial_capital)
        if not backtest_result or 'trades' not in backtest_result:
            return {'buy_points': [], 'sell_points': [], 'initial_capital': initial_capital, 'total_trades': 0}

        trades = backtest_result['trades']

        buy_points = []
        sell_points = []

        # 为日期匹配准备字符串视图，避免 dtype 差异导致比较失败
        if 'date' in df.columns:
            date_str_series = df['date'].astype(str)
            last_date_str = str(df.iloc[-1]['date'])
        else:
            date_str_series = df.index.astype(str)
            last_date_str = str(df.index[-1])

        # 从trades中提取买卖点信息
        for i, trade in enumerate(trades):
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']

            # 判断是否是最后一笔未平仓交易
            # 当日收盘价买卖逻辑下：
            # - 如果是最后一笔交易，且卖出日期是最后一天
            # - 需要检查是否真的有卖出信号（buy_signal=0）
            # - 如果有卖出信号，说明已经在当日收盘卖出了
            # - 如果没有卖出信号，说明是持仓到最后一天（未平仓）
            is_last_open = False
            if i == len(trades) - 1:
                sell_date_str = str(sell_date)
                if sell_date_str.split()[0] == last_date_str.split()[0]:
                    # 检查最后一天是否有真正的卖出信号
                    last_day_signal = df.iloc[-1]['buy_signal']
                    # 如果最后一天 buy_signal=1，说明是持仓状态（未卖出）
                    # 如果最后一天 buy_signal=0，说明当天已经卖出了
                    is_last_open = (last_day_signal == 1)

            # 获取买入/卖出日期对应的行信息（用于显示技术指标）
            buy_row = None
            sell_row = None

            buy_date_str = str(buy_date).split()[0]
            sell_date_str = str(sell_date).split()[0]

            buy_mask = (date_str_series.str.split().str[0] == buy_date_str)
            if buy_mask.any():
                buy_row = df.loc[buy_mask].iloc[0]

            sell_mask = (date_str_series.str.split().str[0] == sell_date_str)
            if sell_mask.any():
                sell_row = df.loc[sell_mask].iloc[0]

            if buy_row is not None:
                # 构建买入理由
                reason = []

                # 检查是否是RSI买入
                rsi_buy_type = buy_row.get('rsi_buy_type', '')
                if rsi_buy_type:
                    reason.append(rsi_buy_type)

                # 检查是否是DIFF顶背离失效
                if buy_row.get('diff_invalidation', 0) == 1:
                    reason.append("DIFF顶背离失效")

                if buy_row.get('bottom', 0) == 1:
                    reason.append("底部背离")
                if buy_row['close'] >= buy_row[f"{self.config['short_ma']}_ma"]:
                    reason.append(f"突破{self.config['short_ma']}日均线")
                if buy_row.get('k', 0) < self.config['k_threshold']:
                    reason.append(f"K值超卖({buy_row['k']:.1f})")
                if buy_row.get('macd', 0) > 0:
                    reason.append("MACD多头")

                # 直接使用trade中保存的买入手续费
                buy_commission = trade.get('buy_commission', 0)

                buy_points.append({
                    'date': buy_date,
                    'price': trade['buy_price'],  # 使用实际买入价（次日开盘价）
                    'reason': ', '.join(reason) if reason else '满足买入条件',
                    'k': buy_row.get('k', 0),
                    'd': buy_row.get('d', 0),
                    'macd': buy_row.get('macd', 0),
                    'ma_16': buy_row.get(f"{self.config['short_ma']}_ma", 0),
                    'commission': buy_commission,  # 买入手续费
                })

            if sell_row is not None:
                # 如果是最后一笔未平仓交易，需要特殊处理
                if is_last_open:
                    # 未平仓时，检查策略是否真正执行了卖出
                    # buy_signal == 1 表示持有状态，说明虽然可能检测到卖出信号，但策略没有执行
                    # buy_signal == 0 表示卖出状态，说明策略真正执行了卖出
                    is_holding = (sell_row['buy_signal'] == 1)

                    if is_holding:
                        # 策略判定为持有，显示持有理由和警告信息
                        hold_conditions = []  # 持有的积极理由
                        warning_conditions = []  # 风险警告
                        reason_explanation = []  # 策略未卖出的原因说明

                        # 积极的持有理由
                        if sell_row.get('top', 0) != 1:
                            hold_conditions.append("无顶部背离")

                        if sell_row['close'] >= sell_row[f"{self.config['short_ma']}_ma"]:
                            hold_conditions.append(f"站稳{self.config['short_ma']}日均线")

                        if sell_row.get('macd', 0) >= 0:
                            hold_conditions.append("MACD多头")

                        # 如果技术指标检测到顶背离但策略没卖，说明被保护逻辑屏蔽了
                        if sell_row.get('top', 0) == 1:
                            # 检查是否因为上升趋势保护
                            ma16 = sell_row.get('16_ma', 0)
                            ma45 = sell_row.get('45_ma', 0)
                            if ma16 > ma45:
                                hold_conditions.append("上升趋势保护(屏蔽顶背离)")

                        # 风险警告及原因说明（虽然策略判定持有，但存在风险信号）
                        ma16 = sell_row.get('16_ma', 0)
                        ma45 = sell_row.get('45_ma', 0)
                        close = sell_row.get('close', 0)

                        # MACD空头警告
                        if sell_row.get('macd', 0) < 0:
                            warning_conditions.append("⚠️MACD空头")
                            # 检查当前使用的策略
                            current_strategy = self.optimal_strategy if hasattr(self, 'optimal_strategy') and self.optimal_strategy else 'gradual'
                            if current_strategy == 'gradual':
                                reason_explanation.append("渐进式策略不以MACD作为卖出条件")

                        # 跌破均线警告
                        if close < sell_row[f"{self.config['short_ma']}_ma"]:
                            warning_conditions.append(f"⚠️跌破{self.config['short_ma']}日均线")
                            # 检查当前使用的策略和趋势
                            current_strategy = self.optimal_strategy if hasattr(self, 'optimal_strategy') and self.optimal_strategy else 'gradual'
                            if current_strategy == 'gradual':
                                if ma16 >= ma45:
                                    reason_explanation.append(f"上升趋势(MA16≥MA45), 需连续3天跌破才卖出")
                                else:
                                    # 下跌趋势应该立即卖出，但没卖，可能是数据边界情况
                                    reason_explanation.append("下跌趋势但未满足止损条件")

                        # 组合理由和警告
                        reason_parts = []
                        if hold_conditions:
                            reason_parts.append(', '.join(hold_conditions))
                        if warning_conditions:
                            warnings_str = ' | '.join(warning_conditions)
                            reason_parts.append(warnings_str)

                        # 如果有风险警告，添加原因说明
                        if reason_explanation:
                            reason = ', '.join(reason_parts) + ' 【原因: ' + '; '.join(reason_explanation) + '】(未平仓)'
                        else:
                            reason = (', '.join(reason_parts) if reason_parts else "持有中") + "(未平仓)"
                    else:
                        # 策略判定为卖出，显示卖出条件
                        sell_conditions = []

                        if sell_row.get('top', 0) == 1:
                            sell_conditions.append("顶部背离")

                        if sell_row['close'] < sell_row[f"{self.config['short_ma']}_ma"]:
                            sell_conditions.append(f"跌破{self.config['short_ma']}日均线")

                        if sell_row.get('macd', 0) < 0:
                            sell_conditions.append("MACD空头")

                        reason = ', '.join(sell_conditions) + "(未平仓)" if sell_conditions else "满足卖出条件(未平仓)"
                else:
                    # 已平仓的正常交易，显示真正导致卖出的条件
                    sell_conditions = []

                    if sell_row.get('top', 0) == 1:
                        sell_conditions.append("顶部背离")

                    if sell_row['close'] < sell_row[f"{self.config['short_ma']}_ma"]:
                        sell_conditions.append(f"跌破{self.config['short_ma']}日均线")

                    if sell_row.get('macd', 0) < 0:
                        sell_conditions.append("MACD空头")

                    reason = ', '.join(sell_conditions) if sell_conditions else '满足卖出条件'

                # 直接使用trade中保存的卖出手续费
                sell_commission = trade.get('sell_commission', 0)

                sell_points.append({
                    'date': sell_date,
                    'price': trade['sell_price'],  # 使用实际卖出价（次日开盘价或收盘价）
                    'reason': reason,
                    'k': sell_row.get('k', 0),
                    'd': sell_row.get('d', 0),
                    'macd': sell_row.get('macd', 0),
                    'ma_16': sell_row.get(f"{self.config['short_ma']}_ma", 0),
                    'is_open': is_last_open,  # 标记是否为未平仓
                    'commission': sell_commission,  # 卖出手续费
                })

        return {
            'buy_points': buy_points,
            'sell_points': sell_points,
            'total_trades': len(trades),
            'initial_capital': initial_capital,
            'trades': trades,  # 添加完整的trades数据，包含实际的交易后资金
            'gross_return': backtest_result.get('gross_return', 0),  # 添加毛收益率
        }
