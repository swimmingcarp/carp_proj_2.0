"""
Mixed Strategy - 核心交易策略
结合趋势跟随和顶底背离的混合策略
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from .indicators import calculate_all_indicators
from .divergence import (
    get_bottom_divergence_index,
    get_peak_divergence_index
)
from .indicator_validator import IndicatorValidator

logger = logging.getLogger(__name__)


class MixedStrategy:
    """混合交易策略"""

    def __init__(self, config: Dict = None, validate_indicators: bool = True, sell_strategy: str = 'auto', order: str = 'auto', market: str = 'CN-A', use_simple_divergence: bool = False):
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
        self.market = market
        self.use_simple_divergence = use_simple_divergence  # 是否使用简化版顶背离检测

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

        Returns:
            list: 保护期的日期索引列表
        """
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

        return list(set(protection_periods))  # 去重

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

        # 1. 跌停保护检查（只检查最近一段时间的数据，避免因历史跌停拒绝分析）
        df_temp = df.copy()
        df_temp['p_change_temp'] = df_temp['close'].pct_change() * 100

        # 只检查最近30天的数据
        recent_data = df_temp.tail(30) if len(df_temp) > 30 else df_temp

        if (recent_data['p_change_temp'] <= self.config['stop_loss']).any():
            # 找出最近的跌停日期
            drop_dates = recent_data[recent_data['p_change_temp'] <= self.config['stop_loss']]
            latest_drop = drop_dates.iloc[-1] if len(drop_dates) > 0 else None

            if latest_drop is not None:
                logger.warning(
                    f"近期存在跌停风险: {latest_drop['date']}, "
                    f"跌幅: {latest_drop['p_change_temp']:.2f}%, 跳过该股票"
                )
            return None, None

        # 2. 计算所有技术指标
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
        # 步骤1: 选择卖出策略 (original vs gradual)
        # 步骤2: 选择执行顺序 (high_frequency vs high_quality)

        # === 步骤1: 选择卖出策略 ===
        if self.sell_strategy == 'auto':
            logger.info("自适应模式：正在评估最优卖出策略...")

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
                        logger.info(f"选择渐进式策略（资金提升{improvement*100:.1f}%: "
                                  f"¥{backtest_orig['capital']:,.2f} → ¥{backtest_grad['capital']:,.2f}, "
                                  f"收益率{return_orig:.1f}% → {return_grad:.1f}%）")
                    else:
                        self.optimal_strategy = 'original'
                        logger.info(f"选择原始策略（渐进式提升不足30%: {improvement*100:.1f}%, "
                                  f"¥{backtest_orig['capital']:,.2f} vs ¥{backtest_grad['capital']:,.2f}）")
                else:
                    # 其他情况默认使用原始策略
                    self.optimal_strategy = 'original'
                    logger.info(f"选择原始策略（默认选择: "
                              f"¥{backtest_orig['capital']:,.2f}[{return_orig:.1f}%] vs "
                              f"¥{backtest_grad['capital']:,.2f}[{return_grad:.1f}%]）")
            else:
                self.optimal_strategy = 'original'
                logger.warning("回测失败，默认使用原始策略")
        else:
            # 使用指定的策略
            self.optimal_strategy = self.sell_strategy

        # === 步骤2: 选择执行顺序 ===
        if self.order_mode == 'auto':
            # 自适应选择执行顺序：在选定的策略基础上，比较high_frequency和high_quality
            logger.info(f"自适应模式：正在评估{self.optimal_strategy}策略的最优执行顺序...")

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
                    logger.info(f"✓ 高频率模式收益显著更高（+{improvement:.1f}%），直接选择高频率模式")
                else:
                    # HIGH_FREQUENCY优势不显著或HIGH_QUALITY更优，进行智能评分比较
                    self.optimal_order = self._select_better_order(backtest_new, backtest_old)
            else:
                self.optimal_order = 'high_frequency'
                logger.warning("执行顺序回测失败，默认使用高频率模式")
        else:
            # 使用指定的执行顺序（固定模式）
            self.optimal_order = self.order_mode
            logger.info(f"使用固定执行顺序: {self.optimal_order}")

        # 6. 应用选定的策略和执行顺序组合
        self._apply_combination(df, self.optimal_strategy, self.optimal_order, bottom_index, top_index)

        # 7. 计算持仓状态
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
                logger.info(f"  ⚠️ 交易次数增长{(trades_ratio-1)*100:.1f}%，"
                           f"但收益仅增长{(capital_ratio-1)*100:.1f}%")
                logger.info(f"  → 效率损失: {efficiency_gap*100:.1f}%，选择高质量模式")
                return 'high_quality'

        # 否则，选择收益更高的模式
        if capital_old > capital_new:
            improvement = (capital_old - capital_new) / capital_new * 100
            logger.info(f"✓ 选择高质量模式（收益更高，+{improvement:.1f}%）")
            return 'high_quality'
        else:
            logger.info(f"✓ 选择高频率模式（收益更高或相同）")
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

        规则1: RSI极度超卖 (<18) + 接近均线 + 不在跌停 = 抄底买入
        规则2: RSI短期金叉 (6日突破14日) + RSI < 40 + (MACD>0.1 或 涨幅>3%) = 反转买入

        Args:
            df: 数据框（会直接修改）
        """
        rsi_oversold = self.config.get('rsi_oversold', 18)
        rsi_threshold = self.config.get('rsi_threshold', 40)
        rsi_ma_ratio = self.config.get('rsi_ma_ratio', 0.95)
        rsi_min_strength = self.config.get('rsi_min_strength', True)
        rsi_min_gap = self.config.get('rsi_min_gap', 5)  # 最小间隔天数

        # 初始化RSI买入类型标记列
        df['rsi_buy_type'] = ''

        last_rsi_buy_idx = -999  # 上次RSI买入的位置

        for i in range(1, len(df)):
            idx = df.index[i]
            prev_idx = df.index[i-1]

            # 检查是否距离上次RSI买入太近
            if i - last_rsi_buy_idx < rsi_min_gap:
                continue  # 跳过，避免频繁交易

            rsi = df.loc[idx, 'rsi']
            rsi_6 = df.loc[idx, 'rsi_6']
            prev_rsi_6 = df.loc[prev_idx, 'rsi_6']
            prev_rsi = df.loc[prev_idx, 'rsi']
            close = df.loc[idx, 'close']
            ma_16 = df.loc[idx, f"{self.config['short_ma']}_ma"]
            p_change = df.loc[idx, 'p_change']
            macd = df.loc[idx, 'macd']

            # 规则1：RSI极度超卖（< 18），接近均线，且不在跌停
            if rsi < rsi_oversold and close >= ma_16 * rsi_ma_ratio and p_change > -8:
                # 检查是否在主升浪保护期内
                current_date = df.loc[idx, 'date']
                if protection_periods is None or current_date not in protection_periods:
                    df.loc[idx, 'buy_signal'] = 1
                    df.loc[idx, 'rsi_buy_type'] = 'RSI超卖'
                    last_rsi_buy_idx = i
                    logger.debug(f"RSI超卖买入: {current_date}, RSI={rsi:.1f}")
                else:
                    logger.debug(f"RSI超卖买入被主升浪保护期阻止: {current_date}")
                logger.debug(f"RSI超卖买入: {df.loc[idx, 'date']}, RSI={rsi:.1f}")

            # 规则2：RSI短期金叉，且在极低位，并满足更严格的确认条件
            elif rsi_6 > rsi and prev_rsi_6 <= prev_rsi and rsi < rsi_threshold:
                confirmed = False

                if rsi_min_strength:
                    # 条件A：MACD > 0.1（趋势明显向上，不是弱势多头）
                    if macd > 0.1:
                        confirmed = True
                        logger.debug(f"RSI金叉+强MACD: {df.loc[idx, 'date']}")

                    # 条件B：价格强势反弹 > 3%（改进：从2%提高到3%）
                    elif p_change > 3:
                        confirmed = True
                        logger.debug(f"RSI金叉+强反弹: {df.loc[idx, 'date']}, 涨幅={p_change:.1f}%")

                    # 条件C：接近均线且RSI < 35（极低位金叉，从40降到35）
                    elif close >= ma_16 * 0.98 and rsi < 35:
                        confirmed = True
                        logger.debug(f"RSI极低位金叉: {df.loc[idx, 'date']}, RSI={rsi:.1f}")
                else:
                    if close >= ma_16 * 0.98:
                        confirmed = True

                if confirmed:
                    # 检查是否在主升浪保护期内
                    current_date = df.loc[idx, 'date']
                    if protection_periods is None or current_date not in protection_periods:
                        df.loc[idx, 'buy_signal'] = 1
                        df.loc[idx, 'rsi_buy_type'] = 'RSI金叉'
                        last_rsi_buy_idx = i
                        logger.debug(f"RSI金叉买入: {current_date}, RSI_6={rsi_6:.1f}, RSI={rsi:.1f}")
                    else:
                        logger.debug(f"RSI金叉买入被主升浪保护期阻止: {current_date}")
                    logger.debug(f"RSI金叉买入: {df.loc[idx, 'date']}, RSI_6={rsi_6:.1f}, RSI={rsi:.1f}")

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

        # 检测买入信号点
        buy_signal_dates = df[df['buy_signal'] == 1].index

        # 创建一个新列记录每日的资金
        capital = initial_capital
        capital_list = []
        trades = []
        holding = False
        buy_price = 0
        buy_date = None
        shares = 0  # 持有股数
        total_commission = 0.0  # 累计手续费

        for i in range(len(df)):
            row = df.iloc[i]

            # 当日有买入信号，当日收盘买入
            if row['buy_signal'] == 1 and not holding:
                buy_price = row['close']
                buy_date = row['date']

                # 计算可买股数
                shares = capital / buy_price
                transaction_amount = shares * buy_price

                # 计算并扣除买入手续费
                buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                capital -= buy_commission
                total_commission += buy_commission

                holding = True
                logger.debug(f"买入: {row['date']}, 价格: {buy_price:.2f}, 股数: {shares:.2f}, 手续费: {buy_commission:.2f}")

            # 当日有卖出信号（buy_signal=0），当日收盘卖出
            elif row['buy_signal'] == 0 and holding:
                sell_price = row['close']
                transaction_amount = shares * sell_price

                # 计算并扣除卖出手续费
                sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
                total_commission += sell_commission

                # 更新资金：卖出所得 - 手续费
                capital = transaction_amount - sell_commission

                # 计算收益率（基于初始买入资金）
                initial_investment = shares * buy_price
                profit = capital - (initial_capital - initial_investment + buy_commission)
                profit_rate = profit / initial_investment if initial_investment > 0 else 0

                trades.append({
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': row['date'],
                    'sell_price': sell_price,
                    'profit_rate': profit_rate,
                    'capital': capital,
                    'commission': buy_commission + sell_commission,  # 本次交易总手续费
                    'buy_commission': buy_commission,  # 买入手续费
                    'sell_commission': sell_commission,  # 卖出手续费
                })

                holding = False
                shares = 0
                logger.debug(f"卖出: {row['date']}, 价格: {sell_price:.2f}, 收益率: {profit_rate*100:.2f}%, 手续费: {sell_commission:.2f}")

            capital_list.append(capital)

        # 如果最后还持仓，用最后一天的收盘价计算
        if holding:
            last_row = df.iloc[-1]
            sell_price = last_row['close']
            transaction_amount = shares * sell_price

            # 计算卖出手续费（用于净值计算）
            sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
            total_commission += sell_commission

            # 最终资金 = 卖出所得 - 手续费
            capital = transaction_amount - sell_commission

            # 计算收益率
            initial_investment = shares * buy_price
            profit = capital - (initial_capital - initial_investment)
            profit_rate = profit / initial_investment if initial_investment > 0 else 0

            trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': last_row['date'],
                'sell_price': sell_price,
                'profit_rate': profit_rate,
                'capital': capital,
                'commission': sell_commission,  # 未平仓只计算卖出手续费
                'buy_commission': 0,  # 未平仓暂无买入手续费记录
                'sell_commission': sell_commission,  # 卖出手续费
            })

            # 更新最后一天的资金
            capital_list[-1] = capital

        df['capital'] = capital_list

        # 计算毛收益率（不扣手续费的理论收益）
        # 需要重新运行一次回测，但关闭手续费
        commission_enabled_backup = self.config.get('commission_enabled', True)
        self.config['commission_enabled'] = False  # 临时关闭手续费

        # 运行不含手续费的回测
        backtest_no_fee = self._quick_backtest(df)
        if backtest_no_fee is not None and isinstance(backtest_no_fee, dict):
            gross_capital = backtest_no_fee['capital']
            gross_return = (gross_capital - initial_capital) / initial_capital * 100
        else:
            gross_return = total_return

        # 恢复手续费设置
        self.config['commission_enabled'] = commission_enabled_backup

        # 计算指标
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

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
            if i == len(trades) - 1 and sell_date == df.iloc[-1]['date']:
                # 检查最后一天是否有真正的卖出信号
                last_day_signal = df.iloc[-1]['buy_signal']
                # 如果最后一天 buy_signal=1，说明是持仓状态（未卖出）
                # 如果最后一天 buy_signal=0，说明当天已经卖出了
                is_last_open = (last_day_signal == 1)

            # 获取买入日期的行信息（用于显示技术指标）
            buy_row = df[df['date'] == buy_date].iloc[0] if len(df[df['date'] == buy_date]) > 0 else None
            sell_row = df[df['date'] == sell_date].iloc[0] if len(df[df['date'] == sell_date]) > 0 else None

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
