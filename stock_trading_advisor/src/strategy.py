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
    get_peak_divergence_index,
    get_peak_divergence_index_kdj,
    get_peak_divergence_index_kd_variant
)
from .indicator_validator import IndicatorValidator

logger = logging.getLogger(__name__)


class MixedStrategy:
    """混合交易策略"""

    def __init__(self, config: Dict = None, validate_indicators: bool = True, sell_strategy: str = 'auto', market: str = 'CN-A'):
        """
        初始化策略

        Args:
            config: 策略配置参数
            validate_indicators: 是否启用技术指标验证
            sell_strategy: 卖出策略类型
                - 'auto': 自动选择（对每只股票先回测，选择最优策略）
                - 'original': 原始策略（跌破MA16或MACD<0）
                - 'gradual': 渐进式止损（连续3天跌破MA16）
            market: 市场类型 ('CN-A'-A股, 'HK'-港股, 'US'-美股)，用于选择正确的手续费率
        """
        self.config = self._default_config()
        if config:
            # 合并用户配置，保留默认配置中用户未指定的参数
            self.config.update(config)
        self.validate_indicators = validate_indicators
        self.sell_strategy = sell_strategy
        self.optimal_strategy = None  # 记录为当前股票选择的最优策略
        self.market = market

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
        df['top'] = 0
        df['bottom'] = 0

        # 初始化背离索引
        top_index = []
        bottom_index = []

        try:
            # 顶部背离（三种方法的并集）
            top_divergence_kdj = get_peak_divergence_index_kdj(df, self.config['lookback_days'])
            top_divergence_diff = get_peak_divergence_index(df, self.config['lookback_days'])
            top_divergence_kd = get_peak_divergence_index_kd_variant(df, self.config['lookback_days'])

            top_index = list(set(top_divergence_kdj) | set(top_divergence_diff) | set(top_divergence_kd))
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
        buy_index = df[df['close'] >= df[f"{self.config['short_ma']}_ma"]].index
        df.loc[buy_index, 'buy_signal'] = 1

        # 5. 生成卖出信号（自适应策略选择）
        # 如果是'auto'模式，先运行两种策略的回测，选择最优的
        if self.sell_strategy == 'auto':
            # 快速测试原始策略和渐进式策略
            logger.info("自适应模式：正在评估最优卖出策略...")

            # 测试原始策略
            df_test_orig = df.copy()
            self._apply_sell_strategy(df_test_orig, 'original', top_index)
            backtest_orig = self._quick_backtest(df_test_orig)

            # 测试渐进式策略
            df_test_grad = df.copy()
            self._apply_sell_strategy(df_test_grad, 'gradual', top_index)
            backtest_grad = self._quick_backtest(df_test_grad)

            # 选择最优策略：如果渐进式收益比原始高30%以上，使用渐进式；否则使用原始
            if backtest_grad and backtest_orig:
                improvement = (backtest_grad - backtest_orig) / backtest_orig if backtest_orig != 0 else 0
                if improvement > 0.30:  # 提升超过30%
                    self.optimal_strategy = 'gradual'
                    logger.info(f"选择渐进式策略（提升{improvement*100:.1f}%: {backtest_orig:.1f}% → {backtest_grad:.1f}%）")
                else:
                    self.optimal_strategy = 'original'
                    logger.info(f"选择原始策略（渐进提升不足30%: {improvement*100:.1f}%）")
            else:
                self.optimal_strategy = 'original'
                logger.warning("回测失败，默认使用原始策略")
        else:
            # 使用指定的策略
            self.optimal_strategy = self.sell_strategy

        # 应用选定的策略
        self._apply_sell_strategy(df, self.optimal_strategy, top_index)

        # 6. 底部背离次日强制买入
        bottom_shift_index = df[df['bottom'].shift(1) == 1].index
        df.loc[bottom_shift_index, 'buy_signal'] = 1

        # 7. 顶部背离期间阻止买入
        block_index, diff_invalidation_dates = self._calculate_block_index(df, top_index)
        df.loc[block_index, 'buy_signal'] = 0

        # 标记DIFF顶背离失效的日期
        df['diff_invalidation'] = 0
        if len(diff_invalidation_dates) > 0:
            df.loc[diff_invalidation_dates, 'diff_invalidation'] = 1

        # 8. RSI增强买入（可选）
        if self.config.get('rsi_enabled', False):
            self._apply_rsi_enhancements(df)

        # 9. 计算持仓状态
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
            # 渐进式止损：连续3天跌破MA16才卖出
            for i in range(2, len(df)):
                idx = df.index[i]
                idx_1 = df.index[i-1]
                idx_2 = df.index[i-2]

                below_ma_0 = df.loc[idx, 'close'] < df.loc[idx, f"{self.config['short_ma']}_ma"]
                below_ma_1 = df.loc[idx_1, 'close'] < df.loc[idx_1, f"{self.config['short_ma']}_ma"]
                below_ma_2 = df.loc[idx_2, 'close'] < df.loc[idx_2, f"{self.config['short_ma']}_ma"]

                if below_ma_0 and below_ma_1 and below_ma_2:
                    sell_index.add(idx)

        # 顶部背离强制卖出（所有策略共用）
        sell_index = sell_index.union(set(top_index))

        # 应用卖出信号
        df.loc[list(sell_index), 'buy_signal'] = 0

    def _quick_backtest(self, df: pd.DataFrame) -> float:
        """
        快速回测，只返回收益率（用于自适应策略选择）

        Args:
            df: 包含buy_signal的数据框

        Returns:
            总收益率（百分比），如果回测失败返回None
        """
        try:
            capital = 10000.0
            holding = False
            buy_price = 0
            shares = 0

            for i in range(len(df)):
                if i > 0:
                    prev_signal = df.iloc[i-1]['buy_signal']
                    curr_price = df.iloc[i]['open']

                    # 买入
                    if prev_signal == 1 and not holding:
                        buy_price = curr_price
                        shares = capital / buy_price  # 计算可买股数
                        transaction_amount = shares * buy_price

                        # 扣除买入手续费
                        buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                        capital -= buy_commission

                        holding = True

                    # 卖出
                    elif prev_signal == 0 and holding:
                        sell_price = curr_price
                        transaction_amount = shares * sell_price

                        # 扣除卖出手续费
                        sell_commission = self._calculate_commission(transaction_amount, is_buy=False)

                        # 更新资金：卖出所得 - 手续费
                        capital = transaction_amount - sell_commission
                        holding = False
                        shares = 0

            # 最后还持仓，用收盘价计算
            if holding:
                sell_price = df.iloc[-1]['close']
                transaction_amount = shares * sell_price
                sell_commission = self._calculate_commission(transaction_amount, is_buy=False)
                capital = transaction_amount - sell_commission

            return (capital / 10000.0 - 1) * 100  # 返回收益率百分比

        except Exception as e:
            logger.error(f"快速回测失败: {e}")
            return None

    def _apply_rsi_enhancements(self, df: pd.DataFrame) -> None:
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
                df.loc[idx, 'buy_signal'] = 1
                df.loc[idx, 'rsi_buy_type'] = 'RSI超卖'
                last_rsi_buy_idx = i
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
                    df.loc[idx, 'buy_signal'] = 1
                    df.loc[idx, 'rsi_buy_type'] = 'RSI金叉'
                    last_rsi_buy_idx = i
                    logger.debug(f"RSI金叉买入: {df.loc[idx, 'date']}, RSI_6={rsi_6:.1f}, RSI={rsi:.1f}")

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
            if latest['k'] < self.config['k_threshold']:
                reason.append(f"K值处于超卖区域({latest['k']:.1f})")
                strength += 1

            if latest[f"{self.config['mid_ma']}_ma"] > previous[f"{self.config['mid_ma']}_ma"]:
                reason.append("中期趋势向上")
                strength += 1

            if latest['bottom'] == 1.0 or previous['bottom'] == 1.0:
                reason.append("底部背离信号")
                strength += 2

            if latest['macd'] > 0:
                reason.append("MACD在零轴上方")
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

        正确的交易逻辑：
        - 看到买入信号(buy_signal=1)后，次日开盘买入
        - 看到卖出信号(buy_signal=0)后，次日开盘卖出
        - 收益 = (卖出开盘价 - 买入开盘价) / 买入开盘价
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

            # 检查前一天是否有买入信号
            if i > 0:
                prev_row = df.iloc[i - 1]

                # 前一天有买入信号，今天开盘买入
                if prev_row['buy_signal'] == 1 and not holding:
                    buy_price = row['open']
                    buy_date = prev_row['date']

                    # 计算可买股数
                    shares = capital / buy_price
                    transaction_amount = shares * buy_price

                    # 计算并扣除买入手续费
                    buy_commission = self._calculate_commission(transaction_amount, is_buy=True)
                    capital -= buy_commission
                    total_commission += buy_commission

                    holding = True
                    logger.debug(f"买入: {row['date']}, 价格: {buy_price:.2f}, 股数: {shares:.2f}, 手续费: {buy_commission:.2f}")

                # 前一天有卖出信号（buy_signal=0），今天开盘卖出
                elif prev_row['buy_signal'] == 0 and holding:
                    sell_price = row['open']
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
                        'sell_date': prev_row['date'],
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
        gross_return = backtest_no_fee if backtest_no_fee is not None else total_return

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
            is_last_open = (i == len(trades) - 1 and sell_date == df.iloc[-1]['date'])

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
                # 构建卖出理由（无论是否平仓都要检测）
                sell_conditions = []  # 满足的卖出条件
                hold_conditions = []  # 持有的理由（未满足卖出条件）

                # 检测所有可能的卖出条件
                if sell_row.get('top', 0) == 1:
                    sell_conditions.append("顶部背离")
                else:
                    hold_conditions.append("无顶部背离")

                if sell_row['close'] < sell_row[f"{self.config['short_ma']}_ma"]:
                    sell_conditions.append(f"跌破{self.config['short_ma']}日均线")
                else:
                    hold_conditions.append(f"站稳{self.config['short_ma']}日均线")

                if sell_row.get('macd', 0) < 0:
                    sell_conditions.append("MACD空头")
                else:
                    hold_conditions.append("MACD多头")

                # 如果是最后一笔未平仓交易，添加标注
                if is_last_open:
                    if sell_conditions:
                        # 有卖出条件
                        reason = ', '.join(sell_conditions) + "(未平仓)"
                    else:
                        # 无卖出条件，显示持有理由
                        reason = ', '.join(hold_conditions) + "(未平仓)"
                else:
                    # 已平仓的正常交易
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
