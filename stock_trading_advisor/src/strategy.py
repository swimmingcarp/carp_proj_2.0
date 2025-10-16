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

    def __init__(self, config: Dict = None, validate_indicators: bool = True):
        """
        初始化策略

        Args:
            config: 策略配置参数
            validate_indicators: 是否启用技术指标验证
        """
        self.config = config or self._default_config()
        self.validate_indicators = validate_indicators

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
        }

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
                rsi_fast_period=self.config.get('rsi_fast_period', 6),
                rsi_slow_period=self.config.get('rsi_slow_period', 14)
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

        # 5. 生成卖出信号
        # 条件：跌破16日均线 OR MACD < 0 OR 顶部背离
        sell_index1 = df[df['close'] < df[f"{self.config['short_ma']}_ma"]].index
        sell_index2 = df[df['macd'] < 0].index
        sell_index = sell_index1.union(sell_index2).union(top_index)

        df.loc[sell_index, 'buy_signal'] = 0

        # 6. 底部背离次日强制买入
        bottom_shift_index = df[df['bottom'].shift(1) == 1].index
        df.loc[bottom_shift_index, 'buy_signal'] = 1

        # 7. 顶部背离期间阻止买入
        block_index = self._calculate_block_index(df, top_index)
        df.loc[block_index, 'buy_signal'] = 0

        # 8. RSI增强买入（可选）
        if self.config.get('rsi_enabled', False):
            self._apply_rsi_enhancements(df)

        # 9. 计算持仓状态
        df['position'] = df['buy_signal'].shift(1)
        df['position'] = df['position'].ffill()
        df.loc[:self.config['init_date'], 'position'] = 0

        return df, indicator_report

    def _calculate_block_index(self, df: pd.DataFrame, top_index: list) -> list:
        """
        计算顶部背离期间的买入阻止区域

        Args:
            df: 数据框
            top_index: 顶部背离日期列表

        Returns:
            阻止买入的日期列表
        """
        block_index = set()
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
                    break
                else:
                    block_index.add(date)

        return list(block_index)

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
        回测策略

        正确的交易逻辑：
        - 看到买入信号(buy_signal=1)后，次日开盘买入
        - 看到卖出信号(buy_signal=0)后，次日开盘卖出
        - 收益 = (卖出开盘价 - 买入开盘价) / 买入开盘价

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

        for i in range(len(df)):
            row = df.iloc[i]

            # 检查前一天是否有买入信号
            if i > 0:
                prev_row = df.iloc[i - 1]

                # 前一天有买入信号，今天开盘买入
                if prev_row['buy_signal'] == 1 and not holding:
                    buy_price = row['open']
                    buy_date = prev_row['date']
                    holding = True
                    logger.debug(f"买入: {row['date']}, 价格: {buy_price:.2f}")

                # 前一天有卖出信号（buy_signal=0），今天开盘卖出
                elif prev_row['buy_signal'] == 0 and holding:
                    sell_price = row['open']
                    profit_rate = (sell_price - buy_price) / buy_price
                    capital = capital * (1 + profit_rate)

                    trades.append({
                        'buy_date': buy_date,
                        'buy_price': buy_price,
                        'sell_date': prev_row['date'],
                        'sell_price': sell_price,
                        'profit_rate': profit_rate,
                        'capital': capital
                    })

                    holding = False
                    logger.debug(f"卖出: {row['date']}, 价格: {sell_price:.2f}, 收益率: {profit_rate*100:.2f}%")

            capital_list.append(capital)

        # 如果最后还持仓，用最后一天的收盘价计算
        if holding:
            last_row = df.iloc[-1]
            sell_price = last_row['close']
            profit_rate = (sell_price - buy_price) / buy_price
            capital = capital * (1 + profit_rate)

            trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': last_row['date'],
                'sell_price': sell_price,
                'profit_rate': profit_rate,
                'capital': capital
            })

        df['capital'] = capital_list

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
                if buy_row.get('bottom', 0) == 1:
                    reason.append("底部背离")
                if buy_row['close'] >= buy_row[f"{self.config['short_ma']}_ma"]:
                    reason.append(f"突破{self.config['short_ma']}日均线")
                if buy_row.get('k', 0) < self.config['k_threshold']:
                    reason.append(f"K值超卖({buy_row['k']:.1f})")
                if buy_row.get('macd', 0) > 0:
                    reason.append("MACD多头")

                buy_points.append({
                    'date': buy_date,
                    'price': trade['buy_price'],  # 使用实际买入价（次日开盘价）
                    'reason': ', '.join(reason) if reason else '满足买入条件',
                    'k': buy_row.get('k', 0),
                    'd': buy_row.get('d', 0),
                    'macd': buy_row.get('macd', 0),
                    'ma_16': buy_row.get(f"{self.config['short_ma']}_ma", 0),
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

                sell_points.append({
                    'date': sell_date,
                    'price': trade['sell_price'],  # 使用实际卖出价（次日开盘价或收盘价）
                    'reason': reason,
                    'k': sell_row.get('k', 0),
                    'd': sell_row.get('d', 0),
                    'macd': sell_row.get('macd', 0),
                    'ma_16': sell_row.get(f"{self.config['short_ma']}_ma", 0),
                    'is_open': is_last_open,  # 标记是否为未平仓
                })

        return {
            'buy_points': buy_points,
            'sell_points': sell_points,
            'total_trades': len(trades),
            'initial_capital': initial_capital
        }
