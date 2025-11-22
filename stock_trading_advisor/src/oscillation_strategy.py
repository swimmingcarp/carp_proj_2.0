"""
Oscillation Strategy - 震荡周期间隔交易策略
只在震荡区间外的间隔期间交易一笔
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .indicators import calculate_all_indicators
    from .strategy import MixedStrategy
except ImportError:
    from indicators import calculate_all_indicators
    from strategy import MixedStrategy

logger = logging.getLogger(__name__)


class OscillationStrategy:
    """
    震荡周期间隔交易策略
    
    核心逻辑：
    1. 检测所有震荡周期（使用逐日判断，无前瞻性偏差）
    2. 识别震荡区间之外的间隔区间（正常行情）
    3. 在每个间隔区间内只交易一笔（一买一卖）
    4. 震荡期间不交易
    """
    
    def __init__(self, config: Dict = None, market: str = 'CN-A'):
        """
        初始化策略
        
        Args:
            config: 策略配置参数
            market: 市场类型 ('CN-A'-A股, 'HK'-港股, 'US'-美股)
        """
        self.config = self._default_config()
        if config:
            self.config.update(config)
        self.market = market
        
        # 初始化持仓状态
        self._has_open_position = False
        self._buy_price = None
        
        # 使用 MixedStrategy 来检测震荡周期和计算指标
        self._mixed_strategy = MixedStrategy(
            config=config,
            market=market,
            validate_indicators=False
        )
    
    def _default_config(self) -> Dict:
        """默认配置参数"""
        return {
            'short_ma': 16,              # 短期均线（买卖信号）
            'commission_rate': 0.0003,   # 佣金率
            'min_commission': 5.0,       # 最低佣金
            'stamp_duty_cn': 0.0005,     # A股印花税（卖出时）
            'oscillation_detection_enabled': True,
        }
    
    def analyze(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        执行策略分析
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            
        Returns:
            (添加了买卖信号的 DataFrame, None)
        """
        if df is None or len(df) == 0:
            logger.warning("数据为空，无法分析")
            return None, None
        
        # 1. 计算技术指标
        # 需要传递 init_k 和 init_d 才能计算KDJ
        init_k = self.config.get('init_k', 50)
        init_d = self.config.get('init_d', 50)
        init_date = self.config.get('init_date', '2018-01-02')
        df = calculate_all_indicators(df, init_k=init_k, init_d=init_d, init_date=init_date)
        
        # 2. 初始化信号列
        df['buy_signal'] = 0
        df['sell_signal'] = 0  # 新增：明确标记卖出信号
        df['oscillation_state'] = 0
        df['trading_window'] = 0  # 0=禁止交易（震荡中），1=允许交易（间隔期）
        
        # 3. 检测震荡周期（逐日判断）
        oscillation_periods = self._detect_oscillation_periods(df)
        
        # 保存震荡周期以供绘图使用
        self._oscillation_periods = oscillation_periods
        
        if not oscillation_periods:
            logger.warning("⚠️ 未检测到震荡周期，将允许全程交易")
            # 如果没有震荡周期，整个时期都是一个间隔区间
            self._trade_in_interval(df, 0, len(df) - 1, 1)
            return df, None
        
        logger.info(f"🔍 检测到 {len(oscillation_periods)} 个震荡周期")
        
        # 4. 标记震荡区间
        for confirmed_idx, end_idx, confirmed_date, end_date in oscillation_periods:
            df.iloc[confirmed_idx:end_idx + 1, df.columns.get_loc('oscillation_state')] = 1
        
        # 5. 识别并交易间隔区间
        self._identify_and_trade_intervals(df, oscillation_periods)
        
        # 6. 计算持仓状态（根据买卖信号）
        df['position'] = 0
        current_position = 0
        for idx in range(len(df)):
            if df.iloc[idx]['buy_signal'] == 1:
                current_position = 1
            elif df.iloc[idx]['sell_signal'] == 1:
                current_position = 0
            df.iloc[idx, df.columns.get_loc('position')] = current_position
        
        return df, None
    
    def _detect_oscillation_periods(self, df: pd.DataFrame) -> List[Tuple]:
        """
        检测震荡周期（逐日判断，无前瞻性偏差）
        
        Returns:
            list: [(confirmed_idx, end_idx, confirmed_date, end_date), ...]
        """
        # 参数配置
        window_size = max(20, self.config.get('oscillation_window_size', 30))
        algorithm_score_threshold = 4.0
        confirm_days = max(2, self.config.get('oscillation_confirm_days', 5))
        release_days = max(3, confirm_days)
        
        outer_upper = 0.30
        inner_upper = 0.20
        
        n = len(df)
        if n < window_size + confirm_days:
            return []
        
        # 准备数据
        close_arr = df['close'].to_numpy()
        width_arr = df['bb_width'].to_numpy() if 'bb_width' in df.columns else np.full(n, np.nan)
        pos_arr = df['bb_percent'].to_numpy() if 'bb_percent' in df.columns else np.full(n, np.nan)
        
        ma_arrays = {}
        for p in [5, 10, 20, 30]:
            col = f'{p}_ma'
            if col in df.columns:
                ma_arrays[p] = df[col].to_numpy()
        
        # 逐日判断震荡状态
        pending_high_scores = []
        active_oscillation = False
        below_counter = 0
        oscillation_periods = []
        period_start_idx = None
        period_confirmed_idx = None
        
        for idx in range(window_size - 1, n):
            # 只使用历史数据窗口
            window_start = idx - window_size + 1
            segment = close_arr[window_start:idx + 1]
            
            if len(segment) < max(window_size // 2, confirm_days):
                continue
            
            # 计算震荡得分
            score = 0.0
            
            # 1. 布林带分析
            bb_width_current = width_arr[idx]
            bb_position_current = pos_arr[idx]
            if not np.isnan(bb_width_current):
                if bb_width_current < 0.25:
                    score += 1
                    if bb_width_current < 0.15:
                        score += 1
                if not np.isnan(bb_position_current) and 0.1 <= bb_position_current <= 0.9:
                    score += 0.5
            
            # 2. 均线纠缠分析
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
            
            # 3. 趋势强度分析
            first_close = segment[0]
            last_close = segment[-1]
            if first_close != 0:
                trend_strength = abs((last_close / first_close) - 1)
                if trend_strength < 0.20:
                    score += 1
                    if trend_strength < 0.08:
                        score += 1
            
            # 4. 价格震荡范围
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
            
            # 更新待确认队列
            if is_high_score:
                pending_high_scores.append((idx, score))
                if len(pending_high_scores) > confirm_days:
                    pending_high_scores = pending_high_scores[-confirm_days:]
            else:
                pending_high_scores.clear()
            
            # 判断是否进入震荡状态
            if not active_oscillation and len(pending_high_scores) >= confirm_days:
                active_oscillation = True
                period_start_idx = pending_high_scores[0][0]
                period_confirmed_idx = idx
                below_counter = 0
            
            # 更新震荡状态
            if active_oscillation:
                if is_high_score:
                    below_counter = 0
                else:
                    below_counter += 1
                    if below_counter >= release_days:
                        # 退出震荡状态
                        period_end_idx = idx - release_days
                        if period_end_idx >= period_start_idx and period_confirmed_idx is not None:
                            if 'date' in df.columns:
                                confirmed_date = df.iloc[period_confirmed_idx]['date']
                                end_date = df.iloc[period_end_idx]['date']
                            else:
                                confirmed_date = period_confirmed_idx
                                end_date = period_end_idx
                            
                            oscillation_periods.append((
                                period_confirmed_idx,
                                period_end_idx,
                                confirmed_date,
                                end_date
                            ))
                        
                        active_oscillation = False
                        below_counter = 0
                        pending_high_scores.clear()
                        period_start_idx = None
                        period_confirmed_idx = None
        
        # 处理最后仍处于震荡状态的情况
        if active_oscillation and period_start_idx is not None and period_confirmed_idx is not None:
            period_end_idx = n - 1
            if 'date' in df.columns:
                confirmed_date = df.iloc[period_confirmed_idx]['date']
                end_date = df.iloc[period_end_idx]['date']
            else:
                confirmed_date = period_confirmed_idx
                end_date = period_end_idx
            
            oscillation_periods.append((
                period_confirmed_idx,
                period_end_idx,
                confirmed_date,
                end_date
            ))
        
        return oscillation_periods
    
    def _identify_and_trade_intervals(self, df: pd.DataFrame, oscillation_periods: List[Tuple]) -> None:
        """
        识别震荡区间之外的间隔区间，并在每个间隔内交易一笔
        
        Args:
            df: 数据DataFrame
            oscillation_periods: 震荡周期列表 [(confirmed_idx, end_idx, confirmed_date, end_date), ...]
        """
        n = len(df)
        
        # 用于跟踪是否有未平仓持仓
        self._has_open_position = False
        self._buy_price = None  # 记录买入价格（用于震荡期止损判断）
        
        # 构建间隔区间列表
        intervals = []
        
        # 第一个间隔：从数据开始到第一个震荡周期确认点
        if oscillation_periods:
            first_osc_start = oscillation_periods[0][0]  # 震荡确认点
            if first_osc_start > 0:
                intervals.append((0, first_osc_start - 1, "间隔1（数据开始）"))
        
        # 中间的间隔：两个震荡周期之间
        for i in range(len(oscillation_periods) - 1):
            current_osc_end = oscillation_periods[i][1]  # 当前震荡结束
            current_osc_end_date = oscillation_periods[i][3]  # 震荡结束日期
            next_osc_start = oscillation_periods[i + 1][0]  # 下一个震荡确认
            
            interval_start = current_osc_end + 1
            interval_end = next_osc_start - 1
            
            if interval_start <= interval_end:
                intervals.append((interval_start, interval_end, f"间隔{i + 2}（震荡{i+1}结束于{current_osc_end_date}后）"))
        
        # 最后一个间隔：从最后震荡结束到数据末尾
        if oscillation_periods:
            last_osc_end = oscillation_periods[-1][1]
            if last_osc_end < n - 1:
                intervals.append((last_osc_end + 1, n - 1, f"间隔{len(intervals) + 1}（数据结束）"))
        
        logger.info(f"📊 识别到 {len(intervals)} 个间隔区间（震荡外的正常行情）")
        
        # 在每个间隔区间内交易一笔
        total_trades = 0
        for interval_idx, (start_idx, end_idx, interval_name) in enumerate(intervals):
            trades = self._trade_in_interval(df, start_idx, end_idx, interval_name)
            if trades > 0:
                total_trades += trades
            
            # 检查间隔结束后是否进入震荡期，如果有持仓则在震荡期内允许卖出
            if self._has_open_position and interval_idx < len(oscillation_periods):
                # 获取下一个震荡期
                osc_confirmed_idx = oscillation_periods[interval_idx][0]
                osc_end_idx = oscillation_periods[interval_idx][1]
                osc_end_date = oscillation_periods[interval_idx][3]
                
                # 在震荡期内尝试卖出
                sell_trades = self._sell_in_oscillation(df, osc_confirmed_idx, osc_end_idx, 
                                                        f"震荡{interval_idx + 1}（结束于{osc_end_date}）")
                if sell_trades > 0:
                    total_trades += sell_trades
        
        logger.info(f"✅ 间隔交易策略：完成 {total_trades} 笔交易")
    
    def _trade_in_interval(self, df: pd.DataFrame, start_idx: int, end_idx: int, interval_name: str) -> int:
        """
        在指定的间隔区间内交易一笔
        
        Args:
            df: 数据DataFrame
            start_idx: 间隔起始索引
            end_idx: 间隔结束索引
            interval_name: 间隔名称（用于日志）
            
        Returns:
            完成的交易数量（0或1）
        """
        if start_idx >= len(df) or end_idx >= len(df) or start_idx > end_idx:
            return 0
        
        # 标记交易窗口
        df.iloc[start_idx:end_idx + 1, df.columns.get_loc('trading_window')] = 1
        
        # 检查是否有未平仓的持仓（从上一个间隔继承）
        if self._has_open_position:
            # 已有持仓，直接寻找卖出信号（使用渐进式止损策略）
            sell_idx = None
            for idx in range(start_idx, end_idx + 1):
                # 检查当前趋势
                ma16 = df.iloc[idx]['16_ma']
                ma45 = df.iloc[idx]['45_ma']
                close = df.iloc[idx]['close']
                short_ma = df.iloc[idx][f"{self.config['short_ma']}_ma"]
                
                if ma16 < ma45:
                    # 下跌趋势：使用传统止损（跌破MA16立即止损）
                    if close < short_ma:
                        sell_idx = idx
                        break
                else:
                    # 上升/横盘趋势：使用渐进式止损（连续3天跌破MA16才止损）
                    # 检查是否有足够的历史数据
                    if idx >= start_idx + 2:
                        below_ma_0 = close < short_ma
                        below_ma_1 = df.iloc[idx - 1]['close'] < df.iloc[idx - 1][f"{self.config['short_ma']}_ma"]
                        below_ma_2 = df.iloc[idx - 2]['close'] < df.iloc[idx - 2][f"{self.config['short_ma']}_ma"]
                        
                        # 严格要求：只有连续3天都跌破才卖出
                        if below_ma_0 and below_ma_1 and below_ma_2:
                            sell_idx = idx
                            break
                    elif idx == start_idx + 1:
                        # 只有2天历史，检查连续2天
                        below_ma_0 = close < short_ma
                        below_ma_1 = df.iloc[idx - 1]['close'] < df.iloc[idx - 1][f"{self.config['short_ma']}_ma"]
                        if below_ma_0 and below_ma_1:
                            sell_idx = idx
                            break
                    # 如果只有1天数据（idx == start_idx），在上升趋势中不立即止损
            
            if sell_idx is not None:
                # 标记卖出信号
                df.iloc[sell_idx, df.columns.get_loc('sell_signal')] = 1
                
                if 'date' in df.columns:
                    sell_date = df.iloc[sell_idx]['date']
                    ma16_val = df.iloc[sell_idx]['16_ma']
                    ma45_val = df.iloc[sell_idx]['45_ma']
                    trend = "下跌趋势" if ma16_val < ma45_val else "上升趋势"
                    logger.info(f"  {interval_name}: 完成卖出（继承持仓） 卖出={sell_date} ({trend})")
                
                self._has_open_position = False  # 已平仓
                self._buy_price = None  # 清除买入价格
                return 1
            else:
                if 'date' in df.columns:
                    logger.info(f"  {interval_name}: 继承持仓，未找到卖出时机（持仓至间隔结束）")
                # self._has_open_position 保持 True
                return 0
        
        # 没有持仓，寻找买入信号（close >= MA16）
        buy_idx = None
        for idx in range(start_idx, end_idx + 1):
            if df.iloc[idx]['close'] >= df.iloc[idx][f"{self.config['short_ma']}_ma"]:
                buy_idx = idx
                break
        
        if buy_idx is None:
            if 'date' in df.columns:
                start_date = df.iloc[start_idx]['date']
                end_date = df.iloc[end_idx]['date']
                logger.debug(f"  {interval_name}: 未找到买入信号 ({start_date} ~ {end_date})")
            return 0
        
        # 标记买入
        df.iloc[buy_idx, df.columns.get_loc('buy_signal')] = 1
        
        # 记录买入价格（用于震荡期止损判断）
        self._buy_price = df.iloc[buy_idx]['close']
        
        # 寻找卖出信号（使用渐进式止损策略）
        sell_idx = None
        for idx in range(buy_idx + 1, end_idx + 1):
            # 检查当前趋势
            ma16 = df.iloc[idx]['16_ma']
            ma45 = df.iloc[idx]['45_ma']
            close = df.iloc[idx]['close']
            short_ma = df.iloc[idx][f"{self.config['short_ma']}_ma"]
            
            if ma16 < ma45:
                # 下跌趋势：使用传统止损（跌破MA16立即止损）
                if close < short_ma:
                    sell_idx = idx
                    break
            else:
                # 上升/横盘趋势：使用渐进式止损（连续3天跌破MA16才止损）
                # 确保买入后至少有3天数据
                days_since_buy = idx - buy_idx
                if days_since_buy >= 3:
                    below_ma_0 = close < short_ma
                    below_ma_1 = df.iloc[idx - 1]['close'] < df.iloc[idx - 1][f"{self.config['short_ma']}_ma"]
                    below_ma_2 = df.iloc[idx - 2]['close'] < df.iloc[idx - 2][f"{self.config['short_ma']}_ma"]
                    
                    # 严格要求：只有连续3天都跌破才卖出
                    if below_ma_0 and below_ma_1 and below_ma_2:
                        sell_idx = idx
                        break
                elif days_since_buy == 2:
                    # 只有2天，检查连续2天
                    below_ma_0 = close < short_ma
                    below_ma_1 = df.iloc[idx - 1]['close'] < df.iloc[idx - 1][f"{self.config['short_ma']}_ma"]
                    if below_ma_0 and below_ma_1:
                        sell_idx = idx
                        break
                # 如果只有1天数据，在上升趋势中不立即止损
        
        # 记录交易日志
        if sell_idx is not None:
            # 标记卖出信号
            df.iloc[sell_idx, df.columns.get_loc('sell_signal')] = 1
            
            if 'date' in df.columns:
                buy_date = df.iloc[buy_idx]['date']
                sell_date = df.iloc[sell_idx]['date']
                logger.info(f"  {interval_name}: 完成交易 买入={buy_date} 卖出={sell_date}")
            
            self._has_open_position = False  # 已平仓
            self._buy_price = None  # 清除买入价格
            return 1
        else:
            if 'date' in df.columns:
                buy_date = df.iloc[buy_idx]['date']
                logger.info(f"  {interval_name}: 买入后未卖出 买入={buy_date} (持仓至间隔结束)")
            
            self._has_open_position = True  # 有未平仓持仓
            return 0
    
    def _sell_in_oscillation(self, df: pd.DataFrame, start_idx: int, end_idx: int, oscillation_name: str) -> int:
        """
        在震荡期间尝试卖出已有持仓（震荡期禁止买入，但允许严重亏损止损）
        
        震荡期动态止损策略：
        1. 计算震荡区间的波动率（振幅）
        2. 波动大 → 止损宽松（允许更大回撤）
        3. 波动小 → 止损严格（及时止损）
        4. 震荡期开始下跌趋势 → 止损更严格
        
        Args:
            df: 数据DataFrame
            start_idx: 震荡期起始索引
            end_idx: 震荡期结束索引
            oscillation_name: 震荡期名称（用于日志）
            
        Returns:
            完成的交易数量（0或1）
        """
        if not self._has_open_position or self._buy_price is None:
            return 0
        
        if start_idx >= len(df) or end_idx >= len(df) or start_idx > end_idx:
            return 0
        
        # 震荡期止损阈值：15%（经过测试的最优值）
        stop_loss_threshold = 0.15
        
        # 在震荡期内寻找严重亏损卖出信号
        sell_idx = None
        for idx in range(start_idx, end_idx + 1):
            close = df.iloc[idx]['close']
            
            # 计算当前亏损比例
            loss_ratio = (self._buy_price - close) / self._buy_price
            
            # 只有当亏损超过阈值时才触发止损
            if loss_ratio > stop_loss_threshold:
                sell_idx = idx
                break
        
        if sell_idx is not None:
            # 标记卖出信号
            df.iloc[sell_idx, df.columns.get_loc('sell_signal')] = 1
            
            if 'date' in df.columns:
                sell_date = df.iloc[sell_idx]['date']
                sell_price = df.iloc[sell_idx]['close']
                loss_pct = (self._buy_price - sell_price) / self._buy_price * 100
                logger.info(f"  {oscillation_name}: 震荡期止损 卖出={sell_date} (亏损{loss_pct:.2f}%，阈值{stop_loss_threshold*100:.1f}%，买入价={self._buy_price:.2f})")
            
            self._has_open_position = False  # 已平仓
            self._buy_price = None  # 清除买入价格
            return 1
        else:
            if 'date' in df.columns:
                logger.debug(f"  {oscillation_name}: 震荡期持仓，未触发止损条件（阈值{stop_loss_threshold*100:.1f}%）")
            # self._has_open_position 保持 True
            return 0
    
    def _calculate_dynamic_stop_loss(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """
        计算震荡区间的动态止损阈值
        
        策略：
        1. 计算震荡区间的平均波动率（每日高低价振幅）
        2. 波动率越大，止损阈值越宽松
        3. 如果震荡期开始趋势向下（MA16 < MA45），止损更严格
        
        Args:
            df: 数据DataFrame
            start_idx: 震荡期起始索引
            end_idx: 震荡期结束索引
            
        Returns:
            动态止损阈值（例如 0.10 表示 10%）
        """
        # 基础止损阈值
        base_threshold = 0.12  # 12%
        
        # 计算震荡区间的平均日振幅
        volatility_sum = 0
        valid_days = 0
        for idx in range(start_idx, min(end_idx + 1, len(df))):
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            close = df.iloc[idx]['close']
            if close > 0:
                daily_volatility = (high - low) / close
                volatility_sum += daily_volatility
                valid_days += 1
        
        avg_volatility = volatility_sum / valid_days if valid_days > 0 else 0.05
        
        # 检查震荡期开始时的趋势（核心判断依据）
        trend_adjustment = 0
        if start_idx < len(df):
            ma16_start = df.iloc[start_idx]['16_ma']
            ma45_start = df.iloc[start_idx]['45_ma']
            
            # 趋势是最重要的判断因素
            if ma16_start < ma45_start:
                # 下跌趋势进入震荡期：严格止损（基础阈值）
                trend_adjustment = 0
            else:
                # 上升趋势进入震荡期：宽松止损（允许更大回撤）
                trend_adjustment = 0.05  # 增加5%容忍度
        
        # 根据波动率微调止损阈值
        # 波动率大 → 略微收紧止损（风险控制）
        # 波动率小 → 略微放宽止损（避免过早止损）
        if avg_volatility > 0.10:
            volatility_adjustment = -0.02  # 剧烈波动，收紧2%
        else:
            volatility_adjustment = 0  # 正常波动，不调整
        
        # 计算最终止损阈值
        final_threshold = base_threshold + volatility_adjustment + trend_adjustment
        
        # 限制在合理范围内：最小8%，最大25%
        final_threshold = max(0.08, min(0.25, final_threshold))
        
        return final_threshold
    
    def get_oscillation_confirmed_periods(self) -> List[Tuple]:
        """
        返回震荡确认时间列表（用于绘图）
        
        Returns:
            list: [(confirmed_idx, end_idx, confirmed_date, end_date), ...]
        """
        # 这个方法需要在 analyze() 调用后才有数据
        return getattr(self, '_oscillation_periods', [])
    
    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """
        获取最新的交易信号
        
        Args:
            df: 分析后的DataFrame
            
        Returns:
            包含最新信号信息的字典
        """
        if df is None or len(df) == 0:
            return {
                'date': None,
                'close': None,
                'signal': '无信号',
                'oscillation_state': '未知'
            }
        
        latest = df.iloc[-1]
        
        # 判断当前状态
        is_in_oscillation = latest.get('oscillation_state', 0) == 1
        buy_signal = latest.get('buy_signal', 0)
        sell_signal = latest.get('sell_signal', 0)
        
        # 根据信号和状态确定信号类型
        signal = 'HOLD'
        reason = []
        strength = 0
        
        if buy_signal == 1:
            signal = 'BUY'
            reason.append(f"震荡间隔买入（close >= MA16）")
            strength = 3
        elif sell_signal == 1:
            signal = 'SELL'
            # 判断卖出原因
            ma16 = latest.get('16_ma', 0)
            ma45 = latest.get('45_ma', 0)
            if ma16 < ma45:
                reason.append("下跌趋势止损（close < MA16）")
            else:
                reason.append("连续3天跌破MA16止损")
            strength = 3
        elif is_in_oscillation:
            signal = 'HOLD'
            reason.append("震荡期观望")
            strength = 0
        
        return {
            'date': latest.get('date', ''),
            'close': latest.get('close', 0),
            'price': latest.get('close', 0),
            'signal': signal,
            'reason': '，'.join(reason) if reason else '无',
            'strength': strength,
            'oscillation_state': '震荡中' if is_in_oscillation else '正常行情',
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            # 技术指标
            'k': latest.get('k', 0),
            'd': latest.get('d', 0),
            'macd': latest.get('macd', 0),
            'diff': latest.get('diff', 0),
            'ma_16': latest.get('16_ma', 0),
            'ma_45': latest.get('45_ma', 0)
        }
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Optional[Dict]:
        """
        执行回测
        
        Args:
            df: 分析后的DataFrame
            initial_capital: 初始资金
            
        Returns:
            回测结果字典
        """
        if df is None or len(df) == 0:
            return None
        
        capital = initial_capital
        position = 0  # 持仓数量
        buy_price = 0  # 买入价格
        buy_date = None  # 买入日期
        buy_commission = 0  # 买入手续费
        trades = []  # 配对的交易记录（每条包含买入和卖出）
        equity_curve = []  # 资金曲线
        
        commission_rate = self.config.get('commission_rate', 0.0003)
        min_commission = self.config.get('min_commission', 5.0)
        stamp_duty = self.config.get('stamp_duty_cn', 0.0005) if self.market == 'CN-A' else 0
        
        max_equity = initial_capital
        max_drawdown = 0
        
        for idx, row in df.iterrows():
            current_price = row['close']
            
            # 计算当前资产
            current_equity = capital + position * current_price
            equity_curve.append(current_equity)
            
            # 更新最大回撤
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = (max_equity - current_equity) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            # 买入信号
            if row.get('buy_signal', 0) == 1 and position == 0:
                # 计算可买入数量（考虑手续费的精确计算）
                shares = int(capital / current_price)
                while shares > 0:
                    cost = shares * current_price
                    buy_commission = max(cost * commission_rate, min_commission)
                    total_cost = cost + buy_commission
                    
                    if total_cost <= capital:
                        capital -= total_cost
                        position = shares
                        buy_price = current_price
                        buy_date = row.get('date', idx)
                        # 保存买入信息，等卖出时配对
                        break
                    
                    shares -= 1
            
            # 卖出信号
            elif row.get('sell_signal', 0) == 1 and position > 0:
                # 卖出
                revenue = position * current_price
                sell_commission = max(revenue * commission_rate, min_commission)
                stamp = revenue * stamp_duty
                total_revenue = revenue - sell_commission - stamp
                
                capital += total_revenue
                
                # 计算收益
                initial_investment = position * buy_price
                profit = (current_price - buy_price) * position - buy_commission - sell_commission - stamp
                profit_rate = profit / initial_investment if initial_investment > 0 else 0
                
                # 添加配对的交易记录（与MixedStrategy保持一致）
                trades.append({
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': row.get('date', idx),
                    'sell_price': current_price,
                    'profit_rate': profit_rate,
                    'capital': capital,  # 卖出后的资金
                    'commission': buy_commission + sell_commission + stamp,  # 总手续费
                    'buy_commission': buy_commission,
                    'sell_commission': sell_commission,
                })
                
                position = 0
                buy_price = 0
                buy_date = None
        
        # 如果最后还有持仓，记录未平仓状态（用于计算最大回撤等指标，但不计入trades）
        # 与MixedStrategy保持一致：不强制平仓，保持持仓状态
        final_capital_with_position = capital
        open_position_info = None
        if position > 0:
            # 计算当前持仓的市值（用于最大回撤等指标）
            final_price = df.iloc[-1]['close']
            revenue = position * final_price
            commission = max(revenue * commission_rate, min_commission)
            stamp = revenue * stamp_duty
            total_revenue = revenue - commission - stamp
            
            final_capital_with_position = capital + total_revenue
            # 保存未平仓的买入信息（不添加到trades中）
            open_position_info = {
                'buy_date': buy_date,
                'buy_price': buy_price,
                'buy_commission': buy_commission,
                'shares': position,
                'current_price': final_price,
                'current_value': total_revenue,
            }
        
        # 计算统计指标
        # final_capital: 用于显示的最终资金（包含持仓市值）
        # final_capital_closed: 已完成交易的资金（不含持仓）
        final_capital_closed = capital
        final_capital = final_capital_with_position  # 显示时使用含持仓的资金
        
        # 计算收益率
        total_return_closed = (final_capital_closed / initial_capital - 1) * 100  # 已完成交易的收益率
        total_return = (final_capital / initial_capital - 1) * 100  # 含持仓的总收益率
        
        # 统计交易：trades现在是配对的，每条记录代表一对完整的买卖
        total_trades = len(trades)
        
        # 胜率：基于已完成的交易对
        winning_trades = len([t for t in trades if t.get('profit_rate', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 计算夏普比率
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,  # 含持仓市值的最终资金（用于显示）
            'final_capital_closed': final_capital_closed,  # 已完成交易的资金（不含持仓）
            'total_return': total_return,  # 含持仓的总收益率
            'total_return_closed': total_return_closed,  # 已完成交易的收益率
            'total_trades': total_trades,
            'has_open_position': (position > 0),  # 是否有持仓
            'open_position_info': open_position_info,  # 未平仓的买入信息
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,  # 配对的交易记录
            'equity_curve': equity_curve,
            'gross_return': total_return,  # 毛收益率（含持仓）
        }
    
    def get_trading_signals(self, df: pd.DataFrame, initial_capital: float = 10000) -> Optional[Dict]:
        """
        获取交易信号列表（与MixedStrategy保持一致的逻辑）
        
        Args:
            df: 分析后的DataFrame
            initial_capital: 初始资金
            
        Returns:
            交易信号字典
        """
        backtest_result = self.backtest(df, initial_capital)
        if not backtest_result:
            return None
        
        trades = backtest_result['trades']
        
        # 从配对的trades中提取买入点和卖出点（与MixedStrategy保持一致）
        buy_points = []
        sell_points = []
        
        for trade in trades:
            # 买入点
            buy_points.append({
                'date': trade['buy_date'],
                'price': trade['buy_price'],
                'reason': '震荡间隔买入（close >= MA16）',
                'commission': trade.get('buy_commission', 0),
            })
            
            # 卖出点
            sell_points.append({
                'date': trade['sell_date'],
                'price': trade['sell_price'],
                'reason': '震荡间隔卖出（close < MA16 or MACD < 0）',
                'commission': trade.get('sell_commission', 0),
            })
        
        # 检查是否有持仓
        has_open_position = backtest_result.get('has_open_position', False)
        open_position_info = backtest_result.get('open_position_info')
        
        # 如果有持仓，需要添加最后一笔未配对的买入和虚拟卖出（用于显示）
        all_trades = trades.copy()  # 完整的交易记录（包括未平仓）
        if has_open_position and open_position_info:
            # 添加买入点
            buy_points.append({
                'date': open_position_info['buy_date'],
                'price': open_position_info['buy_price'],
                'reason': '震荡间隔买入（close >= MA16）',
                'commission': open_position_info.get('buy_commission', 0),
            })
            
            # 添加虚拟卖出点（用于配对显示）
            sell_points.append({
                'date': open_position_info['buy_date'],  # 使用买入日期作为占位
                'price': open_position_info.get('current_price', 0),  # 当前价格
                'reason': '未平仓',
                'commission': 0,
                'is_open': True,  # 标记为未平仓
            })
            
            # 添加未平仓的交易对到all_trades（用于交易对收益分析表）
            all_trades.append({
                'buy_date': open_position_info['buy_date'],
                'buy_price': open_position_info['buy_price'],
                'sell_date': open_position_info['buy_date'],  # 占位，实际未卖出
                'sell_price': open_position_info.get('current_price', 0),
                'profit_rate': 0,  # 未实现收益
                'capital': backtest_result.get('final_capital_closed', backtest_result['final_capital']),  # 已完成交易的资金（不含持仓）
                'commission': open_position_info.get('buy_commission', 0),
                'buy_commission': open_position_info.get('buy_commission', 0),
                'sell_commission': 0,
                'is_open': True,  # 标记为未平仓
            })
        
        # 构造buy_indices和sell_indices（用于图表标记）
        buy_indices = [t['buy_date'] for t in trades]
        sell_indices = [t['sell_date'] for t in trades]
        
        # 如果有持仓，需要添加持仓的买入日期到buy_indices（用于图表标记）
        if has_open_position and open_position_info:
            buy_indices.append(open_position_info['buy_date'])
            # sell_indices不需要添加，因为是虚拟卖出点
        
        return {
            'total_trades': len(trades) * 2 + (1 if has_open_position else 0),  # 买入+卖出
            'buy_points': buy_points,
            'sell_points': sell_points,
            'buy_indices': buy_indices,
            'sell_indices': sell_indices,
            'trades': all_trades,  # 包含未平仓的完整交易记录
            'initial_capital': initial_capital,
            'has_open_position': has_open_position,
            'gross_return': backtest_result.get('gross_return', 0),  # 含持仓的总收益率
        }
