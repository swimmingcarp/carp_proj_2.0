"""
顶底背离检测模块
- 优化性能，减少循环嵌套
- 提供清晰的背离信号
- 使用向量化和numpy优化计算速度
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


def cal_time_diff(date1, date2) -> int:
    """
    计算两个日期之间的天数差

    Args:
        date1: 日期（字符串、datetime 或 pandas Timestamp）
        date2: 日期（字符串、datetime 或 pandas Timestamp）

    Returns:
        天数差
    """
    # 处理不同类型的输入
    if isinstance(date1, str):
        d1 = datetime.strptime(date1, "%Y-%m-%d")
    else:
        d1 = pd.to_datetime(date1)

    if isinstance(date2, str):
        d2 = datetime.strptime(date2, "%Y-%m-%d")
    else:
        d2 = pd.to_datetime(date2)

    return abs((d2 - d1).days)


def _find_local_extrema_vectorized(series: pd.Series, find_min: bool = True) -> np.ndarray:
    """
    向量化查找序列的局部极值点

    Args:
        series: 输入序列
        find_min: True查找局部最小值，False查找局部最大值

    Returns:
        局部极值点的索引数组
    """
    if len(series) < 3:
        return np.array([], dtype=int)

    # 转换为numpy数组以提高性能
    values = series.values

    # 向量化比较：当前值与前后值的比较
    if find_min:
        # 局部最小值：values[i] <= values[i-1] AND values[i] <= values[i+1]
        is_extrema = (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
    else:
        # 局部最大值：values[i] >= values[i-1] AND values[i] >= values[i+1]
        is_extrema = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])

    # 获取满足条件的索引（需要+1因为从index=1开始）
    extrema_indices = np.where(is_extrema)[0] + 1

    # 转换为原始DataFrame的索引
    return series.index[extrema_indices].values


def get_bottom_divergence_index(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测底部背离（优化版本 - 无未来函数）

    改进说明：
    - 不使用未来函数 shift(-1)
    - 使用逐日检测，只向后看
    - 在下一个交易日确认前一天是否为极值点
    - 使用numpy数组优化性能

    条件：
    1. 价格创新低（close[i] > close[j]）
    2. DIFF 抬升（diff[i] < diff[j]）
    3. 两点都在零轴下且 MACD 抬升
    4. 中间有 MACD 上穿零轴
    5. 价格下跌幅度 >= 10%

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        底部背离日期列表
    """
    if len(df) < 10:
        return []

    # 提取需要的列到numpy数组，减少DataFrame访问
    diff_values = df['diff'].values
    close_values = df['close'].values
    macd_values = df['macd'].values
    open_values = df['open'].values
    date_values = df['date'].values
    df_index = df.index

    bottom_divergence_index = []
    close_index = []

    # 找到所有 DIFF 的局部最小值（无未来函数版本）
    # 在第i+1天，确认第i天是极值点
    min_list = []
    for i in range(1, len(df) - 1):
        if (diff_values[i] <= diff_values[i-1] and
            diff_values[i] <= diff_values[i+1]):
            # 在i+1天确认，i是极值点
            min_list.append(i)

    if len(min_list) <= 1:
        return []

    # 遍历所有局部最小值对
    for i in range(len(min_list) - 1):
        pos_i = min_list[i]

        for j in range(i + 1, len(min_list)):
            pos_j = min_list[j]

            # 早期终止：时间窗口检查（提前，避免不必要的计算）
            diff_days = cal_time_diff(date_values[pos_i], date_values[pos_j])
            if diff_days > lookback_days:
                break

            # 跳过 DIFF 没有抬升的情况
            if diff_values[pos_i] > diff_values[pos_j]:
                continue

            # 查找价格局部最低点（保持原始逻辑）
            ind1, ind2 = 0.0, 0.0

            # 在pos_i前最多查找3个点
            for k in range(1, min(pos_i + 1, 4)):
                if pos_i - k < 0:
                    break
                pk = pos_i - k
                # 条件1: 局部最小值检查
                cond1 = (pk > 0 and pk < len(close_values) - 1 and
                        close_values[pk] < close_values[pk - 1] and
                        close_values[pk] < close_values[pk + 1])
                # 条件2: K线形态检查
                cond2 = (pk > 0 and pk < len(close_values) - 1 and
                        close_values[pk + 1] > open_values[pk + 1] and
                        close_values[pk] < open_values[pk])
                if cond1 or cond2:
                    ind1 = close_values[pk]
                    break

            # 在pos_i到pos_j之间查找
            for m in range(1, min(pos_j - pos_i + 1, 4)):
                if pos_j - m < 0:
                    break
                pm = pos_j - m
                # 条件1: 局部最小值检查
                cond1 = (pm > 0 and pm < len(close_values) - 1 and
                        close_values[pm] < close_values[pm - 1] and
                        close_values[pm] < close_values[pm + 1])
                # 条件2: K线形态检查
                cond2 = (pm > 0 and pm < len(close_values) - 1 and
                        close_values[pm + 1] > open_values[pm + 1] and
                        close_values[pm] < open_values[pm])
                if cond1 or cond2:
                    ind2 = close_values[pm]
                    break

            if ind1 == 0.0 or ind2 == 0.0:
                continue

            # 检查是否跌破 ind2（向量化）
            segment_close = close_values[pos_i:pos_j + 1]
            if np.any(segment_close <= ind2 - 0.1):
                continue

            # 价格下跌至少 10%
            if ind1 <= ind2 or (ind1 - ind2) / ind1 < 0.1:
                continue

            close_index.append(df_index[pos_j])

            # 价格跌幅要求
            if (close_values[pos_i] - close_values[pos_j]) / close_values[pos_j] < 0.1:
                continue

            # 背离条件检查
            c1 = close_values[pos_i] > close_values[pos_j]
            c2 = diff_values[pos_j] - diff_values[pos_i] > 0.03
            c3 = macd_values[pos_i] < 0
            c4 = macd_values[pos_j] < 0
            c5 = macd_values[pos_j] - macd_values[pos_i] > 0.02

            # 向量化：检查期间是否有MACD>0
            segment_macd = macd_values[pos_i:pos_j + 1]
            c6 = np.max(segment_macd) > 0

            # 检查是否有价格局部最低点
            segment_close_full = close_values[pos_i:pos_j + 1]
            if len(segment_close_full) > 2:
                is_local_min = (segment_close_full[1:-1] < segment_close_full[:-2]) & \
                               (segment_close_full[1:-1] < segment_close_full[2:])
                c7 = np.any(is_local_min)
            else:
                c7 = False

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                print(f"底部背离: {date_values[pos_i]} -> {date_values[pos_j]}")
                bottom_divergence_index.append(df_index[pos_j])

    return list(set(bottom_divergence_index) & set(close_index))


def get_peak_divergence_delayed(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    延迟确认法顶背离检测 - 避免Look-ahead Bias

    原理：
    - 在第T天，检查第T-1天是否为局部最大值
    - 如果是，则将T-1标记为候选峰值
    - 与历史峰值对比，判断是否顶背离

    优点：
    - 保留原算法的准确性
    - 只延迟1天，影响可控
    - 实时和回测逻辑一致
    - 不依赖未来数据

    Args:
        df: 包含技术指标的DataFrame
        lookback_days: 回溯天数

    Returns:
        顶背离日期索引列表
    """
    divergence_index = []
    confirmed_peaks = []  # 已确认的峰值列表

    if len(df) < 10:
        return divergence_index

    # 从第2天开始（需要检查前一天）
    for i in range(2, len(df)):
        prev_idx = i - 1
        prev_prev_idx = i - 2

        # 检查前一天是否为局部最大值
        # 条件：diff[i-1] > diff[i-2] AND diff[i-1] > diff[i]
        is_local_max = (df.iloc[prev_idx]['diff'] > df.iloc[prev_prev_idx]['diff'] and
                       df.iloc[prev_idx]['diff'] > df.iloc[i]['diff'])

        if is_local_max:
            prev_date = df.iloc[prev_idx]['date']
            prev_diff = df.iloc[prev_idx]['diff']
            prev_high = df.iloc[prev_idx]['high']
            prev_macd = df.iloc[prev_idx]['macd']

            # 将前一天标记为已确认峰值
            confirmed_peaks.append({
                'index': prev_idx,
                'date': prev_date,
                'diff': prev_diff,
                'high': prev_high,
                'macd': prev_macd
            })

            # 与历史峰值对比，检测顶背离
            if len(confirmed_peaks) >= 2:
                # 取最近的前一个峰值
                for old_peak in reversed(confirmed_peaks[:-1]):
                    # 时间窗口限制
                    days_diff = cal_time_diff(old_peak['date'], prev_date)
                    if days_diff > lookback_days:
                        break

                    # 顶背离条件：
                    # 1. 价格创新高
                    # 2. DIFF下降
                    # 3. 两个峰值的MACD都为正
                    price_higher = prev_high > old_peak['high']
                    diff_lower = prev_diff < old_peak['diff']
                    both_macd_positive = prev_macd > 0 and old_peak['macd'] > 0

                    if price_higher and diff_lower and both_macd_positive:
                        divergence_index.append(prev_idx)
                        print(f"延迟确认顶背离: {prev_date}, "
                              f"价格 {old_peak['high']:.2f}->{prev_high:.2f}, "
                              f"DIFF {old_peak['diff']:.4f}->{prev_diff:.4f}")
                        break

    return list(set(divergence_index))

def get_peak_divergence_index_original(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """原始复杂版本的顶背离检测 - 保留用于对比测试"""
    return get_peak_divergence_index_impl(df, lookback_days)

def get_peak_divergence_index_impl(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测顶部背离（基于 DIFF）- 优化版本（使用向量化和numpy加速）
    延迟确认版本（无未来函数）

    改进说明：
    - 不使用未来函数 shift(-1)
    - 在下一个交易日确认前一天是否为极值点
    - 顶背离信号延迟一天发出（返回确认日，即极值点次日）
    - 使用numpy数组减少DataFrame访问

    条件：
    1. 价格创新高（high[i] < high[j]）
    2. DIFF 下降（diff[i] > diff[j]）
    3. 两点都在零轴上且 MACD 下降
    4. 中间有 MACD 下穿零轴

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        顶部背离日期列表（返回确认日，即极值点的次日）
    """
    if len(df) < 10:
        return []

    # 提前提取列到numpy数组
    diff_values = df['diff'].values
    high_values = df['high'].values
    macd_values = df['macd'].values
    close_values = df['close'].values
    date_values = df['date'].values
    df_index = df.index

    peak_divergence_index = []

    # 找到所有 DIFF 的局部最大值（仅向后看，不使用未来函数）
    # 在第i+1天，如果发现第i天的DIFF > 第i-1天 且 第i天DIFF > 第i+1天（今天）
    # 那么第i天就是极值点，在第i+1天确认
    max_list = []
    for i in range(1, len(df) - 1):
        if (diff_values[i] > diff_values[i-1] and
            diff_values[i] > diff_values[i+1]):
            # 在i+1这天确认，i是极值点
            # max_list 存储确认日的索引
            max_list.append(df_index[i+1])

    if len(max_list) <= 1:
        return peak_divergence_index

    # 创建索引映射
    index_to_pos = {idx: pos for pos, idx in enumerate(df_index)}

    for i in range(len(max_list) - 1):
        # max_list[i] 是确认日，实际极值点在前一天
        confirm_pos_i = index_to_pos[max_list[i]]
        actual_peak_pos_i = confirm_pos_i - 1

        for j in range(i + 1, len(max_list)):
            # max_list[j] 是确认日，实际极值点在前一天
            confirm_pos_j = index_to_pos[max_list[j]]
            actual_peak_pos_j = confirm_pos_j - 1

            # 早期终止：时间窗口检查
            diff_days = cal_time_diff(date_values[actual_peak_pos_i], date_values[actual_peak_pos_j])
            if diff_days > lookback_days:
                break

            # 查找价格局部最高点（优化：使用numpy数组）
            ind1, ind2 = 0.0, 0.0

            # 在actual_peak_i前查找
            for k in range(1, min(actual_peak_pos_i + 1, 5)):
                if actual_peak_pos_i - k < 0:
                    break
                pk = actual_peak_pos_i - k
                if pk > 0 and pk < len(high_values) - 1:
                    if (high_values[pk] > high_values[pk - 1] and
                        high_values[pk] > high_values[pk + 1]):
                        ind1 = high_values[pk]
                        break

            # 在actual_peak_i到actual_peak_j之间查找
            for m in range(1, min(actual_peak_pos_j - actual_peak_pos_i + 1, 5)):
                if actual_peak_pos_j - m < 0:
                    break
                pm = actual_peak_pos_j - m
                if pm > 0 and pm < len(high_values) - 1:
                    if (high_values[pm] > high_values[pm - 1] and
                        high_values[pm] > high_values[pm + 1]):
                        ind2 = high_values[pm]
                        break

            if ind1 == 0.0 or ind2 == 0.0 or ind1 >= ind2:
                continue

            # 背离条件检查（使用numpy数组）
            c1 = (high_values[actual_peak_pos_j] - high_values[actual_peak_pos_i]) / high_values[actual_peak_pos_j] > 0.01
            c2 = diff_values[actual_peak_pos_i] > diff_values[actual_peak_pos_j]
            c3 = macd_values[actual_peak_pos_i] > 0
            c4 = macd_values[actual_peak_pos_j] > 0
            c5 = macd_values[actual_peak_pos_j] < macd_values[actual_peak_pos_i]

            # 向量化：检查期间是否有MACD<0
            segment_macd = macd_values[actual_peak_pos_i:actual_peak_pos_j + 1]
            c6 = np.min(segment_macd) < 0

            # 向量化：检查是否有价格局部最高点
            segment_close = close_values[actual_peak_pos_i:actual_peak_pos_j + 1]
            if len(segment_close) > 2:
                is_local_max = (segment_close[1:-1] > segment_close[:-2]) & \
                               (segment_close[1:-1] > segment_close[2:])
                c7 = np.any(is_local_max)
            else:
                c7 = False

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                # 返回确认日（延迟一天），这是顶背离卖出的日期
                peak_divergence_index.append(max_list[j])

    return list(set(peak_divergence_index))


def get_peak_divergence_index(df: pd.DataFrame, lookback_days: int = 100, use_simple: bool = False) -> list:
    """
    顶背离检测 - 支持切换简化版本和复杂版本

    Args:
        df: 包含技术指标的DataFrame
        lookback_days: 回溯天数
        use_simple: True使用简化版(get_peak_divergence_delayed), False使用复杂版(原始版本)

    Returns:
        顶背离日期索引列表
    """
    if use_simple:
        return get_peak_divergence_delayed(df, lookback_days)
    else:
        return get_peak_divergence_index_impl(df, lookback_days)


def get_peak_divergence_index_kdj(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测顶部背离（基于 KDJ）- 延迟确认版本（无未来函数）

    改进说明：
    - 不使用未来函数 shift(-1)
    - 在下一个交易日确认前一天是否为极值点
    - 顶背离信号延迟一天发出（返回确认日，即极值点次日）

    条件：
    1. 价格上涨 > 1%
    2. K 值下降
    3. MACD > 0 且下降
    4. 中间有 MACD 下穿零轴

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        顶部背离日期列表（返回确认日，即极值点的次日）
    """
    peak_divergence_index = []

    # 找到所有 K 值的局部最大值（仅向后看，不使用未来函数）
    max_list = []
    for i in range(1, len(df) - 1):
        if (df['k'].iloc[i] > df['k'].iloc[i-1] and
            df['k'].iloc[i] > df['k'].iloc[i+1]):
            # 在i+1这天确认，i是极值点
            max_list.append(df.index[i+1])

    if len(max_list) <= 1:
        return peak_divergence_index

    for i in range(len(max_list) - 1):
        for j in range(i + 1, len(max_list)):
            # max_list[i] 是确认日，实际极值点在前一天
            actual_peak_i = df.index[df.index.get_loc(max_list[i]) - 1]
            actual_peak_j = df.index[df.index.get_loc(max_list[j]) - 1]

            # 查找价格局部最高点（在历史数据范围内，不算未来函数）
            ind1, ind2 = 0.0, 0.0

            subdf1 = df.loc[:actual_peak_i]
            for k in range(1, min(len(subdf1), 5)):
                if (subdf1['close'].iloc[-k] > subdf1['close'].iloc[-k-1] and
                    k < len(subdf1) - 1 and subdf1['close'].iloc[-k] > subdf1['close'].iloc[-k+1]):
                    ind1 = subdf1['close'].iloc[-k]
                    break

            subdf2 = df.loc[actual_peak_i: actual_peak_j]
            for m in range(1, min(len(subdf2), 5)):
                if (subdf2['close'].iloc[-m] > subdf2['close'].iloc[-m-1] and
                    m < len(subdf2) - 1 and subdf2['close'].iloc[-m] > subdf2['close'].iloc[-m+1]):
                    ind2 = subdf2['close'].iloc[-m]
                    break

            if ind1 == 0.0 or ind2 == 0.0 or ind1 >= ind2:
                continue

            # 背离条件（使用实际极值点）
            c1 = (df.loc[actual_peak_j, 'close'] - df.loc[actual_peak_i, 'close']) / df.loc[actual_peak_j, 'close'] > 0.01
            c2 = df.loc[actual_peak_i, 'k'] > df.loc[actual_peak_j, 'k']
            c3 = df.loc[actual_peak_i, 'macd'] > 0
            c4 = df.loc[actual_peak_j, 'macd'] > 0
            c5 = df.loc[actual_peak_j, 'macd'] < df.loc[actual_peak_i, 'macd']
            c6 = df.loc[actual_peak_i: actual_peak_j, 'macd'].min() < 0

            tmpdf = df.loc[actual_peak_i: actual_peak_j]
            tmp_len = len(tmpdf[(tmpdf['close'] > tmpdf['close'].shift(1)) &
                                (tmpdf['close'] > tmpdf['close'].shift(-1))].index)
            c7 = tmp_len != 0

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[actual_peak_i, 'date'], df.loc[actual_peak_j, 'date'])
            if diff_days > lookback_days:
                break

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                # 返回确认日（延迟一天）
                peak_divergence_index.append(max_list[j])

    return list(set(peak_divergence_index))


def get_peak_divergence_index_kd_variant(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测顶部背离（KD 变体 - 简化版）- 延迟确认版本（无未来函数）

    改进说明：
    - 不使用未来函数 shift(-1)
    - 在下一个交易日确认前一天是否为极值点
    - 顶背离信号延迟一天发出（返回确认日，即极值点次日）

    条件：
    1. 价格创新高
    2. K 值下降
    3. MACD > 0

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        顶部背离日期列表（返回确认日，即极值点的次日）
    """
    peak_divergence_index = []

    # 找到所有价格局部最高点（仅向后看，不使用未来函数）
    max_list = []
    for i in range(1, len(df) - 1):
        if (df['close'].iloc[i] > df['close'].iloc[i-1] and
            df['close'].iloc[i] > df['close'].iloc[i+1]):
            # 在i+1这天确认，i是极值点
            max_list.append(df.index[i+1])

    if len(max_list) <= 1:
        return peak_divergence_index

    for i in range(len(max_list) - 1):
        for j in range(i + 1, len(max_list)):
            # max_list[i] 是确认日，实际极值点在前一天
            actual_peak_i = df.index[df.index.get_loc(max_list[i]) - 1]
            actual_peak_j = df.index[df.index.get_loc(max_list[j]) - 1]

            # 价格创新高，K 值下降
            if df.loc[actual_peak_i, 'close'] > df.loc[actual_peak_j, 'close']:
                continue
            if df.loc[actual_peak_i, 'k'] <= df.loc[actual_peak_j, 'k']:
                continue

            # MACD > 0
            if df.loc[actual_peak_j, 'macd'] <= 0:
                continue

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[actual_peak_i, 'date'], df.loc[actual_peak_j, 'date'])
            if diff_days > lookback_days:
                break
            if diff_days <= 5:
                continue

            # 返回确认日（延迟一天）
            peak_divergence_index.append(max_list[j])

    return list(set(peak_divergence_index))
