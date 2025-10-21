"""
顶底背离检测模块
- 优化性能，减少循环嵌套
- 提供清晰的背离信号
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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


def get_bottom_divergence_index(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测底部背离

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
    bottom_divergence_index = []
    close_index = []

    # 找到所有 DIFF 的局部最小值
    min_list = df[(df['diff'] <= df['diff'].shift(1)) &
                  (df['diff'] <= df['diff'].shift(-1))].index

    if len(min_list) <= 1:
        return bottom_divergence_index

    for i in range(len(min_list) - 1):
        for j in range(i + 1, len(min_list)):
            # 跳过 DIFF 没有抬升的情况
            if df.loc[min_list[i], 'diff'] > df.loc[min_list[j], 'diff']:
                continue

            # 查找两个 DIFF 极值点前的价格局部最低点
            ind1, ind2 = 0.0, 0.0

            subdf1 = df.loc[:min_list[i]]
            for k in range(1, min(len(subdf1), 4)):
                if (subdf1['close'].iloc[-k] < subdf1['close'].iloc[-k-1] and
                    subdf1['close'].iloc[-k] < subdf1['close'].iloc[-k+1]) or \
                   (subdf1['close'].iloc[-k+1] > subdf1['open'].iloc[-k+1] and
                    subdf1['close'].iloc[-k] < subdf1['open'].iloc[-k]):
                    ind1 = subdf1['close'].iloc[-k]
                    break

            subdf2 = df.loc[min_list[i]: min_list[j]]
            for m in range(1, min(len(subdf2), 4)):
                if (subdf2['close'].iloc[-m] < subdf2['close'].iloc[-m-1] and
                    subdf2['close'].iloc[-m] < subdf2['close'].iloc[-m+1]) or \
                   (subdf2['close'].iloc[-m+1] > subdf2['open'].iloc[-m+1] and
                    subdf2['close'].iloc[-m] < subdf2['open'].iloc[-m]):
                    ind2 = subdf2['close'].iloc[-m]
                    break

            if ind1 == 0.0 or ind2 == 0.0:
                continue

            # 检查是否跌破 ind2
            break_sig = False
            for idx in subdf2.index:
                if subdf2.loc[idx, 'close'] <= ind2 - 0.1:
                    break_sig = True
                    break
            if break_sig:
                continue

            # 价格下跌至少 10%
            if ind1 <= ind2 or (ind1 - ind2) / ind1 < 0.1:
                continue

            close_index.append(min_list[j])

            # 价格跌幅要求
            if (df.loc[min_list[i], 'close'] - df.loc[min_list[j], 'close']) / df.loc[min_list[j], 'close'] < 0.1:
                continue

            # 背离条件检查
            c1 = df.loc[min_list[i], 'close'] > df.loc[min_list[j], 'close']
            c2 = df.loc[min_list[j], 'diff'] - df.loc[min_list[i], 'diff'] > 0.03
            c3 = df.loc[min_list[i], 'macd'] < 0
            c4 = df.loc[min_list[j], 'macd'] < 0
            c5 = df.loc[min_list[j], 'macd'] - df.loc[min_list[i], 'macd'] > 0.02
            c6 = df.loc[min_list[i]: min_list[j], 'macd'].max() > 0

            tmpdf = df.loc[min_list[i]: min_list[j]]
            tmp_len = len(tmpdf[(tmpdf['close'] < tmpdf['close'].shift(1)) &
                                (tmpdf['close'] < tmpdf['close'].shift(-1))].index)
            c7 = tmp_len != 0

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[min_list[i], 'date'], df.loc[min_list[j], 'date'])
            if diff_days > lookback_days:
                break

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                date_i = df.loc[min_list[i], 'date']
                date_j = df.loc[min_list[j], 'date']
                print(f"底部背离: {date_i} -> {date_j}")  # 注释掉以加快速度
                bottom_divergence_index.append(min_list[j])

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
    检测顶部背离（基于 DIFF）- 延迟确认版本（无未来函数）

    改进说明：
    - 不使用未来函数 shift(-1)
    - 在下一个交易日确认前一天是否为极值点
    - 顶背离信号延迟一天发出（返回确认日，即极值点次日）

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
    peak_divergence_index = []

    # 找到所有 DIFF 的局部最大值（仅向后看，不使用未来函数）
    # 在第i+1天，如果发现第i天的DIFF > 第i-1天 且 第i天DIFF > 第i+1天（今天）
    # 那么第i天就是极值点，在第i+1天确认
    max_list = []
    for i in range(1, len(df) - 1):
        if (df['diff'].iloc[i] > df['diff'].iloc[i-1] and
            df['diff'].iloc[i] > df['diff'].iloc[i+1]):
            # 在i+1这天确认，i是极值点
            # max_list 存储确认日的索引
            max_list.append(df.index[i+1])

    if len(max_list) <= 1:
        return peak_divergence_index

    for i in range(len(max_list) - 1):
        for j in range(i + 1, len(max_list)):
            # max_list[i] 是确认日，实际极值点在前一天
            actual_peak_i = df.index[df.index.get_loc(max_list[i]) - 1]
            actual_peak_j = df.index[df.index.get_loc(max_list[j]) - 1]

            # 查找两个 DIFF 极值点前的价格局部最高点
            # 这部分在回测中使用历史数据，不算未来函数
            ind1, ind2 = 0.0, 0.0

            subdf1 = df.loc[:actual_peak_i]
            for k in range(1, min(len(subdf1), 5)):
                if (subdf1['high'].iloc[-k] > subdf1['high'].iloc[-k-1] and
                    k < len(subdf1) - 1 and subdf1['high'].iloc[-k] > subdf1['high'].iloc[-k+1]):
                    ind1 = subdf1['high'].iloc[-k]
                    break

            subdf2 = df.loc[actual_peak_i: actual_peak_j]
            for m in range(1, min(len(subdf2), 5)):
                if (subdf2['high'].iloc[-m] > subdf2['high'].iloc[-m-1] and
                    m < len(subdf2) - 1 and subdf2['high'].iloc[-m] > subdf2['high'].iloc[-m+1]):
                    ind2 = subdf2['high'].iloc[-m]
                    break

            if ind1 == 0.0 or ind2 == 0.0 or ind1 >= ind2:
                continue

            # 背离条件检查（使用实际极值点）
            c1 = (df.loc[actual_peak_j, 'high'] - df.loc[actual_peak_i, 'high']) / df.loc[actual_peak_j, 'high'] > 0.01
            c2 = df.loc[actual_peak_i, 'diff'] > df.loc[actual_peak_j, 'diff']
            c3 = df.loc[actual_peak_i, 'macd'] > 0
            c4 = df.loc[actual_peak_j, 'macd'] > 0
            c5 = df.loc[actual_peak_j, 'macd'] < df.loc[actual_peak_i, 'macd']
            c6 = df.loc[actual_peak_i: actual_peak_j, 'macd'].min() < 0

            tmpdf = df.loc[actual_peak_i: actual_peak_j]
            # 这里在历史数据范围内检查，不算未来函数
            tmp_len = len(tmpdf[(tmpdf['close'] > tmpdf['close'].shift(1)) &
                                (tmpdf['close'] > tmpdf['close'].shift(-1))].index)
            c7 = tmp_len != 0

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[actual_peak_i, 'date'], df.loc[actual_peak_j, 'date'])
            if diff_days > lookback_days:
                break

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
