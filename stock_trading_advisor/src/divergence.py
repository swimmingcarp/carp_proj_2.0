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
                print(f"底部背离: {date_i} -> {date_j}")
                bottom_divergence_index.append(min_list[j])

    return list(set(bottom_divergence_index) & set(close_index))


def get_peak_divergence_index(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测顶部背离（基于 DIFF）

    条件：
    1. 价格创新高（high[i] < high[j]）
    2. DIFF 下降（diff[i] > diff[j]）
    3. 两点都在零轴上且 MACD 下降
    4. 中间有 MACD 下穿零轴

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        顶部背离日期列表
    """
    peak_divergence_index = []
    close_index = []

    # 找到所有 DIFF 的局部最大值
    max_list = df[(df['diff'] > df['diff'].shift(1)) &
                  (df['diff'] > df['diff'].shift(-1))].index

    if len(max_list) <= 1:
        return peak_divergence_index

    for i in range(len(max_list) - 1):
        for j in range(i + 1, len(max_list)):
            # 查找两个 DIFF 极值点前的价格局部最高点
            ind1, ind2 = 0.0, 0.0

            subdf1 = df.loc[:max_list[i]]
            for k in range(1, min(len(subdf1), 5)):
                if (subdf1['high'].iloc[-k] > subdf1['high'].iloc[-k-1] and
                    subdf1['high'].iloc[-k] > subdf1['high'].iloc[-k+1]):
                    ind1 = subdf1['high'].iloc[-k]
                    break

            subdf2 = df.loc[max_list[i]: max_list[j]]
            for m in range(1, min(len(subdf2), 5)):
                if (subdf2['high'].iloc[-m] > subdf2['high'].iloc[-m-1] and
                    subdf2['high'].iloc[-m] > subdf2['high'].iloc[-m+1]):
                    ind2 = subdf2['high'].iloc[-m]
                    break

            if ind1 == 0.0 or ind2 == 0.0 or ind1 >= ind2:
                continue

            # 背离条件检查
            c1 = (df.loc[max_list[j], 'high'] - df.loc[max_list[i], 'high']) / df.loc[max_list[j], 'high'] > 0.01
            c2 = df.loc[max_list[i], 'diff'] > df.loc[max_list[j], 'diff']
            c3 = df.loc[max_list[i], 'macd'] > 0
            c4 = df.loc[max_list[j], 'macd'] > 0
            c5 = df.loc[max_list[j], 'macd'] < df.loc[max_list[i], 'macd']
            c6 = df.loc[max_list[i]: max_list[j], 'macd'].min() < 0

            tmpdf = df.loc[max_list[i]: max_list[j]]
            tmp_len = len(tmpdf[(tmpdf['close'] > tmpdf['close'].shift(1)) &
                                (tmpdf['close'] > tmpdf['close'].shift(-1))].index)
            c7 = tmp_len != 0

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[max_list[i], 'date'], df.loc[max_list[j], 'date'])
            if diff_days > lookback_days:
                break

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                peak_divergence_index.append(max_list[j])
                close_index.append(subdf2.iloc[-m].name)

    return list(set(close_index))


def get_peak_divergence_index_kdj(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测顶部背离（基于 KDJ）

    条件：
    1. 价格上涨 > 1%
    2. K 值下降
    3. MACD > 0 且下降
    4. 中间有 MACD 下穿零轴

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        顶部背离日期列表
    """
    peak_divergence_index = []
    close_index = []

    # 找到所有 K 值的局部最大值
    max_list = df[(df['k'] > df['k'].shift(1)) &
                  (df['k'] > df['k'].shift(-1))].index

    if len(max_list) <= 1:
        return peak_divergence_index

    for i in range(len(max_list) - 1):
        for j in range(i + 1, len(max_list)):
            # 查找价格局部最高点
            ind1, ind2 = 0.0, 0.0

            subdf1 = df.loc[:max_list[i]]
            for k in range(1, min(len(subdf1), 5)):
                if (subdf1['close'].iloc[-k] > subdf1['close'].iloc[-k-1] and
                    subdf1['close'].iloc[-k] > subdf1['close'].iloc[-k+1]):
                    ind1 = subdf1['close'].iloc[-k]
                    break

            subdf2 = df.loc[max_list[i]: max_list[j]]
            for m in range(1, min(len(subdf2), 5)):
                if (subdf2['close'].iloc[-m] > subdf2['close'].iloc[-m-1] and
                    subdf2['close'].iloc[-m] > subdf2['close'].iloc[-m+1]):
                    ind2 = subdf2['close'].iloc[-m]
                    break

            if ind1 == 0.0 or ind2 == 0.0:
                continue

            if ind1 >= ind2:
                continue

            close_index.append(max_list[j])

            # 背离条件
            c1 = (df.loc[max_list[j], 'close'] - df.loc[max_list[i], 'close']) / df.loc[max_list[j], 'close'] > 0.01
            c2 = df.loc[max_list[i], 'k'] > df.loc[max_list[j], 'k']
            c3 = df.loc[max_list[i], 'macd'] > 0
            c4 = df.loc[max_list[j], 'macd'] > 0
            c5 = df.loc[max_list[j], 'macd'] < df.loc[max_list[i], 'macd']
            c6 = df.loc[max_list[i]: max_list[j], 'macd'].min() < 0

            tmpdf = df.loc[max_list[i]: max_list[j]]
            tmp_len = len(tmpdf[(tmpdf['close'] > tmpdf['close'].shift(1)) &
                                (tmpdf['close'] > tmpdf['close'].shift(-1))].index)
            c7 = tmp_len != 0

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[max_list[i], 'date'], df.loc[max_list[j], 'date'])
            if diff_days > lookback_days:
                break

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                peak_divergence_index.append(max_list[j])

    return list(set(peak_divergence_index) & set(close_index))


def get_peak_divergence_index_kd_variant(df: pd.DataFrame, lookback_days: int = 100) -> list:
    """
    检测顶部背离（KD 变体 - 简化版）

    条件：
    1. 价格创新高
    2. K 值下降
    3. MACD > 0

    Args:
        df: 包含技术指标的 DataFrame
        lookback_days: 回溯天数限制

    Returns:
        顶部背离日期列表
    """
    peak_divergence_index = []

    # 找到所有价格局部最高点
    max_list = df[(df['close'] > df['close'].shift(1)) &
                  (df['close'] > df['close'].shift(-1))].index

    if len(max_list) <= 1:
        return peak_divergence_index

    for i in range(len(max_list) - 1):
        for j in range(i + 1, len(max_list)):
            # 价格创新高，K 值下降
            if df.loc[max_list[i], 'close'] > df.loc[max_list[j], 'close']:
                continue
            if df.loc[max_list[i], 'k'] <= df.loc[max_list[j], 'k']:
                continue

            # MACD > 0
            if df.loc[max_list[j], 'macd'] <= 0:
                continue

            # 时间窗口限制
            diff_days = cal_time_diff(df.loc[max_list[i], 'date'], df.loc[max_list[j], 'date'])
            if diff_days > lookback_days:
                break
            if diff_days <= 5:
                continue

            peak_divergence_index.append(max_list[j])

    return list(set(peak_divergence_index))
