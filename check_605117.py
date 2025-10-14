#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 605117 的跌停情况
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import DataFetcher
import pandas as pd

# 获取数据
fetcher = DataFetcher(source='akshare', validate_data=False)
df, _ = fetcher.get_k_data('605117', start_date='2020-01-01')

if df is not None:
    # 计算涨跌幅
    df['p_change_temp'] = df['close'].pct_change() * 100

    # 找出跌幅 >= 15% 的日期
    big_drops = df[df['p_change_temp'] <= -15.0]

    print(f"\n找到 {len(big_drops)} 天跌幅 >= 15%:")
    print("=" * 60)

    if len(big_drops) > 0:
        for idx, row in big_drops.iterrows():
            print(f"日期: {row['date']}, 跌幅: {row['p_change_temp']:.2f}%, 收盘价: {row['close']:.2f}")

    # 统计最近的大跌
    recent_drops = df[df['p_change_temp'] <= -10.0].tail(10)
    print(f"\n最近10次跌幅 >= 10%:")
    print("=" * 60)
    for idx, row in recent_drops.iterrows():
        print(f"日期: {row['date']}, 跌幅: {row['p_change_temp']:.2f}%, 收盘价: {row['close']:.2f}")

    # 检查最新数据
    latest = df.iloc[-10:]
    print(f"\n最近10天数据:")
    print("=" * 60)
    print(latest[['date', 'close', 'p_change_temp']].to_string(index=False))
