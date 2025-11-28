"""
Stock Trading Advisor
基于技术指标的股票交易策略系统
"""

__version__ = '1.0.0'
__author__ = 'Stock Trading Advisor Team'

from .indicators import (
    ma_indicator,
    ema_indicator,
    macd_indicator,
    kdj_indicator,
    p_change_indicator,
    calculate_all_indicators
)

from .divergence import (
    get_bottom_divergence_index,
    get_peak_divergence_index,
    get_peak_divergence_index_kdj,
    get_peak_divergence_index_kd_variant
)

from .data_fetcher import DataFetcher
from .strategy import MixedStrategy
from .new_strategy import RSITrendStrategy
from .analyzer import SignalAnalyzer

__all__ = [
    'DataFetcher',
    'MixedStrategy',
    'RSITrendStrategy',
    'SignalAnalyzer',
    'calculate_all_indicators',
    'get_bottom_divergence_index',
    'get_peak_divergence_index',
]
