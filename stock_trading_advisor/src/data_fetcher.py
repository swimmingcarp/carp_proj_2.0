"""
股票数据获取模块
支持多个数据源：
- akshare (推荐，开源免费)
- tushare (需要积分)
- yfinance (国际市场)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import logging

from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class DataFetcher:
    """统一的数据获取接口"""

    def __init__(self, source: str = 'akshare', cache_enabled: bool = True,
                 validate_data: bool = True):
        """
        初始化数据获取器

        Args:
            source: 数据源 ('akshare', 'tushare', 'yfinance')
            cache_enabled: 是否启用本地缓存
            validate_data: 是否启用数据验证
        """
        self.source = source
        self.cache_enabled = cache_enabled
        self.cache_dir = 'data/cache'
        self.validate_data = validate_data

        # 初始化数据验证器
        if self.validate_data:
            self.validator = DataValidator()
        else:
            self.validator = None

        # 根据数据源初始化
        if source == 'akshare':
            try:
                import akshare as ak
                self.ak = ak
                logger.info("使用 AKShare 数据源")
            except ImportError:
                raise ImportError("请安装 akshare: pip install akshare")

        elif source == 'tushare':
            try:
                import tushare as ts
                self.ts = ts
                logger.info("使用 Tushare 数据源")
            except ImportError:
                raise ImportError("请安装 tushare: pip install tushare")

        elif source == 'yfinance':
            try:
                import yfinance as yf
                self.yf = yf
                logger.info("使用 yfinance 数据源")
            except ImportError:
                raise ImportError("请安装 yfinance: pip install yfinance")

        else:
            raise ValueError(f"不支持的数据源: {source}")

    def get_k_data(self, code: str, start_date: str = None, end_date: str = None,
                   adjust: str = 'qfq') -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        获取 K 线数据（带数据验证）

        Args:
            code: 股票代码（如 '000001' 或 'sh000001'）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            adjust: 复权类型 ('qfq'-前复权, 'hfq'-后复权, ''-不复权)

        Returns:
            (包含 date, open, close, high, low, volume, code 的 DataFrame, 验证报告)
            如果不启用验证，返回 (DataFrame, None)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            # 默认获取 2 年数据
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        try:
            if self.source == 'akshare':
                df = self._fetch_akshare(code, start_date, end_date, adjust)
            elif self.source == 'tushare':
                df = self._fetch_tushare(code, start_date, end_date, adjust)
            elif self.source == 'yfinance':
                df = self._fetch_yfinance(code, start_date, end_date)
            else:
                return None, None

            # 数据验证
            if self.validate_data and self.validator and df is not None:
                df, report = self.validator.validate(df, code)
                return df, report
            else:
                return df, None

        except Exception as e:
            logger.error(f"获取股票 {code} 数据失败: {e}")
            return None, None

    def _fetch_akshare(self, code: str, start_date: str, end_date: str,
                       adjust: str) -> pd.DataFrame:
        """使用 AKShare 获取数据"""
        # AKShare 股票代码格式：直接使用 6 位数字代码（如 000001, 600519）
        # 如果代码带有 sh 或 sz 前缀，需要去掉
        original_code = code
        if code.startswith(('sh', 'sz')):
            code = code[2:]  # 去掉前缀

        # 获取历史行情数据
        adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
        df = self.ak.stock_zh_a_hist(
            symbol=code,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            adjust=adjust_map.get(adjust, 'qfq')
        )

        if df is None or len(df) == 0:
            return None

        # 重命名列以匹配标准格式
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'pct_change'
        })

        # 选择需要的列
        df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
        df['code'] = code
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        return df.reset_index(drop=True)

    def _fetch_tushare(self, code: str, start_date: str, end_date: str,
                       adjust: str) -> pd.DataFrame:
        """使用 Tushare 获取数据"""
        # Tushare 需要设置 token
        # ts.set_token('your_token')
        pro = self.ts.pro_api()

        # Tushare 代码格式：000001.SZ
        if '.' not in code:
            if code.startswith('6'):
                code = code + '.SH'
            else:
                code = code + '.SZ'

        # 获取数据
        df = pro.daily(
            ts_code=code,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )

        if df is None or len(df) == 0:
            return None

        # 复权处理
        if adjust in ['qfq', 'hfq']:
            adj_factor = pro.adj_factor(ts_code=code, start_date=start_date.replace('-', ''))
            df = pd.merge(df, adj_factor[['trade_date', 'adj_factor']], on='trade_date', how='left')
            df['adj_factor'].fillna(method='ffill', inplace=True)

            if adjust == 'qfq':
                df['open'] = df['open'] * df['adj_factor']
                df['close'] = df['close'] * df['adj_factor']
                df['high'] = df['high'] * df['adj_factor']
                df['low'] = df['low'] * df['adj_factor']

        # 重命名列
        df = df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume'
        })

        df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
        df['code'] = code
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        return df.sort_values('date').reset_index(drop=True)

    def _fetch_yfinance(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用 yfinance 获取数据（主要用于港股、美股）"""
        ticker = self.yf.Ticker(code)
        df = ticker.history(start=start_date, end=end_date)

        if df is None or len(df) == 0:
            return None

        # 重命名列
        df = df.rename(columns={
            'Open': 'open',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })

        df['date'] = df.index.strftime('%Y-%m-%d')
        df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
        df['code'] = code

        return df.reset_index(drop=True)

    def get_realtime_data(self, code: str) -> Optional[dict]:
        """
        获取实时行情数据

        Args:
            code: 股票代码

        Returns:
            包含实时价格、涨跌幅等信息的字典
        """
        try:
            if self.source == 'akshare':
                # 格式化代码
                if not code.startswith(('sh', 'sz')):
                    if code.startswith('6'):
                        code = 'sh' + code
                    else:
                        code = 'sz' + code

                # 获取实时数据
                df = self.ak.stock_zh_a_spot_em()
                stock_data = df[df['代码'] == code.replace('sh', '').replace('sz', '')]

                if len(stock_data) == 0:
                    return None

                row = stock_data.iloc[0]
                return {
                    'code': code,
                    'name': row['名称'],
                    'price': row['最新价'],
                    'change': row['涨跌额'],
                    'pct_change': row['涨跌幅'],
                    'volume': row['成交量'],
                    'amount': row['成交额'],
                    'high': row['最高'],
                    'low': row['最低'],
                    'open': row['今开'],
                    'last_close': row['昨收']
                }

        except Exception as e:
            logger.error(f"获取实时数据失败 {code}: {e}")
            return None

    def get_stock_info(self, code: str) -> Optional[dict]:
        """
        获取股票基本信息

        Args:
            code: 股票代码

        Returns:
            包含股票名称、行业等信息的字典
        """
        try:
            if self.source == 'akshare':
                df = self.ak.stock_individual_info_em(symbol=code)
                info = {}
                for _, row in df.iterrows():
                    info[row['item']] = row['value']
                return info
        except Exception as e:
            logger.error(f"获取股票信息失败 {code}: {e}")
            return None
