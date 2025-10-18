"""
股票数据获取模块
支持多个数据源：
- akshare (推荐，开源免费，支持A股、港股)
- tushare (需要积分，支持A股)
- yfinance (国际市场，支持港股、美股)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import logging
import time
import os
import hashlib
from pathlib import Path

from .data_validator import DataValidator

logger = logging.getLogger(__name__)


class DataFetcher:
    """统一的数据获取接口"""

    def __init__(self, source: str = 'akshare', cache_enabled: bool = True,
                 validate_data: bool = True,
                 max_retries: int = 3, retry_delay: float = 2.0):
        """
        初始化数据获取器

        Args:
            source: 数据源 ('akshare', 'tushare', 'yfinance')
            cache_enabled: 是否启用本地缓存
            validate_data: 是否启用数据验证
            max_retries: 最大重试次数
            retry_delay: 重试基础延迟（秒），实际延迟会指数增长
        """
        self.source = source
        self.cache_enabled = cache_enabled
        # 使用绝对路径，确保缓存目录固定
        script_dir = Path(__file__).parent.parent  # src的父目录，即stock_trading_advisor
        self.cache_dir = script_dir / 'data' / 'cache'
        self.validate_data = validate_data
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 确保缓存目录存在
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _detect_market(self, code: str) -> str:
        """
        检测市场类型

        Args:
            code: 股票代码

        Returns:
            市场类型: 'CN-A' (A股), 'HK' (港股), 'US' (美股)
        """
        # 港股代码检测
        if code.isdigit():
            code_num = int(code)
            # 港股代码范围: 00001-99999 (5位数字)
            if 1 <= code_num <= 99999 and len(code) == 5:
                return 'HK'
            # A股代码: 6位数字
            elif len(code) == 6:
                return 'CN-A'

        # 带后缀的港股代码
        if code.endswith('.HK'):
            return 'HK'

        # 带前缀的A股代码
        if code.startswith(('sh', 'sz', 'SH', 'SZ')):
            return 'CN-A'

        # 带后缀的A股代码
        if code.endswith(('.SH', '.SZ')):
            return 'CN-A'

        # 美股代码（字母开头）
        if code[0].isalpha():
            return 'US'

        # 默认返回A股
        return 'CN-A'

    def _get_cache_path(self, code: str, start_date: str, end_date: str, adjust: str) -> Path:
        """生成缓存文件路径"""
        # 使用参数生成唯一的缓存文件名
        cache_key = f"{code}_{start_date}_{end_date}_{adjust}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        filename = f"{code}_{cache_hash}.csv"
        return Path(self.cache_dir) / filename

    def _load_from_cache(self, cache_path: Path, max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        if not cache_path.exists():
            return None

        # 检查缓存是否过期
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - file_mtime > timedelta(days=max_age_days):
            logger.debug(f"缓存已过期: {cache_path}")
            return None

        try:
            df = pd.read_csv(cache_path)
            logger.info(f"从缓存加载数据: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"读取缓存失败 {cache_path}: {e}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """保存数据到缓存"""
        try:
            df.to_csv(cache_path, index=False)
            logger.debug(f"数据已缓存: {cache_path}")
        except Exception as e:
            logger.warning(f"保存缓存失败 {cache_path}: {e}")

    def get_k_data(self, code: str, start_date: str = None, end_date: str = None,
                   adjust: str = 'qfq') -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        获取 K 线数据（带数据验证、缓存、重试机制）

        Args:
            code: 股票代码
                  - A股: '000001' 或 'sh000001' (6位数字)
                  - 港股: '00700' 或 '00700.HK' (5位数字)
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

        # 检测市场类型
        market = self._detect_market(code)
        logger.info(f"检测到市场类型: {market}, 股票代码: {code}")

        # 尝试从缓存加载
        df = None
        if self.cache_enabled:
            cache_path = self._get_cache_path(code, start_date, end_date, adjust)
            df = self._load_from_cache(cache_path)

        # 如果缓存未命中，进行网络请求（带重试）
        if df is None:
            df = self._fetch_with_retry(code, start_date, end_date, adjust, market)

            # 保存到缓存
            if df is not None and self.cache_enabled:
                cache_path = self._get_cache_path(code, start_date, end_date, adjust)
                self._save_to_cache(df, cache_path)

        if df is None:
            return None, None

        # 数据验证（传入市场类型）
        if self.validate_data and self.validator:
            df, report = self.validator.validate(df, code, market=market)
            return df, report
        else:
            return df, None

    def _fetch_with_retry(self, code: str, start_date: str, end_date: str,
                          adjust: str, market: str = 'CN-A') -> Optional[pd.DataFrame]:
        """带指数退避的重试机制"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # 根据数据源和市场类型获取数据
                if self.source == 'akshare':
                    if market == 'HK':
                        df = self._fetch_akshare_hk(code, start_date, end_date, adjust)
                    else:
                        df = self._fetch_akshare(code, start_date, end_date, adjust)
                elif self.source == 'tushare':
                    df = self._fetch_tushare(code, start_date, end_date, adjust)
                elif self.source == 'yfinance':
                    df = self._fetch_yfinance(code, start_date, end_date)
                else:
                    return None

                if df is not None and len(df) > 0:
                    logger.info(f"成功获取股票 {code} 数据")
                    return df
                else:
                    logger.warning(f"股票 {code} 数据为空")
                    return None

            except Exception as e:
                last_exception = e
                logger.warning(f"获取股票 {code} 数据失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

                # 如果不是最后一次尝试，进行指数退避
                if attempt < self.max_retries - 1:
                    # 指数退避: delay * (2 ^ attempt)
                    backoff_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"等待 {backoff_time:.1f} 秒后重试...")
                    time.sleep(backoff_time)

        # 所有重试都失败
        logger.error(f"获取股票 {code} 数据失败，已重试 {self.max_retries} 次: {last_exception}")
        return None

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

    def _fetch_akshare_hk(self, code: str, start_date: str, end_date: str,
                          adjust: str) -> pd.DataFrame:
        """
        使用 AKShare 获取港股数据

        Args:
            code: 港股代码 (5位数字，如 '00700' 或带后缀 '00700.HK')
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型 ('qfq'-前复权, 'hfq'-后复权, ''-不复权)

        Returns:
            标准格式的DataFrame
        """
        # 格式化港股代码：确保是5位数字，去掉 .HK 后缀
        original_code = code
        if code.endswith('.HK'):
            code = code[:-3]

        # 确保代码是5位数字（前面补0）
        if code.isdigit():
            code = code.zfill(5)

        logger.info(f"获取港股数据: {code} (原始代码: {original_code})")

        try:
            # 使用 AKShare 的港股历史数据接口
            # stock_hk_hist: 获取港股历史行情数据
            adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
            df = self.ak.stock_hk_hist(
                symbol=code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust=adjust_map.get(adjust, 'qfq')
            )

            if df is None or len(df) == 0:
                logger.warning(f"港股 {code} 数据为空")
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

            logger.info(f"成功获取港股 {code} 数据，共 {len(df)} 条")
            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取港股 {code} 数据失败: {e}")
            # 如果 AKShare 失败，尝试使用 yfinance 作为备用
            logger.info(f"尝试使用 yfinance 获取港股 {code} 数据")
            try:
                # yfinance 需要 .HK 后缀
                yf_code = f"{code}.HK"
                import yfinance as yf
                ticker = yf.Ticker(yf_code)
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

                logger.info(f"使用 yfinance 成功获取港股 {code} 数据，共 {len(df)} 条")
                return df.reset_index(drop=True)

            except Exception as yf_error:
                logger.error(f"yfinance 也无法获取港股 {code} 数据: {yf_error}")
                return None


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
