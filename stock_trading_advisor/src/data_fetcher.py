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
import random
from pathlib import Path

from .data_validator import DataValidator
from .market_hours import MarketHours

logger = logging.getLogger(__name__)


class DataFetcher:
    """统一的数据获取接口"""

    def __init__(self, source: str = 'akshare', cache_enabled: bool = True,
                 validate_data: bool = True,
                 max_retries: int = 3, retry_delay: float = 2.0,
                 is_backtest_mode: bool = False):
        """
        初始化数据获取器

        Args:
            source: 数据源 ('akshare', 'tushare', 'yfinance')
            cache_enabled: 是否启用本地缓存
            validate_data: 是否启用数据验证
            max_retries: 最大重试次数
            retry_delay: 重试基础延迟（秒），实际延迟会指数增长
            is_backtest_mode: 是否为回测模式
        """
        self.source = source
        self.cache_enabled = cache_enabled
        self.is_backtest_mode = is_backtest_mode
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
        """
        生成缓存文件路径

        统一使用简化格式：{code}_{adjust}.csv
        不管是回测模式还是实时模式，都使用相同的缓存文件
        """
        filename = f"{code}_{adjust}.csv"
        return Path(self.cache_dir) / filename

    def _load_from_cache(self, cache_path: Path, market: str = 'CN-A',
                         check_freshness: bool = False) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据

        Args:
            cache_path: 缓存文件路径
            market: 市场类型
            check_freshness: 是否检查数据新鲜度

        Returns:
            DataFrame或None
        """
        if not cache_path.exists():
            return None

        try:
            df = pd.read_csv(cache_path)

            # 如果不需要检查新鲜度（回测模式），直接返回
            if not check_freshness:
                logger.info(f"从缓存加载数据: {cache_path}")
                return df

            # 检查缓存是否包含最新交易日数据
            if len(df) == 0:
                logger.debug(f"缓存数据为空: {cache_path}")
                return None

            # 获取缓存中的最后日期
            last_date_in_cache = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d')

            # 获取最近交易日
            latest_trading_date = MarketHours.get_latest_trading_date(market)

            # 如果缓存包含最新交易日数据，认为是新鲜的
            if last_date_in_cache >= latest_trading_date:
                logger.info(f"从缓存加载最新数据: {cache_path} (最后日期: {last_date_in_cache})")
                return df
            else:
                logger.debug(f"缓存数据过期: {cache_path} (缓存: {last_date_in_cache}, 最新: {latest_trading_date})")
                return None

        except Exception as e:
            logger.warning(f"读取缓存失败 {cache_path}: {e}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """
        增量保存数据到缓存

        策略：
        1. 如果缓存文件已存在，加载旧数据
        2. 智能合并：保留缓存中更早的历史数据
        3. 按日期去重（保留最新）
        4. 保存完整的历史数据

        重要：即使新数据的起始日期晚于缓存，也会保留缓存中更早的历史数据

        Args:
            df: 新获取的数据
            cache_path: 缓存文件路径
        """
        try:
            if cache_path.exists():
                # 读取现有缓存
                try:
                    old_df = pd.read_csv(cache_path)

                    if len(old_df) > 0:
                        # 智能合并：保留所有历史数据
                        # 合并新旧数据（old_df 在前，确保保留更早的历史）
                        combined_df = pd.concat([old_df, df], ignore_index=True)

                        # 确保 date 列是字符串格式（统一格式）
                        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')

                        # 按日期去重，保留最后出现的（即新数据）
                        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')

                        # 按日期排序
                        combined_df = combined_df.sort_values('date').reset_index(drop=True)

                        # 计算变化
                        old_count = len(old_df)
                        new_count = len(df)
                        total_count = len(combined_df)
                        added_count = total_count - old_count

                        # 保存合并后的完整数据
                        combined_df.to_csv(cache_path, index=False)

                        # 显示详细信息
                        old_date_range = f"{old_df['date'].iloc[0]} ~ {old_df['date'].iloc[-1]}"
                        new_date_range = f"{combined_df['date'].iloc[0]} ~ {combined_df['date'].iloc[-1]}"

                        if added_count > 0:
                            logger.info(f"增量更新缓存: {cache_path.name}")
                            logger.info(f"  原有: {old_count} 条 ({old_date_range})")
                            logger.info(f"  更新: {total_count} 条 ({new_date_range}), 新增 {added_count} 条")
                        elif added_count == 0:
                            logger.info(f"缓存已是最新: {cache_path.name} ({total_count} 条)")
                        else:
                            # 理论上不应该出现
                            logger.warning(f"缓存记录减少: {cache_path.name} ({old_count} → {total_count})")

                    else:
                        # 旧缓存为空，直接保存新数据
                        df.to_csv(cache_path, index=False)
                        logger.debug(f"缓存为空，保存新数据: {cache_path}")

                except Exception as e:
                    # 读取旧缓存失败，直接覆盖
                    logger.warning(f"读取旧缓存失败 {cache_path}: {e}，将覆盖保存")
                    df.to_csv(cache_path, index=False)
            else:
                # 缓存文件不存在，直接保存
                df.to_csv(cache_path, index=False)
                date_range = f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}"
                logger.info(f"创建新缓存: {cache_path.name} ({len(df)} 条, {date_range})")

        except Exception as e:
            logger.warning(f"保存缓存失败 {cache_path}: {e}")

    def get_k_data(self, code: str, start_date: str = None, end_date: str = None,
                   adjust: str = 'qfq') -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        获取 K 线数据（带数据验证、智能缓存、重试机制）

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

        # 智能缓存策略
        df = None
        should_fetch_new_data = False

        if self.cache_enabled:
            cache_path = self._get_cache_path(code, start_date, end_date, adjust)

            if self.is_backtest_mode:
                # 回测模式：优先使用缓存，不检查新鲜度
                df = self._load_from_cache(cache_path, market, check_freshness=False)
                logger.debug(f"回测模式: 缓存{'命中' if df is not None else '未命中'}")

            else:
                # 实时模式：需要判断是否在交易时间
                is_trading = MarketHours.is_trading_time(market)

                if is_trading:
                    # 交易时间内：盘中数据实时变化，必须从网络获取最新数据
                    logger.info(f"交易时间内，获取盘中最新数据")
                    should_fetch_new_data = True
                else:
                    # 非交易时间：盘已收，检查缓存新鲜度
                    df = self._load_from_cache(cache_path, market, check_freshness=True)

                    if df is None:
                        # 缓存过期或不存在，需要获取新数据
                        logger.info(f"非交易时间，缓存过期或不存在，获取最新数据")
                        should_fetch_new_data = True
                    else:
                        # 缓存新鲜，直接使用
                        logger.info(f"非交易时间，使用新鲜缓存")
                        should_fetch_new_data = False
        else:
            # 缓存未启用，总是获取新数据
            should_fetch_new_data = True

        # 如果需要获取新数据
        if df is None or should_fetch_new_data:
            df = self._fetch_with_retry(code, start_date, end_date, adjust, market)

            # 保存到缓存（如果启用）
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
        """
        获取数据（支持多数据源自动切换）
        注意：对于akshare，重试逻辑已经在 _fetch_akshare 内部实现（多数据源切换）
        """
        try:
            # 根据数据源和市场类型获取数据
            if self.source == 'akshare':
                if market == 'HK':
                    df = self._fetch_akshare_hk(code, start_date, end_date, adjust)
                else:
                    # akshare内部已实现多数据源切换，无需外层重试
                    df = self._fetch_akshare(code, start_date, end_date, adjust)
            elif self.source == 'tushare':
                df = self._fetch_tushare(code, start_date, end_date, adjust)
            elif self.source == 'yfinance':
                df = self._fetch_yfinance(code, start_date, end_date)
            else:
                return None

            if df is not None and len(df) > 0:
                return df
            else:
                logger.warning(f"股票 {code} 数据为空")
                return None

        except Exception as e:
            logger.error(f"获取股票 {code} 数据失败: {e}")
            return None

    def _fetch_akshare(self, code: str, start_date: str, end_date: str,
                       adjust: str) -> pd.DataFrame:
        """使用 AKShare 获取数据，支持多数据源备用"""
        # AKShare 股票代码格式：直接使用 6 位数字代码（如 000001, 600519）
        # 如果代码带有 sh 或 sz 前缀，需要去掉
        original_code = code
        if code.startswith(('sh', 'sz')):
            code = code[2:]  # 去掉前缀

        adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}

        # 定义多个数据源，随机选择以分散负载
        data_sources = [
            ('东方财富', lambda: self.ak.stock_zh_a_hist(
                symbol=code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust=adjust_map.get(adjust, 'qfq')
            )),
            ('新浪财经', lambda: self._fetch_sina(code, start_date, end_date, adjust)),
            ('腾讯财经', lambda: self._fetch_tencent(code, start_date, end_date, adjust)),
            ('Baostock', lambda: self._fetch_baostock(code, start_date, end_date, adjust)),
            # 网易财经：API已下线（502错误），已用Baostock替代
        ]

        # 随机打乱数据源顺序，避免单一网站访问过量
        random.shuffle(data_sources)

        last_error = None
        for source_name, fetch_func in data_sources:
            try:
                logger.info(f"尝试从 {source_name} 获取股票 {code} 数据...")
                df = fetch_func()

                if df is not None and len(df) > 0:
                    # 检查数据是否已经是标准格式
                    if 'date' in df.columns:
                        logger.info(f"✓ 成功从 {source_name} 获取股票 {code} 数据")
                        return df

                    # 如果是东方财富数据，需要重命名
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

                    logger.info(f"✓ 成功从 {source_name} 获取股票 {code} 数据")
                    return df.reset_index(drop=True)

            except Exception as e:
                last_error = e
                error_msg = str(e)
                # 检查是否是连接错误
                if 'RemoteDisconnected' in error_msg or 'Connection' in error_msg:
                    logger.warning(f"✗ {source_name} 连接失败: {error_msg[:100]}")
                else:
                    logger.warning(f"✗ {source_name} 获取失败: {error_msg[:100]}")
                continue

        # 所有数据源都失败
        logger.error(f"所有数据源均失败，无法获取股票 {code} 数据")
        return None

    def _fetch_sina(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """从新浪财经获取数据"""
        try:
            # 新浪财经需要带市场前缀的代码格式: sh600519 或 sz000001
            if code.startswith(('sh', 'sz')):
                symbol = code
            elif code.startswith('6'):
                symbol = 'sh' + code
            else:
                symbol = 'sz' + code

            # 使用 AKShare 的新浪日线数据接口
            adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
            df = self.ak.stock_zh_a_daily(
                symbol=symbol,
                adjust=adjust_map.get(adjust, 'qfq')
            )

            if df is not None and len(df) > 0:
                # 筛选日期范围
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

                # 确保有必要的列
                if 'date' in df.columns:
                    # 选择需要的列
                    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
                    df['code'] = code
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    return df.reset_index(drop=True)
        except AttributeError:
            logger.debug("新浪财经接口不可用")
        except Exception as e:
            logger.debug(f"新浪财经获取失败: {e}")
        return None

    def _fetch_tencent(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """
        从腾讯财经获取数据

        注意：腾讯接口需要带市场标识的代码（如 sz002916）
        """
        try:
            # 腾讯接口需要带市场前缀的代码格式
            if code.startswith(('sh', 'sz')):
                symbol = code  # 已经有前缀
            elif code.startswith('6'):
                symbol = 'sh' + code  # 上海主板
            else:
                symbol = 'sz' + code  # 深圳市场（主板、创业板、中小板）

            # 使用 AKShare 的腾讯数据源接口
            adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
            df = self.ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust=adjust_map.get(adjust, 'qfq')
            )
            if df is not None and len(df) > 0:
                # 腾讯返回的数据已经是标准英文列名
                # 列名: ['date', 'open', 'close', 'high', 'low', 'amount']
                # 注意: 腾讯返回的是 'amount' (成交额) 而不是 'volume' (成交量)

                # 检查是否需要重命名（兼容可能的中文列名）
                if '日期' in df.columns:
                    df = df.rename(columns={
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                    })

                # 处理列名: amount -> volume（保持一致性）
                if 'amount' in df.columns and 'volume' not in df.columns:
                    df['volume'] = df['amount']

                # 确保有必要的列
                if 'date' in df.columns and 'volume' in df.columns:
                    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
                    df['code'] = code
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    return df.reset_index(drop=True)

        except AttributeError:
            logger.debug("腾讯财经接口不可用")
        except Exception as e:
            logger.debug(f"腾讯财经获取失败: {e}")
        return None

    def _fetch_baostock(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """
        从Baostock获取数据

        Baostock数据说明：
        - 免费开源，由证券宝提供
        - 支持前复权、后复权、不复权
        - 数据质量高，更新及时
        - 需要登录/登出操作

        注意：网易财经API已下线（返回502错误），改用Baostock作为第4数据源
        """
        try:
            import baostock as bs

            # 格式化代码
            if code.startswith(('sh', 'sz')):
                clean_code = code[2:]
            else:
                clean_code = code

            # 生成Baostock代码格式
            if clean_code.startswith('6'):
                bs_code = f'sh.{clean_code}'
            else:
                bs_code = f'sz.{clean_code}'

            # 登录（如果未登录）
            if not hasattr(self, '_baostock_logged_in') or not self._baostock_logged_in:
                lg = bs.login()
                if lg.error_code != '0':
                    logger.debug(f"Baostock登录失败: {lg.error_msg}")
                    return None
                self._baostock_logged_in = True

            # 复权类型映射
            adjust_map = {'qfq': '2', 'hfq': '1', '': '3'}
            adjustflag = adjust_map.get(adjust, '2')

            # 查询数据（注意：Baostock需要带分隔符的日期格式 YYYY-MM-DD）
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,  # 直接使用 YYYY-MM-DD 格式
                end_date=end_date,      # 直接使用 YYYY-MM-DD 格式
                frequency='d',
                adjustflag=adjustflag
            )

            if rs.error_code != '0':
                logger.debug(f"Baostock查询失败: {rs.error_msg}")
                return None

            # 转换为DataFrame
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 标准化列名和数据类型
            df['code'] = clean_code

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'code']]
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            return df.reset_index(drop=True)

        except ImportError:
            logger.debug("Baostock未安装 (pip install baostock)")
            return None
        except Exception as e:
            logger.debug(f"Baostock获取失败: {e}")
            return None


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
        获取股票基本信息（支持多数据源备用）

        Args:
            code: 股票代码

        Returns:
            包含股票名称、行业等信息的字典
        """
        if self.source != 'akshare':
            return None

        # 格式化代码（去掉前缀）
        clean_code = code
        if code.startswith(('sh', 'sz')):
            clean_code = code[2:]

        # 定义多个数据源，随机选择以分散负载
        data_sources = [
            ('东方财富个股信息', lambda: self._get_stock_info_em(clean_code)),
            ('东方财富实时行情', lambda: self._get_stock_info_from_spot(clean_code)),
            ('股票名称缓存', lambda: self._get_stock_name_from_cache(clean_code)),
        ]

        # 随机打乱数据源顺序，避免单一网站访问过量
        random.shuffle(data_sources)

        last_error = None
        for source_name, fetch_func in data_sources:
            try:
                logger.info(f"尝试从 {source_name} 获取股票 {clean_code} 信息...")
                info = fetch_func()

                if info is not None and len(info) > 0:
                    logger.info(f"✓ 成功从 {source_name} 获取股票 {clean_code} 信息")
                    return info

            except Exception as e:
                last_error = e
                error_msg = str(e)
                if 'RemoteDisconnected' in error_msg or 'Connection' in error_msg:
                    logger.warning(f"✗ {source_name} 连接失败: {error_msg[:100]}")
                else:
                    logger.warning(f"✗ {source_name} 获取失败: {error_msg[:100]}")
                continue

        # 所有数据源都失败，返回最基本的信息
        logger.warning(f"所有数据源均失败，使用默认信息: {clean_code}")
        return {'股票简称': clean_code, '股票代码': clean_code}

    def _get_stock_info_em(self, code: str) -> Optional[dict]:
        """从东方财富获取个股详细信息"""
        try:
            df = self.ak.stock_individual_info_em(symbol=code)
            info = {}
            for _, row in df.iterrows():
                info[row['item']] = row['value']
            return info if len(info) > 0 else None
        except Exception as e:
            logger.debug(f"东方财富个股信息接口失败: {e}")
            return None

    def _get_stock_info_from_spot(self, code: str) -> Optional[dict]:
        """从东方财富实时行情获取股票名称"""
        try:
            # 尝试使用个股实时行情接口（更轻量）
            df = self.ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == code]

            if len(stock_data) > 0:
                row = stock_data.iloc[0]
                return {
                    '股票简称': row['名称'],
                    '股票代码': code,
                    '最新价': row['最新价'],
                    '涨跌幅': row['涨跌幅'],
                }
            return None
        except Exception as e:
            logger.debug(f"东方财富实时行情接口失败: {e}")
            # 尝试从历史数据中获取股票名称
            try:
                df_hist = self.ak.stock_zh_a_hist(
                    symbol=code,
                    start_date='20250101',
                    end_date='20251231',
                    adjust='qfq'
                )
                if df_hist is not None and len(df_hist) > 0:
                    # 历史数据中没有名称，使用stock_individual_info_em
                    return None
            except:
                pass
            return None

    def _get_stock_name_from_cache(self, code: str) -> Optional[dict]:
        """从缓存或静态映射获取股票名称"""
        # 常见股票的静态映射表
        stock_names = {
            '600519': '贵州茅台', '601318': '中国平安', '600036': '招商银行',
            '000858': '五粮液', '000651': '格力电器', '601398': '工商银行',
            '600276': '恒瑞医药', '000333': '美的集团', '002415': '海康威视',
            '600887': '伊利股份', '000002': '万科A', '600030': '中信证券',
            '601166': '兴业银行', '600016': '民生银行', '000001': '平安银行',
            '600000': '浦发银行', '601328': '交通银行', '601288': '农业银行',
            '601939': '建设银行', '601988': '中国银行', '600885': '宏发股份',
            '000034': '神州数码', '002920': '德赛西威', '300293': '蓝英装备',
            '300476': '胜宏科技', '001279': '强邦新材', '605117': '德业股份',
            '603501': '韦尔股份', '688981': '中芯国际',
        }

        name = stock_names.get(code)
        if name:
            logger.info(f"✓ 从静态映射获取股票 {code} 名称: {name}")
            return {
                '股票简称': name,
                '股票代码': code,
            }

        # 如果不在映射表中，使用代码作为名称（最后的保底方案）
        logger.debug(f"股票 {code} 不在名称映射表中，使用代码作为名称")
        return {
            '股票简称': code,
            '股票代码': code,
        }
