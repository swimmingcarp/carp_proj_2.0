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

import requests

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

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path, source_name: str = None):
        """
        增量保存数据到缓存

        策略：
        1. 如果缓存文件已存在，加载旧数据
        2. 智能合并：保留缓存中更早的历史数据
        3. 按日期去重（保留最新）
        4. 保存完整的历史数据
        5. 港股特殊处理：
           - 新浪财经：只增量更新
           - 东方财富/Yahoo：检查并更新历史数据（修正新浪的差异）

        重要：即使新数据的起始日期晚于缓存，也会保留缓存中更早的历史数据

        Args:
            df: 新获取的数据
            cache_path: 缓存文件路径
            source_name: 数据源名称（如 '东方财富', '新浪财经', 'Yahoo Finance'）
        """
        try:
            if cache_path.exists():
                # 读取现有缓存
                try:
                    old_df = pd.read_csv(cache_path)

                    if len(old_df) > 0:
                        # 检测是否是港股（通过代码判断）
                        is_hk_stock = False
                        if 'code' in old_df.columns and len(old_df) > 0:
                            code = str(old_df['code'].iloc[0])
                            # 港股：1-5位数字
                            if code.isdigit() and len(code) <= 5:
                                is_hk_stock = True

                        # 检测旧缓存的数据源
                        old_source = None
                        if 'source' in old_df.columns:
                            old_source = old_df['source'].iloc[0]

                        # 港股特殊处理
                        if is_hk_stock and source_name:
                            # 添加source列到新数据
                            df_with_source = df.copy()
                            df_with_source['source'] = source_name

                            # 判断合并策略
                            if source_name == '新浪财经':
                                # 新浪财经：只增量更新（保留旧数据的历史部分）
                                logger.info(f"港股增量策略: 新浪财经数据，保留历史")
                                combined_df = pd.concat([old_df, df_with_source], ignore_index=True)
                                combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
                                # 去重时保留最新的（新数据优先）
                                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')

                            else:
                                # 东方财富/Yahoo：检查并更新历史数据（仅过去2年）
                                logger.info(f"港股修正策略: {source_name}，检查过去2年内的差异数据")

                                # 统一日期格式
                                old_df['date'] = pd.to_datetime(old_df['date']).dt.strftime('%Y-%m-%d')
                                df_with_source['date'] = pd.to_datetime(df_with_source['date']).dt.strftime('%Y-%m-%d')

                                # 计算2年前的日期
                                from datetime import datetime, timedelta
                                two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

                                # 找出重叠的日期（只考虑过去2年）
                                new_dates = set(df_with_source['date'])
                                old_dates_recent = set(old_df[old_df['date'] >= two_years_ago]['date'])
                                overlap_dates = new_dates & old_dates_recent

                                if len(overlap_dates) > 0:
                                    # 检查重叠日期的数据是否有差异
                                    updates_needed = []

                                    for date in overlap_dates:
                                        old_row = old_df[old_df['date'] == date].iloc[0]
                                        new_row = df_with_source[df_with_source['date'] == date].iloc[0]

                                        # 比较收盘价（允许0.01的误差）
                                        if abs(float(old_row['close']) - float(new_row['close'])) > 0.01:
                                            updates_needed.append(date)

                                    if len(updates_needed) > 0:
                                        logger.info(f"  发现 {len(updates_needed)}/{len(overlap_dates)} 个日期数据不同，用{source_name}数据更新")
                                        # 删除旧数据中需要更新的日期
                                        old_df_filtered = old_df[~old_df['date'].isin(updates_needed)]
                                        # 只保留需要更新的新数据
                                        df_to_merge = df_with_source[df_with_source['date'].isin(updates_needed)]
                                        # 保留新增的日期（不在旧数据中的）
                                        new_only_dates = new_dates - old_dates_recent - set(old_df['date'])
                                        if len(new_only_dates) > 0:
                                            df_new_only = df_with_source[df_with_source['date'].isin(new_only_dates)]
                                            df_to_merge = pd.concat([df_to_merge, df_new_only], ignore_index=True)

                                        # 合并：保留的旧数据 + 更新的数据 + 新增的数据
                                        combined_df = pd.concat([old_df_filtered, df_to_merge], ignore_index=True)
                                    else:
                                        logger.info(f"  检查了 {len(overlap_dates)} 个重叠日期，数据一致，无需更新")
                                        # 只添加新日期
                                        new_only_dates = new_dates - set(old_df['date'])
                                        if len(new_only_dates) > 0:
                                            df_new_only = df_with_source[df_with_source['date'].isin(new_only_dates)]
                                            combined_df = pd.concat([old_df, df_new_only], ignore_index=True)
                                        else:
                                            combined_df = old_df
                                else:
                                    # 没有重叠（只有新日期），直接合并
                                    logger.info(f"  无重叠日期，只添加新数据")
                                    combined_df = pd.concat([old_df, df_with_source], ignore_index=True)

                                # 确保没有重复
                                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')

                        else:
                            # A股或没有source_name：使用原有逻辑
                            combined_df = pd.concat([old_df, df], ignore_index=True)
                            combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
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
                            # 可能是修正了历史数据
                            logger.info(f"缓存数据已修正: {cache_path.name} ({old_count} → {total_count})")


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

        # 智能缓存策略
        df = None
        should_fetch_new_data = False
        # 实时数据获取状态，用于上层区分网络/实时失败场景
        realtime_failed = False

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
                # 从DataFrame中提取source_name（如果有）
                source_name = None
                if 'source' in df.columns:
                    source_name = df['source'].iloc[0] if len(df) > 0 else None
                self._save_to_cache(df, cache_path, source_name)

        if df is None:
            # 日线数据获取失败，直接返回；上层根据 df=None 判断
            return None, None

        # 实时模式：用分钟级实时行情更新当日K线
        # 目的：避免依赖日K历史接口的「最新一行」，确保最新价格来自 get_realtime_data
        if not self.is_backtest_mode:
            try:
                # 判断日线是否已经包含当日数据
                today_str = datetime.now().strftime('%Y-%m-%d')
                has_today_k = False
                if 'date' in df.columns and len(df) > 0:
                    last_date = pd.to_datetime(df['date'].iloc[-1]).strftime('%Y-%m-%d')
                    has_today_k = (last_date == today_str)

                # 当前是否在交易时间（用于分钟级实时数据的调用控制）
                is_trading_now = MarketHours.is_trading_time(market)

                # 逻辑调整：
                # - 只在“交易时间内 + 当日K线还不存在”时，才尝试调用分钟级实时接口
                # - 非交易时间不再请求分钟级数据，避免无意义的实时调用
                # - 如果日线已经包含最新交易日，则不再调用分钟级接口，直接跳过
                # - 是否标记为 REALTIME_FAILED 则仍然只在「交易时间内」处理，避免盘后误报
                if not has_today_k and is_trading_now:
                    realtime = self.get_realtime_data(code)

                    # realtime 结构参考 get_realtime_data 返回
                    if realtime and realtime.get('price') is not None:
                        today_str = datetime.now().strftime('%Y-%m-%d')

                        # 统一日期格式为字符串 YYYY-MM-DD
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                        open_price = realtime.get('open', realtime['price'])
                        high_price = realtime.get('high', realtime['price'])
                        low_price = realtime.get('low', realtime['price'])
                        close_price = realtime['price']
                        volume = realtime.get('volume', 0)

                        new_row = {
                            'date': today_str,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume,
                            'code': code,
                        }

                        if len(df) > 0 and str(df.iloc[-1]['date']) == today_str:
                            # 已经有当日K线（部分数据源会在盘中生成），用实时数据覆盖最后一行
                            last_idx = df.index[-1]
                            for col in ['open', 'high', 'low', 'close', 'volume', 'code']:
                                df.at[last_idx, col] = new_row[col]
                        else:
                            # 追加一根当日的临时K线
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                    else:
                        # 只有在交易时间内未能获取到有效实时价格时，才标记为实时失败
                        if is_trading_now:
                            realtime_failed = True
                else:
                    # 非交易时间或日线已经包含最新交易日，分钟K线获取逻辑跳过
                    if has_today_k:
                        summary_line = f"分钟K线获取[{code}]：日线已经包含最新数据，跳过！"
                    else:
                        summary_line = f"分钟K线获取[{code}]：当前非交易时间，跳过！"
                    logger.info(summary_line)

            except Exception as e:
                # 实时更新失败不影响整体流程，只记录警告
                logger.warning(f"实时模式更新当日K线失败 {code}: {e}")
                if MarketHours.is_trading_time(market):
                    realtime_failed = True

        # 数据验证（传入市场类型）
        extra_report = {}
        if realtime_failed and MarketHours.is_trading_time(market):
            # 标记为实时数据获取失败，让上层在交易时间内避免使用过期数据发出信号
            extra_report['net_status'] = 'REALTIME_FAILED'

        if self.validate_data and self.validator:
            df, report = self.validator.validate(df, code, market=market)
            if extra_report:
                if report is None:
                    report = {}
                report.update(extra_report)
            return df, report
        else:
            # 如果不启用数据验证，但需要传递实时失败信息，也通过第二个返回值返回
            if extra_report:
                return df, extra_report
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
        """使用 AKShare 获取 A 股日 K 数据，支持多数据源备用"""
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
        total_sources = len(data_sources)
        summary_parts = []
        result_df = None

        for idx, (source_name, fetch_func) in enumerate(data_sources, start=1):
            try:
                logger.debug(
                    f"[DAILY_K] 尝试从 {source_name} 获取 A股 {original_code} 日K 数据 "
                    f"（源 {idx}/{total_sources}）"
                )
                df = fetch_func()

                if df is not None and len(df) > 0:
                    # 检查数据是否已经是标准格式
                    if 'date' in df.columns:
                        summary_parts.append(f"{source_name}（成功！{len(df)}条）")
                        result_df = df
                        break

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

                    summary_parts.append(f"{source_name}（成功！{len(df)}条）")
                    result_df = df.reset_index(drop=True)
                    break

                # df 为空视为失败（详细原因写入调试日志，汇总通过 summary_parts 打印）
                logger.debug(
                    f"[DAILY_K] ✗ {source_name} 返回空的 A股 {original_code} 日K 数据"
                )
                summary_parts.append(f"{source_name}（失败：空数据）")

            except Exception as e:
                last_error = e
                error_msg = str(e)
                # 检查是否是连接错误
                if 'RemoteDisconnected' in error_msg or 'Connection' in error_msg:
                    err_short = error_msg[:80]
                    logger.debug(
                        f"[DAILY_K] ✗ {source_name} 获取 A股 {original_code} 日K 数据连接失败: "
                        f"{err_short}"
                    )
                    # 汇总信息只保留高层含义，避免打印过长的异常细节
                    summary_parts.append(f"{source_name}（失败：网络异常）")
                else:
                    err_short = error_msg[:80]
                    logger.debug(
                        f"[DAILY_K] ✗ {source_name} 获取 A股 {original_code} 日K 数据失败: "
                        f"{err_short}"
                    )
                    summary_parts.append(f"{source_name}（失败：其它异常）")

        # 汇总打印：日线K线获取：源A（结果） -> 源B（结果） -> ...
        if summary_parts:
            summary_line = f"日线K线获取[{original_code}]：" + " -> ".join(summary_parts)
        else:
            summary_line = f"日线K线获取[{original_code}]：未尝试任何数据源"

        logger.info(summary_line)

        if result_df is not None:
            return result_df

        # 所有数据源都失败
        msg_final = f"[DAILY_K] 所有数据源均失败，无法获取 A股 {original_code} 日K 数据"
        logger.error(msg_final)
        return None

    def _fetch_sina(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """从新浪财经获取 A 股日 K 数据"""
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
            else:
                logger.debug(f"[DAILY_K] ✗ 新浪财经 返回空的 A股 {code} 日K 数据")
        except AttributeError:
            logger.debug("新浪财经接口不可用")
        except Exception as e:
            logger.debug(f"新浪财经获取失败: {e}")
        return None

    def _fetch_tencent(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """
        从腾讯财经获取 A 股日 K 数据

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
            else:
                logger.debug(f"[DAILY_K] ✗ 腾讯财经 返回空的 A股 {code} 日K 数据")

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
        """使用 Tushare 获取 A 股日 K 数据"""
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
            logger.warning(f"[DAILY_K] ✗ Tushare 返回空的 A股 {code} 日K 数据")
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

        df = df.sort_values('date').reset_index(drop=True)

        summary_line = f"日线K线获取[{code}]：Tushare（成功！{len(df)}条）"
        logger.info(summary_line)
        return df


    def _fetch_yfinance(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用 yfinance 获取日 K 数据（主要用于港股、美股）"""
        # 根据市场类型对代码进行适配，特别是港股需要转换为 4 位数字 + '.HK'
        market = self._detect_market(code)
        yf_code = code
        if market == 'HK':
            yf_code = self._to_yahoo_hk_symbol(code)

        ticker = self.yf.Ticker(yf_code)
        df = ticker.history(start=start_date, end=end_date)

        if df is None or len(df) == 0:
            logger.warning(f"[DAILY_K] ✗ yfinance 返回空的 {yf_code} 日K 数据")
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
        # 这里的 code 字段仍然保留调用者传入的代码形式，便于与外部配置对齐
        df['code'] = code
        df['source'] = 'Yahoo Finance'  # 添加数据源标记

        df = df.reset_index(drop=True)
        summary_line = f"日线K线获取[{code}]：yfinance（成功！{len(df)}条，Yahoo代码: {yf_code}）"
        logger.info(summary_line)
        return df

    def _fetch_akshare_hk(self, code: str, start_date: str, end_date: str,
                          adjust: str) -> pd.DataFrame:
        """
        使用 AKShare 获取港股日 K 数据（支持多数据源备用）

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

        # 定义多个数据源，随机选择以分散负载
        import random
        data_sources = [
            ('东方财富', lambda: self._fetch_hk_eastmoney(code, start_date, end_date, adjust)),
            ('新浪财经', lambda: self._fetch_hk_sina(code, start_date, end_date, adjust)),
            ('Yahoo Finance', lambda: self._fetch_hk_yfinance(code, start_date, end_date)),
        ]

        # 随机打乱数据源顺序，避免单一网站访问过量
        random.shuffle(data_sources)

        last_error = None
        total_sources = len(data_sources)
        summary_parts = []
        result_df = None

        for idx, (source_name, fetch_func) in enumerate(data_sources, start=1):
            try:
                logger.debug(
                    f"[DAILY_K] 尝试从 {source_name} 获取港股 {code} 日K 数据 "
                    f"（源 {idx}/{total_sources}）"
                )
                df = fetch_func()

                if df is not None and len(df) > 0:
                    summary_parts.append(f"{source_name}（成功！{len(df)}条）")
                    result_df = df
                    break

                logger.debug(
                    f"[DAILY_K] ✗ {source_name} 返回空的港股 {code} 日K 数据"
                )
                summary_parts.append(f"{source_name}（失败：空数据）")

            except Exception as e:
                last_error = e
                error_msg = str(e)
                if 'ConnectTimeout' in error_msg or 'Connection' in error_msg:
                    err_short = error_msg[:80]
                    logger.debug(
                        f"[DAILY_K] ✗ {source_name} 获取港股 {code} 日K 数据连接失败: "
                        f"{err_short}"
                    )
                    summary_parts.append(f"{source_name}（失败：网络异常）")
                else:
                    err_short = error_msg[:80]
                    logger.debug(
                        f"[DAILY_K] ✗ {source_name} 获取港股 {code} 日K 数据失败: "
                        f"{err_short}"
                    )
                    summary_parts.append(f"{source_name}（失败：其它异常）")

        # 汇总打印
        if summary_parts:
            summary_line = f"日线K线获取[{original_code}]：" + " -> ".join(summary_parts)
        else:
            summary_line = f"日线K线获取[{original_code}]：未尝试任何数据源"

        logger.info(summary_line)

        if result_df is not None:
            return result_df

        # 所有数据源都失败
        msg_final = f"[DAILY_K] 所有数据源均失败，无法获取港股 {code} 日K 数据"
        logger.error(msg_final)
        return None

    def _fetch_hk_eastmoney(self, code: str, start_date: str, end_date: str,
                            adjust: str) -> pd.DataFrame:
        """从东方财富获取港股数据"""
        try:
            adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
            df = self.ak.stock_hk_hist(
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
            df['source'] = '东方财富'  # 添加数据源标记

            return df.reset_index(drop=True)

        except Exception as e:
            logger.debug(f"东方财富获取失败: {e}")
            return None

    def _fetch_hk_sina(self, code: str, start_date: str, end_date: str,
                       adjust: str) -> pd.DataFrame:
        """从新浪财经获取港股日 K 数据"""
        try:
            adjust_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
            df = self.ak.stock_hk_daily(
                symbol=code,
                adjust=adjust_map.get(adjust, 'qfq')
            )

            if df is None or len(df) == 0:
                logger.debug(f"[DAILY_K] ✗ 新浪财经 返回空的港股 {code} 日K 数据")
                return None

            # 新浪返回的列名已经是英文标准格式
            # 列名: ['date', 'open', 'high', 'low', 'close', 'volume']

            # 确保有所需的列
            required_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None

            # 过滤日期范围（新浪返回全部历史数据）
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

            if len(df) == 0:
                logger.debug(
                    f"[DAILY_K] ✗ 新浪财经 返回空的港股 {code} 日K 数据（筛选日期后为空）"
                )
                return None

            # 选择需要的列
            df = df[required_cols].copy()
            df['code'] = code
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['source'] = '新浪财经'  # 添加数据源标记

            return df.reset_index(drop=True)

        except Exception as e:
            logger.debug(f"新浪财经获取失败: {e}")
            return None

    def _fetch_hk_yfinance(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从 Yahoo Finance 获取港股数据"""
        try:
            # 统一转换为 Yahoo 港股代码格式
            # 例如:
            #   本地: 00700 / 00700.HK -> Yahoo: 0700.HK
            #   本地: 01810 / 01810.HK -> Yahoo: 1810.HK
            yf_code = self._to_yahoo_hk_symbol(code)
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

            df = df.reset_index(drop=True)
            return df

        except ImportError:
            logger.debug("yfinance 未安装，跳过")
            return None
        except Exception as e:
            logger.debug(f"Yahoo Finance 获取失败: {e}")
            return None


    def get_realtime_data(self, code: str) -> Optional[dict]:
        """
        获取实时行情数据（多数据源）

        当前支持：
        - A股: 新浪财经实时 + 东方财富实时 + Baostock 实时（按优先级依次尝试）
        - 港股: 新浪财经实时 + 东方财富实时 + Yahoo Finance 实时（按优先级依次尝试）
        - 美股: Yahoo Finance 实时（若可用）

        Args:
            code: 股票代码（A股/港股/美股）

        Returns:
            包含实时价格、涨跌幅等信息的字典，或 None 表示获取失败
        """
        try:
            market = self._detect_market(code)

            if market == 'HK':
                return self._get_realtime_hk_multi(code)
            elif market == 'CN-A':
                return self._get_realtime_cn_multi(code)
            elif market == 'US':
                return self._get_realtime_us(code)
            else:
                # 兜底：按 A 股处理
                return self._get_realtime_cn_multi(code)

        except Exception as e:
            logger.error(f"获取实时数据失败 {code}: {e}")
            return None

    @staticmethod
    def _safe_float(value):
        """将值安全转换为 float，失败时返回 None"""
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)
        except Exception:
            return None

    def _build_realtime_dict(
        self,
        code: str,
        name: Optional[str],
        price,
        open_price=None,
        high=None,
        low=None,
        last_close=None,
        volume=None,
        amount=None,
        pct_change=None,
        change=None,
    ) -> Optional[dict]:
        """
        统一构建实时行情字典，尽量补全涨跌额/涨跌幅
        """
        price_f = self._safe_float(price)
        if price_f is None:
            return None

        open_f = self._safe_float(open_price)
        high_f = self._safe_float(high)
        low_f = self._safe_float(low)
        last_close_f = self._safe_float(last_close)
        volume_f = self._safe_float(volume)
        amount_f = self._safe_float(amount)

        change_f = self._safe_float(change)
        pct_change_f = self._safe_float(pct_change)

        # 如果缺失，尝试根据 price 和 last_close 计算
        if change_f is None and last_close_f is not None:
            change_f = price_f - last_close_f

        if pct_change_f is None and change_f is not None and last_close_f not in (None, 0):
            pct_change_f = change_f / last_close_f * 100

        return {
            'code': code,
            'name': name or code,
            'price': price_f,
            'change': change_f,
            'pct_change': pct_change_f,
            'volume': volume_f,
            'amount': amount_f,
            'high': high_f,
            'low': low_f,
            'open': open_f,
            'last_close': last_close_f,
        }

    def _normalize_cn_code(self, code: str) -> str:
        """
        规范化 A 股代码为 6 位数字（不带前缀）
        """
        clean = code
        if clean.startswith(('sh', 'sz', 'SH', 'SZ')):
            clean = clean[2:]
        if clean.endswith(('.SH', '.SZ')):
            clean = clean[:-3]
        if clean.isdigit() and len(clean) < 6:
            clean = clean.zfill(6)
        return clean

    def _normalize_hk_code(self, code: str) -> str:
        """
        规范化港股代码为 5 位数字（不带 .HK 后缀）
        """
        clean = code
        if clean.endswith('.HK'):
            clean = clean[:-3]
        if clean.isdigit() and len(clean) < 5:
            clean = clean.zfill(5)
        return clean

    def _to_yahoo_hk_symbol(self, code: str) -> str:
        """
        将本地港股代码转换为 Yahoo Finance 使用的代码格式（4位数字 + '.HK'）

        支持输入形式:
        - '09988', '9988'
        - '09988.HK', '9988.HK'
        统一返回: 例如 '9988.HK'
        """
        base = code
        # 去掉 .HK 后缀（如果有）
        if base.endswith('.HK'):
            base = base[:-3]

        # 纯数字：去掉多余前导 0，再按至少 4 位补零
        if base.isdigit():
            try:
                num = int(base)
                return f"{num:04d}.HK"
            except ValueError:
                # 理论上不会触发，兜底返回原始格式
                return f"{base}.HK"

        # 其它情况：尽量补上 .HK 后缀
        if not code.endswith('.HK'):
            return f"{code}.HK"
        return code

    def _get_realtime_cn_multi(self, code: str) -> Optional[dict]:
        """
        A股实时行情（多数据源）：
        1. 新浪财经 1 分钟分时（ak.stock_zh_a_minute）
        2. 东方财富 1 分钟分时（ak.stock_zh_a_hist_min_em）
        3. 腾讯财经 1 分钟分时（web.ifzq.gtimg.cn）
        """
        raw_code = code
        clean_code = self._normalize_cn_code(code)

        sources = [
            ('新浪财经 A股 分时',   lambda: self._get_realtime_cn_from_sina(clean_code, raw_code)),
            ('东方财富 A股 分时',   lambda: self._get_realtime_cn_from_em(clean_code, raw_code)),
            ('腾讯财经 A股 分时',   lambda: self._get_realtime_cn_from_tx_minute(clean_code, raw_code)),
        ]

        # 随机打乱顺序：相当于随机选择一个起点，如果失败再依次尝试其他源
        random.shuffle(sources)

        # 调试日志：标记已进入多源实时逻辑，以及当前随机顺序（仅写入日志，不在控制台展开）
        try:
            order_str = " -> ".join(name for name, _ in sources)
        except Exception:
            order_str = " / ".join(name for name, _ in sources)
        logger.debug(
            f"[REALTIME_CN_MULTI] {clean_code} 分钟级实时K 源顺序: {order_str}"
        )

        last_error = None
        total_sources = len(sources)
        summary_parts = []
        result_data = None

        for idx, (source_name, fetch_func) in enumerate(sources, start=1):
            try:
                logger.debug(
                    f"[REALTIME_CN_MULTI] 尝试源 {idx}/{total_sources}: "
                    f"{source_name} 获取 A股 {clean_code} 分钟级实时K 数据"
                )
                data = fetch_func()
                if data and data.get('price') is not None:
                    logger.info(
                        f"[REALTIME_CN_MULTI] ✓ 成功从 {source_name} 获取 A股 "
                        f"{clean_code} 分钟级实时K 数据（源 {idx}/{total_sources}）"
                    )
                    summary_parts.append(f"{source_name}（成功！）")
                    result_data = data
                    break
                else:
                    logger.debug(
                        f"[REALTIME_CN_MULTI] ✗ {source_name} 返回空的 A股 {clean_code} "
                        f"分钟级实时K 数据或无价格（源 {idx}/{total_sources}）"
                    )
                    summary_parts.append(f"{source_name}（失败：空数据）")
            except Exception as e:
                last_error = e
                error_msg = str(e)
                err_short = error_msg[:80]
                logger.debug(
                    f"[REALTIME_CN_MULTI] ✗ {source_name} 获取 A股 {clean_code} "
                    f"分钟级实时K 数据失败（源 {idx}/{total_sources}）: {err_short}"
                )
                if 'Connection' in error_msg or 'ConnectTimeout' in error_msg:
                    summary_parts.append(f"{source_name}（失败：网络异常）")
                else:
                    summary_parts.append(f"{source_name}（失败：其它异常）")

        # 汇总打印
        display_code = raw_code
        if summary_parts:
            summary_line = f"分钟K线获取[{display_code}]：" + " -> ".join(summary_parts)
        else:
            summary_line = f"分钟K线获取[{display_code}]：未尝试任何数据源"

        logger.info(summary_line)

        if result_data is not None:
            return result_data

        # 所有实时数据源均失败
        if last_error:
            logger.error(
                f"[REALTIME_CN_MULTI] 所有实时数据源均失败，无法获取 A股 {clean_code} "
                f"分钟级实时K 数据: {last_error}"
            )
        else:
            logger.error(
                f"[REALTIME_CN_MULTI] 所有实时数据源均失败，无法获取 A股 {clean_code} "
                f"分钟级实时K 数据"
            )
        return None

    def _get_realtime_hk_multi(self, code: str) -> Optional[dict]:
        """
        港股实时行情（多数据源）：
        1. 东方财富 1 分钟分时（ak.stock_hk_hist_min_em）
        2. 腾讯财经 1 分钟分时（web.ifzq.gtimg.cn）
        3. Yahoo Finance 1 分钟分时（yfinance）
        """
        raw_code = code
        clean_code = self._normalize_hk_code(code)

        sources = [
            ('东方财富 港股 分时',   lambda: self._get_realtime_hk_from_em(clean_code, raw_code)),
            ('腾讯财经 港股 分时',   lambda: self._get_realtime_hk_from_tx_minute(clean_code, raw_code)),
            ('Yahoo 港股 实时',     lambda: self._get_realtime_hk_from_yahoo(clean_code, raw_code)),
        ]

        # 随机打乱顺序：相当于随机选择一个起点，如果失败再依次尝试其他源
        random.shuffle(sources)

        # 调试日志：标记已进入多源实时逻辑，以及当前随机顺序（仅写入日志）
        try:
            order_str = " -> ".join(name for name, _ in sources)
        except Exception:
            order_str = " / ".join(name for name, _ in sources)
        logger.debug(
            f"[REALTIME_HK_MULTI] {clean_code} 分钟级实时K 源顺序: {order_str}"
        )

        last_error = None
        total_sources = len(sources)
        summary_parts = []
        result_data = None

        for idx, (source_name, fetch_func) in enumerate(sources, start=1):
            try:
                logger.debug(
                    f"[REALTIME_HK_MULTI] 尝试源 {idx}/{total_sources}: "
                    f"{source_name} 获取港股 {clean_code} 分钟级实时K 数据"
                )
                data = fetch_func()
                if data and data.get('price') is not None:
                    logger.info(
                        f"[REALTIME_HK_MULTI] ✓ 成功从 {source_name} 获取港股 "
                        f"{clean_code} 分钟级实时K 数据（源 {idx}/{total_sources}）"
                    )
                    summary_parts.append(f"{source_name}（成功！）")
                    result_data = data
                    break
                else:
                    logger.debug(
                        f"[REALTIME_HK_MULTI] ✗ {source_name} 返回空的港股 {clean_code} "
                        f"分钟级实时K 数据或无价格（源 {idx}/{total_sources}）"
                    )
                    summary_parts.append(f"{source_name}（失败：空数据）")
            except Exception as e:
                last_error = e
                error_msg = str(e)
                err_short = error_msg[:80]
                logger.debug(
                    f"[REALTIME_HK_MULTI] ✗ {source_name} 获取港股 {clean_code} "
                    f"分钟级实时K 数据失败（源 {idx}/{total_sources}）: {err_short}"
                )
                if 'Connection' in error_msg or 'ConnectTimeout' in error_msg:
                    summary_parts.append(f"{source_name}（失败：网络异常）")
                else:
                    summary_parts.append(f"{source_name}（失败：其它异常）")

        # 汇总打印
        display_code = raw_code
        if summary_parts:
            summary_line = f"分钟K线获取[{display_code}]：" + " -> ".join(summary_parts)
        else:
            summary_line = f"分钟K线获取[{display_code}]：未尝试任何数据源"

        logger.info(summary_line)

        if result_data is not None:
            return result_data

        # 所有实时数据源均失败
        if last_error:
            logger.error(
                f"[REALTIME_HK_MULTI] 所有实时数据源均失败，无法获取港股 {clean_code} "
                f"分钟级实时K 数据: {last_error}"
            )
        else:
            logger.error(
                f"[REALTIME_HK_MULTI] 所有实时数据源均失败，无法获取港股 {clean_code} "
                f"分钟级实时K 数据"
            )
        return None

    def _get_realtime_us(self, code: str) -> Optional[dict]:
        """
        美股实时行情（简单版，基于 Yahoo Finance）
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance 未安装，无法获取美股实时行情")
            return None

        try:
            ticker = yf.Ticker(code)
            # 优先用 1 分钟级别数据
            df = ticker.history(period='1d', interval='1m')
            if df is None or len(df) == 0:
                df = ticker.history(period='1d', interval='1d')

            if df is None or len(df) == 0:
                return None

            last = df.iloc[-1]
            price = last.get('Close')
            open_price = last.get('Open')
            high = last.get('High')
            low = last.get('Low')
            volume = last.get('Volume')

            return self._build_realtime_dict(
                code=code,
                name=None,
                price=price,
                open_price=open_price,
                high=high,
                low=low,
                last_close=None,
                volume=volume,
                amount=None,
                pct_change=None,
                change=None,
            )
        except Exception as e:
            logger.debug(f"Yahoo Finance 美股实时行情获取失败: {e}")
            return None

    def _get_realtime_cn_from_sina(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从新浪财经获取 A 股实时行情

        实现说明：
        - 早期使用 ak.stock_zh_a_spot 获取全市场快照，在当前环境中已无法解析返回格式
        - 改为使用 ak.stock_zh_a_minute 获取 1 分钟级别分时数据
        - 取最新一条分钟数据，作为实时行情近似
        """
        # 需要 akshare 支持
        if not hasattr(self, 'ak'):
            return None

        # 将 6 位代码转换为带市场前缀的形式：shXXXXXX / szXXXXXX
        if clean_code.startswith('6'):
            symbol = f"sh{clean_code}"
        else:
            symbol = f"sz{clean_code}"

        # 1 分钟周期，不复权
        df = self.ak.stock_zh_a_minute(
            symbol=symbol,
            period='1',
            adjust=''
        )
        if df is None or len(df) == 0:
            return None

        # 必要列检查
        required_cols = ['day', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # 按时间排序，取最新一条
        df = df.sort_values('day')
        row = df.iloc[-1]

        return self._build_realtime_dict(
            code=display_code,
            name=None,  # 分时接口不提供名称，这里留空交给上层兜底
            price=row.get('close'),
            open_price=row.get('open'),
            high=row.get('high'),
            low=row.get('low'),
            last_close=None,
            volume=row.get('volume'),
            amount=None,
            pct_change=None,
            change=None,
        )

    def _get_realtime_cn_from_em(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从东方财富获取 A 股实时行情

        实现说明：
        - 早期使用 ak.stock_zh_a_spot_em 获取全市场快照，在当前环境中不稳定
        - 改为使用 ak.stock_zh_a_hist_min_em 获取 1 分钟级别分时数据
        - 取最新一条分钟数据，作为实时行情近似
        """
        if not hasattr(self, 'ak'):
            return None

        # 使用 1 分钟周期的分时数据（前复权）
        df = self.ak.stock_zh_a_hist_min_em(
            symbol=clean_code,
            period='1',
            adjust='qfq'
        )
        if df is None or len(df) == 0:
            return None

        # 必要列检查
        required_cols = ['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
        if not all(col in df.columns for col in required_cols):
            return None

        # 按时间排序，取最新一条
        df = df.sort_values('时间')
        row = df.iloc[-1]

        return self._build_realtime_dict(
            code=display_code,
            name=None,  # 分时接口不提供名称，这里留空交给上层兜底
            price=row.get('收盘'),
            open_price=row.get('开盘'),
            high=row.get('最高'),
            low=row.get('最低'),
            last_close=None,
            volume=row.get('成交量'),
            amount=row.get('成交额'),
            pct_change=None,
            change=None,
        )

    def _get_realtime_cn_from_tx_minute(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从腾讯财经获取 A 股分钟级实时行情

        使用接口:
        https://web.ifzq.gtimg.cn/appstock/app/minute/query?code=sh600885
        数据格式示例:
        {
            "data": {
                "sh600885": {
                    "data": {
                        "data": [
                            "0930 31.20 453 1413360.00",
                            "0931 31.17 2865 8919000.00",
                            ...
                        ]
                    }
                }
            }
        }
        每行内容含义: 时间 价格 累计成交量 累计成交额
        """
        # 生成带市场前缀的代码
        if clean_code.startswith('6'):
            symbol = f"sh{clean_code}"
        else:
            symbol = f"sz{clean_code}"

        url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={symbol}"

        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            j = resp.json()
        except Exception as e:
            logger.debug(f"腾讯财经 A股 分时请求失败 {symbol}: {e}")
            return None

        try:
            node = j.get('data', {}).get(symbol, {}).get('data', {})
            lines = node.get('data', [])
            if not lines:
                return None

            prices = []
            volumes = []
            amounts = []
            for line in lines:
                parts = str(line).strip().split()
                if len(parts) < 4:
                    continue
                _, p, v, a = parts[:4]
                try:
                    prices.append(float(p))
                    volumes.append(float(v))
                    amounts.append(float(a))
                except Exception:
                    continue

            if not prices:
                return None

            # 当日聚合K线: 使用分钟数据推导 open/high/low/close/volume/amount
            open_price = prices[0]
            high = max(prices)
            low = min(prices)
            close_price = prices[-1]
            volume = volumes[-1] if volumes else None
            amount = amounts[-1] if amounts else None

            return self._build_realtime_dict(
                code=display_code,
                name=None,
                price=close_price,
                open_price=open_price,
                high=high,
                low=low,
                last_close=None,
                volume=volume,
                amount=amount,
                pct_change=None,
                change=None,
            )
        except Exception as e:
            logger.debug(f"腾讯财经 A股 分时解析失败 {symbol}: {e}")
            return None

    def _get_realtime_hk_from_sina(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从新浪财经获取港股实时行情（ak.stock_hk_spot）
        """
        if not hasattr(self, 'ak'):
            return None

        df = self.ak.stock_hk_spot()
        if df is None or len(df) == 0:
            return None

        # 代码列可能是 '代码' 或 'symbol'
        code_col = None
        if '代码' in df.columns:
            code_col = '代码'
        elif 'symbol' in df.columns:
            code_col = 'symbol'
        else:
            return None

        stock_data = df[df[code_col].astype(str).str.zfill(5) == clean_code]
        if len(stock_data) == 0:
            return None

        row = stock_data.iloc[0]
        return self._build_realtime_dict(
            code=display_code,
            name=row.get('名称', row.get('name', display_code)),
            price=row.get('最新价'),
            open_price=row.get('今开', row.get('open')),
            high=row.get('最高', row.get('high')),
            low=row.get('最低', row.get('low')),
            last_close=row.get('昨收', row.get('preclose')),
            volume=row.get('成交量', row.get('volume')),
            amount=row.get('成交额', row.get('amount')),
            pct_change=row.get('涨跌幅'),
            change=row.get('涨跌额'),
        )

    def _get_realtime_hk_from_em(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从东方财富获取港股实时行情

        实现说明：
        - 早期使用 ak.stock_hk_spot_em，但在当前环境下不稳定
        - 改为使用 ak.stock_hk_hist_min_em 获取 1 分钟级别分时数据
        - 取最新一条分钟数据，作为实时行情近似
        """
        if not hasattr(self, 'ak'):
            return None

        # 使用 1 分钟周期的分时数据
        df = self.ak.stock_hk_hist_min_em(
            symbol=clean_code,
            period='1',
            adjust='qfq'
        )
        if df is None or len(df) == 0:
            return None

        # 必要列检查
        required_cols = ['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '最新价']
        if not all(col in df.columns for col in required_cols):
            return None

        # 按时间排序，取最新一条作为当前实时近似
        df = df.sort_values('时间')
        row = df.iloc[-1]

        return self._build_realtime_dict(
            code=display_code,
            name=None,  # 分时接口不提供名称，这里留空交给上层兜底
            price=row.get('最新价', row.get('收盘')),
            open_price=row.get('开盘'),
            high=row.get('最高', row.get('high')),
            low=row.get('最低', row.get('low')),
            last_close=None,  # 分时数据不直接包含昨收
            volume=row.get('成交量', row.get('volume')),
            amount=row.get('成交额', row.get('amount')),
            pct_change=None,
            change=None,
        )

    def _get_realtime_hk_from_tx_minute(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从腾讯财经获取港股分钟级实时行情

        使用接口:
        https://web.ifzq.gtimg.cn/appstock/app/minute/query?code=hk00700
        数据格式与 A 股类似:
        "0930 628.000 541115 340725435.900"
        对应: 时间 价格 累计成交量 累计成交额
        """
        symbol = f"hk{clean_code.zfill(5)}"
        url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={symbol}"

        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            j = resp.json()
        except Exception as e:
            logger.debug(f"腾讯财经 港股 分时请求失败 {symbol}: {e}")
            return None

        try:
            node = j.get('data', {}).get(symbol, {}).get('data', {})
            lines = node.get('data', [])
            if not lines:
                return None

            prices = []
            volumes = []
            amounts = []
            for line in lines:
                parts = str(line).strip().split()
                if len(parts) < 4:
                    continue
                _, p, v, a = parts[:4]
                try:
                    prices.append(float(p))
                    volumes.append(float(v))
                    amounts.append(float(a))
                except Exception:
                    continue

            if not prices:
                return None

            open_price = prices[0]
            high = max(prices)
            low = min(prices)
            close_price = prices[-1]
            volume = volumes[-1] if volumes else None
            amount = amounts[-1] if amounts else None

            return self._build_realtime_dict(
                code=display_code,
                name=None,
                price=close_price,
                open_price=open_price,
                high=high,
                low=low,
                last_close=None,
                volume=volume,
                amount=amount,
                pct_change=None,
                change=None,
            )
        except Exception as e:
            logger.debug(f"腾讯财经 港股 分时解析失败 {symbol}: {e}")
            return None

    def _get_realtime_hk_from_yahoo(self, clean_code: str, display_code: str) -> Optional[dict]:
        """
        从 Yahoo Finance 获取港股实时行情（1 分钟级别）
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance 未安装，跳过 Yahoo 港股实时行情")
            return None

        # 确定 yfinance 使用的代码格式
        # 注意：Yahoo 港股代码一般为 4 位数字，不带多余前导 0，例如：
        #  - 腾讯控股: 0700.HK （本地代码通常写作 00700）
        #  - 阿里巴巴: 9988.HK （本地代码通常写作 09988）
        # 使用统一的转换函数，避免 09988.HK 这种形式导致 404 或空数据
        yf_code = self._to_yahoo_hk_symbol(display_code or clean_code)

        try:
            ticker = yf.Ticker(yf_code)
            df = ticker.history(period='1d', interval='1m')
            if df is None or len(df) == 0:
                df = ticker.history(period='1d', interval='1d')

            if df is None or len(df) == 0:
                return None

            last = df.iloc[-1]
            price = last.get('Close')
            open_price = last.get('Open')
            high = last.get('High')
            low = last.get('Low')
            volume = last.get('Volume')

            return self._build_realtime_dict(
                code=display_code,
                name=None,
                price=price,
                open_price=open_price,
                high=high,
                low=low,
                last_close=None,
                volume=volume,
                amount=None,
                pct_change=None,
                change=None,
            )
        except Exception as e:
            logger.debug(f"Yahoo 港股实时行情获取失败: {e}")
            return None

    def get_stock_info(self, code: str) -> Optional[dict]:
        """
        获取股票基本信息（支持A股和港股，支持多数据源备用）

        Args:
            code: 股票代码

        Returns:
            包含股票名称、行业等信息的字典
        """
        if self.source != 'akshare':
            return None

        # 检测市场类型
        market = self._detect_market(code)

        if market == 'HK':
            # 港股逻辑：优先走网络，失败再回退到本地映射
            return self._get_hk_stock_info(code)
        else:
            # A股逻辑
            return self._get_cn_stock_info(code)

    def _get_hk_stock_info(self, code: str) -> Optional[dict]:
        """
        获取港股基本信息

        策略（简化版）：
        1. 只使用本地名称缓存（_get_hk_name_from_cache）
        2. 缓存中不存在时，直接使用代码本身作为名称
        3. 不再发起任何网络请求获取名称信息
        """
        # 格式化港股代码：去掉 .HK 后缀，并补齐为 5 位数字
        if code.endswith('.HK'):
            code = code[:-3]
        if code.isdigit():
            code = code.zfill(5)

        clean_code = code

        # 只从本地缓存获取，不再访问网络
        info = self._get_hk_name_from_cache(clean_code)
        if info is not None and len(info) > 0:
            return info

        # 缓存中没有，使用代码本身作为名称
        logger.debug(f"港股 {clean_code} 不在名称缓存中，使用代码作为名称")
        return {'股票简称': clean_code, '股票代码': clean_code}

    def _get_hk_info_from_em(self, code: str) -> Optional[dict]:
        """从东方财富获取港股信息"""
        try:
            # 使用东方财富港股实时行情接口
            df = self.ak.stock_hk_spot_em()

            if df is not None and len(df) > 0:
                # 代码列名可能是'代码'或'symbol'
                if '代码' in df.columns:
                    stock_data = df[df['代码'] == code]
                elif 'symbol' in df.columns:
                    stock_data = df[df['symbol'] == code]
                else:
                    return None

                if len(stock_data) > 0:
                    row = stock_data.iloc[0]
                    # 名称列名可能是'名称'或'name'
                    name = row.get('名称', row.get('name', code))
                    return {
                        '股票简称': name,
                        '股票代码': code,
                    }
            return None
        except Exception as e:
            logger.debug(f"东方财富港股信息接口失败: {e}")
            return None

    def _get_hk_name_from_cache(self, code: str) -> Optional[dict]:
        """从静态映射获取港股名称"""
        # 常见港股的静态映射表
        hk_stock_names = {
            '00700': '腾讯控股', '09988': '阿里巴巴-SW', '00941': '中国移动',
            '03690': '美团-W', '01810': '小米集团-W', '09618': '京东集团-SW',
            '09888': '百度集团-SW', '09999': '网易-S', '01024': '快手-W',
            '00388': '香港交易所', '01398': '工商银行', '03988': '中国银行',
            '00939': '建设银行', '01288': '农业银行', '02318': '中国平安',
            '00883': '中国海洋石油', '00386': '中国石油化工', '02628': '中国人寿',
            '01299': '友邦保险', '00175': '吉利汽车', '02333': '长城汽车',
            '01211': '比亚迪股份', '02015': '理想汽车-W', '09868': '小鹏汽车-W',
            '09866': '蔚来-SW', '01772': '赣锋锂业', '06862': '海底捞',
            '09961': '携程集团-S', '00981': '中芯国际', '00992': '联想集团',
            '02269': '药明生物', '00857': '中国石油股份', '01093': '石药集团',
            '02382': '舜宇光学科技', '02020': '安踏体育', '01177': '中国生物制药',
            '02367': '巨子生物', '06690': '海尔智家', '01347': '华虹半导体',
            '01585': '雅迪控股', '06682': '第四范式', '03692': '翰森制药',
        }

        name = hk_stock_names.get(code)
        if name:
            logger.info(f"✓ 从静态映射获取港股 {code} 名称: {name}")
            return {
                '股票简称': name,
                '股票代码': code,
            }

        # 不在映射表中，返回None让上层继续尝试其他数据源
        return None

    def _get_cn_stock_info(self, code: str) -> Optional[dict]:
        """
        获取A股基本信息（名称）

        简化策略：
        1. 只使用本地名称缓存（_get_stock_name_from_cache）
        2. 缓存中不存在时，直接使用代码本身作为名称
        3. 不再发起任何网络请求获取名称信息
        """
        # 格式化代码（去掉前缀）
        clean_code = code
        if code.startswith(('sh', 'sz')):
            clean_code = code[2:]

        # 只从本地缓存/静态映射获取名称
        return self._get_stock_name_from_cache(clean_code)

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
