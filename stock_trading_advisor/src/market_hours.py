"""
市场交易时间工具
用于判断当前是否在交易时间，以及获取最近交易日
"""

from datetime import datetime, time, timedelta
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MarketHours:
    """市场交易时间管理"""

    # A股交易时间 (北京时间 UTC+8)
    CN_MORNING_START = time(9, 30)
    CN_MORNING_END = time(11, 30)
    CN_AFTERNOON_START = time(13, 0)
    CN_AFTERNOON_END = time(15, 0)

    # 港股交易时间 (香港时间 UTC+8)
    HK_MORNING_START = time(9, 30)
    HK_MORNING_END = time(12, 0)
    HK_AFTERNOON_START = time(13, 0)
    HK_AFTERNOON_END = time(16, 0)

    @staticmethod
    def is_trading_time(market: str = 'CN-A') -> bool:
        """
        判断当前是否在交易时间内

        Args:
            market: 市场类型 ('CN-A'-A股, 'HK'-港股)

        Returns:
            是否在交易时间内
        """
        now = datetime.now()
        current_time = now.time()
        current_weekday = now.weekday()

        # 周末不交易
        if current_weekday >= 5:  # 5=周六, 6=周日
            return False

        if market == 'HK':
            # 港股交易时间
            morning_session = (MarketHours.HK_MORNING_START <= current_time <= MarketHours.HK_MORNING_END)
            afternoon_session = (MarketHours.HK_AFTERNOON_START <= current_time <= MarketHours.HK_AFTERNOON_END)
            return morning_session or afternoon_session
        else:
            # A股交易时间
            morning_session = (MarketHours.CN_MORNING_START <= current_time <= MarketHours.CN_MORNING_END)
            afternoon_session = (MarketHours.CN_AFTERNOON_START <= current_time <= MarketHours.CN_AFTERNOON_END)
            return morning_session or afternoon_session

    @staticmethod
    def is_trading_day(date: datetime = None, market: str = 'CN-A') -> bool:
        """
        判断是否为交易日（简化版，只判断是否为工作日）

        Args:
            date: 日期，默认为今天
            market: 市场类型

        Returns:
            是否为交易日
        """
        if date is None:
            date = datetime.now()

        # 简化判断：周一到周五为交易日
        # 注意：实际应该考虑节假日，但这需要节假日数据库
        return date.weekday() < 5

    @staticmethod
    def get_latest_trading_date(market: str = 'CN-A') -> str:
        """
        获取最近的交易日期

        Args:
            market: 市场类型

        Returns:
            最近交易日期 (YYYY-MM-DD)
        """
        now = datetime.now()
        current_weekday = now.weekday()
        current_time = now.time()

        # 判断是否已收盘
        if market == 'HK':
            market_closed = current_time > MarketHours.HK_AFTERNOON_END
        else:
            market_closed = current_time > MarketHours.CN_AFTERNOON_END

        # 如果是周一到周五，且已收盘，返回今天
        if current_weekday < 5 and market_closed:
            return now.strftime('%Y-%m-%d')

        # 如果是周一到周五，且未收盘，返回昨天（如果昨天是工作日）
        if current_weekday < 5 and not market_closed:
            if current_weekday == 0:  # 周一
                # 返回上周五
                last_trading_day = now - timedelta(days=3)
            else:
                # 返回昨天
                last_trading_day = now - timedelta(days=1)
            return last_trading_day.strftime('%Y-%m-%d')

        # 如果是周六
        if current_weekday == 5:
            # 返回周五
            last_trading_day = now - timedelta(days=1)
            return last_trading_day.strftime('%Y-%m-%d')

        # 如果是周日
        if current_weekday == 6:
            # 返回周五
            last_trading_day = now - timedelta(days=2)
            return last_trading_day.strftime('%Y-%m-%d')

        return now.strftime('%Y-%m-%d')

    @staticmethod
    def get_market_status(market: str = 'CN-A') -> Tuple[str, str]:
        """
        获取市场状态

        Args:
            market: 市场类型

        Returns:
            (状态描述, 最近交易日)
        """
        is_trading = MarketHours.is_trading_time(market)
        latest_date = MarketHours.get_latest_trading_date(market)

        if is_trading:
            status = "交易中"
        else:
            now = datetime.now()
            if now.weekday() >= 5:
                status = "周末休市"
            else:
                current_time = now.time()
                if market == 'HK':
                    if current_time < MarketHours.HK_MORNING_START:
                        status = "盘前"
                    elif MarketHours.HK_MORNING_END < current_time < MarketHours.HK_AFTERNOON_START:
                        status = "午间休市"
                    else:
                        status = "盘后"
                else:
                    if current_time < MarketHours.CN_MORNING_START:
                        status = "盘前"
                    elif MarketHours.CN_MORNING_END < current_time < MarketHours.CN_AFTERNOON_START:
                        status = "午间休市"
                    else:
                        status = "盘后"

        return status, latest_date


if __name__ == '__main__':
    # 测试
    print("当前时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()

    for market in ['CN-A', 'HK']:
        print(f"=== {market} 市场 ===")
        print(f"是否交易时间: {MarketHours.is_trading_time(market)}")
        print(f"最近交易日: {MarketHours.get_latest_trading_date(market)}")
        status, date = MarketHours.get_market_status(market)
        print(f"市场状态: {status}")
        print()
