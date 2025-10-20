#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Trading Advisor - 定时调度器
在固定时间自动运行股票分析并生成买卖提醒
"""

import sys
import yaml
import logging
import argparse
import schedule
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_fetcher import DataFetcher
from src.strategy import MixedStrategy
from src.analyzer import SignalAnalyzer
from src.market_hours import MarketHours
from src.wechat_notifier import WeChatNotificationManager
import config as app_config


class TradingScheduler:
    """交易提醒调度器"""

    def __init__(self, config_path: str = 'config/scheduler_config.yaml'):
        """
        初始化调度器

        Args:
            config_path: 配置文件路径
        """
        # 获取脚本所在目录，确保路径正确
        self.script_dir = Path(__file__).parent

        # 如果配置路径是相对路径，转换为绝对路径
        if not Path(config_path).is_absolute():
            config_path = str(self.script_dir / config_path)

        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # 初始化微信通知管理器
        try:
            self.wechat_manager = WeChatNotificationManager(self.config)
            if self.wechat_manager.notifiers:
                self.logger.info(f"已初始化 {len(self.wechat_manager.notifiers)} 个微信通知器")
            else:
                self.wechat_manager = None
        except Exception as e:
            self.logger.warning(f"初始化微信通知失败: {e}")
            self.wechat_manager = None

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'schedule': {
                'enabled': True,
                'run_times': ['14:57'],  # 默认下午2:57
                'check_trading_day': True,  # 只在交易日运行
            },
            'stocks': {
                'watch_list': [],  # 监控列表
                'source': 'file',  # file/inline
                'file_path': 'config/watch_list.txt'
            },
            'notification': {
                'console': True,
                'log_file': True,
                'report_file': True,
                'report_path': 'reports/trading_alerts.txt'
            },
            'analysis': {
                'show_backtest': False,  # 实时提醒不需要回测
                'signal_filter': 'all'   # all/buy/sell/strong_buy
            },
            'data_source': {
                'provider': 'akshare',
                'cache_enabled': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/scheduler.log',
                'console': True
            }
        }

    def _setup_logging(self):
        """配置日志系统"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'logs/scheduler.log')

        # 如果是相对路径，转换为绝对路径
        if not Path(log_file).is_absolute():
            log_file = str(self.script_dir / log_file)

        # 创建日志目录
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # 配置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        handlers = [logging.FileHandler(log_file, encoding='utf-8')]

        if log_config.get('console', True):
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers
        )

    def _load_watch_list(self) -> List[str]:
        """加载监控股票列表"""
        stocks_config = self.config.get('stocks', {})
        source = stocks_config.get('source', 'file')

        if source == 'inline':
            # 从配置文件中直接读取
            return stocks_config.get('watch_list', [])
        else:
            # 从文件中读取
            file_path = stocks_config.get('file_path', 'config/watch_list.txt')

            # 如果是相对路径，转换为绝对路径
            if not Path(file_path).is_absolute():
                file_path = str(self.script_dir / file_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 读取所有非空行，去除注释和空格
                    codes = [
                        line.split('#')[0].strip()
                        for line in f.readlines()
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    return codes
            except FileNotFoundError:
                self.logger.error(f"监控列表文件不存在: {file_path}")
                print(f"❌ 错误: 监控列表文件不存在")
                print(f"   文件路径: {file_path}")
                print(f"   请创建该文件并添加股票代码，每行一个")
                return []
            except Exception as e:
                self.logger.error(f"读取监控列表文件失败: {e}")
                print(f"❌ 错误: 读取监控列表失败: {e}")
                return []

    def _analyze_stock(self, stock_code: str) -> Dict:
        """
        分析单只股票

        Args:
            stock_code: 股票代码

        Returns:
            信号数据字典
        """
        try:
            # 初始化组件
            data_config = self.config.get('data_source', {})
            strategy_config = self.config.get('strategy', {})

            fetcher = DataFetcher(
                source=data_config.get('provider', app_config.DATA_SOURCE),
                cache_enabled=data_config.get('cache_enabled', app_config.CACHE_ENABLED),
                max_retries=app_config.MAX_RETRIES,
                retry_delay=app_config.RETRY_DELAY,
                is_backtest_mode=False  # 实时模式
            )

            # 检测市场类型
            market = fetcher._detect_market(stock_code)
            strategy = MixedStrategy(config=strategy_config, market=market)

            # 获取数据（最近1年）
            result = fetcher.get_k_data(
                code=stock_code,
                start_date='2020-01-01'
            )

            # 解包结果
            if isinstance(result, tuple):
                df, validation_report = result
            else:
                df = result
                validation_report = None

            # 检查数据验证
            if validation_report and validation_report.get('status') == 'FAILED':
                self.logger.warning(f"{stock_code} 数据验证失败")
                return None

            if df is None or len(df) == 0:
                self.logger.warning(f"{stock_code} 无数据")
                return None

            # 执行策略分析
            result = strategy.analyze(df)

            # 解包结果
            if isinstance(result, tuple):
                df_analyzed, indicator_report = result
            else:
                df_analyzed = result
                indicator_report = None

            if df_analyzed is None:
                return None

            # 获取最新信号
            signal_data = strategy.get_latest_signal(df_analyzed)
            signal_data['code'] = stock_code

            # 获取股票基本信息
            stock_info = fetcher.get_stock_info(stock_code)
            if stock_info:
                signal_data['name'] = stock_info.get('股票简称', stock_code)

            # 映射信号字段：将 signal 映射到 action
            # signal 字段的值：BUY, SELL, HOLD(空仓观望), HOLD_BUY(持仓中)
            signal_map = {
                'BUY': '买入',
                'SELL': '卖出',
                'HOLD': '观望',      # 空仓状态，无明确买卖信号
                'HOLD_BUY': '持有',  # 已持仓，继续持有
                'NO_DATA': '无数据'
            }
            signal_data['action'] = signal_map.get(signal_data.get('signal', 'HOLD'), '观望')

            return signal_data

        except Exception as e:
            self.logger.error(f"分析 {stock_code} 失败: {e}")
            return None

    def _filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        根据配置过滤信号

        Args:
            signals: 信号列表

        Returns:
            过滤后的信号列表
        """
        analysis_config = self.config.get('analysis', {})
        signal_filter = analysis_config.get('signal_filter', 'all')

        if signal_filter == 'all':
            return signals
        elif signal_filter == 'buy':
            return [s for s in signals if s.get('action') == '买入']
        elif signal_filter == 'sell':
            return [s for s in signals if s.get('action') == '卖出']
        elif signal_filter == 'hold':
            return [s for s in signals if s.get('action') == '持有']
        elif signal_filter == 'action_needed':
            # 过滤出需要操作的信号（买入、卖出），排除观望和持有
            return [s for s in signals if s.get('action') in ['买入', '卖出']]
        else:
            return signals

    def _format_alert(self, signals: List[Dict]) -> str:
        """
        格式化交易提醒

        Args:
            signals: 信号列表

        Returns:
            格式化的提醒文本
        """
        if not signals:
            return "📊 无交易信号"

        analyzer = SignalAnalyzer()

        # 分类信号
        buy_signals = [s for s in signals if s.get('action') == '买入']
        sell_signals = [s for s in signals if s.get('action') == '卖出']
        hold_signals = [s for s in signals if s.get('action') == '持有']  # 已持仓
        watch_signals = [s for s in signals if s.get('action') == '观望']  # 空仓观望

        lines = []
        lines.append("=" * 70)
        lines.append(f"📊 股票交易提醒 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # 显示统计信息
        lines.append(f"\n📈 监控股票总数: {len(signals)} 只")
        lines.append(f"   🟢 买入: {len(buy_signals)} 只  |  🔴 卖出: {len(sell_signals)} 只  |  🔵 持有: {len(hold_signals)} 只  |  ⚪ 观望: {len(watch_signals)} 只")

        if buy_signals:
            lines.append(f"\n{'='*70}")
            lines.append(f"🟢 买入信号 ({len(buy_signals)}只)")
            lines.append("=" * 70)
            for signal in buy_signals:
                code = signal.get('code', '')
                name = signal.get('name', '')
                action = signal.get('action', '')
                price = signal.get('price', 0)
                reason = signal.get('reason', '')
                k = signal.get('k', 0)
                macd = signal.get('macd', 0)

                lines.append(f"  【{code}】 {name}")
                lines.append(f"     操作: {action}  |  当前价: ¥{price:.2f}")
                if reason:
                    lines.append(f"     理由: {reason}")
                lines.append(f"     K值: {k:.2f}  |  MACD: {macd:.4f}")
                lines.append("")

        if sell_signals:
            lines.append(f"\n{'='*70}")
            lines.append(f"🔴 卖出信号 ({len(sell_signals)}只)")
            lines.append("=" * 70)
            for signal in sell_signals:
                code = signal.get('code', '')
                name = signal.get('name', '')
                action = signal.get('action', '')
                price = signal.get('price', 0)
                reason = signal.get('reason', '')
                k = signal.get('k', 0)
                macd = signal.get('macd', 0)

                lines.append(f"  【{code}】 {name}")
                lines.append(f"     操作: {action}  |  当前价: ¥{price:.2f}")
                if reason:
                    lines.append(f"     理由: {reason}")
                lines.append(f"     K值: {k:.2f}  |  MACD: {macd:.4f}")
                lines.append("")

        # 显示持有的股票（已持仓）
        if hold_signals:
            lines.append(f"\n{'='*20}")
            lines.append(f"🔵 持有中 ({len(hold_signals)}只)")
            lines.append("=" * 20)
            for signal in hold_signals:
                code = signal.get('code', '')
                name = signal.get('name', '')
                price = signal.get('price', 0)
                lines.append(f"  {code} {name}  |  价格: ¥{price:.2f}")

        # 显示观望的股票（空仓，无明确信号）
        if watch_signals and not buy_signals and not sell_signals and not hold_signals:
            # 如果全部都是观望信号，显示简要信息
            lines.append(f"\n{'='*20}")
            lines.append(f"⚪ 全部观望 ({len(watch_signals)}只)")
            lines.append("=" * 20)
            lines.append("当前所有监控股票均无明确买卖信号，建议空仓观望。")
            lines.append("")
            for signal in watch_signals:
                code = signal.get('code', '')
                name = signal.get('name', '')
                price = signal.get('price', 0)

                lines.append(f"  {code} {name}  |  价格: ¥{price:.2f}")
        elif watch_signals:
            # 有其他信号时，简化显示观望股票
            lines.append(f"\n{'='*20}")
            lines.append(f"⚪ 观望 ({len(watch_signals)}只)")
            lines.append("=" * 20)
            for signal in watch_signals:
                code = signal.get('code', '')
                name = signal.get('name', '')
                price = signal.get('price', 0)
                lines.append(f"  {code} {name}  |  价格: ¥{price:.2f}")

        lines.append("\n" + "=" * 20)

        return "\n".join(lines)

    def _send_notification(self, alert_text: str):
        """
        发送通知

        Args:
            alert_text: 提醒文本
        """
        notification_config = self.config.get('notification', {})

        # 控制台输出
        if notification_config.get('console', True):
            print(alert_text)

        # 日志记录
        if notification_config.get('log_file', True):
            self.logger.info("\n" + alert_text)

        # 保存到报告文件
        if notification_config.get('report_file', True):
            report_path = notification_config.get('report_path', 'reports/trading_alerts.txt')

            # 如果是相对路径，转换为绝对路径
            if not Path(report_path).is_absolute():
                report_path = str(self.script_dir / report_path)

            try:
                Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, 'a', encoding='utf-8') as f:
                    f.write(alert_text + "\n\n")
                self.logger.info(f"报告已保存到: {report_path}")
            except Exception as e:
                self.logger.error(f"保存报告失败: {e}")

        # 发送微信通知
        if self.wechat_manager and self.wechat_manager.notifiers:
            try:
                if self.wechat_manager.send_alert(alert_text):
                    self.logger.info("✓ 微信通知发送成功")
                    print("✓ 微信通知已发送")
                else:
                    self.logger.warning("⚠️  微信通知发送失败")
            except Exception as e:
                self.logger.error(f"发送微信通知异常: {e}")

    def run_analysis(self):
        """执行定时分析任务"""
        schedule_config = self.config.get('schedule', {})

        # 检查是否需要验证交易日
        if schedule_config.get('check_trading_day', True):
            if not MarketHours.is_trading_day():
                self.logger.info("今天不是交易日，跳过分析")
                print("📅 今天不是交易日，跳过分析")
                return

        self.logger.info("=" * 50)
        self.logger.info("开始定时分析任务")

        # 加载监控列表
        watch_list = self._load_watch_list()

        if not watch_list:
            self.logger.warning("监控列表为空")
            print("⚠️  监控列表为空，请在配置文件中添加股票代码")
            return

        self.logger.info(f"监控股票数量: {len(watch_list)}")
        print(f"\n🔍 开始分析 {len(watch_list)} 只股票...")

        # 分析所有股票
        signals = []
        for i, code in enumerate(watch_list, 1):
            print(f"  [{i}/{len(watch_list)}] {code}...", end=' ')
            signal = self._analyze_stock(code)
            if signal:
                signals.append(signal)
                print("✓")
            else:
                print("✗")

        # 过滤信号
        filtered_signals = self._filter_signals(signals)

        # 生成并发送提醒
        alert_text = self._format_alert(filtered_signals)
        self._send_notification(alert_text)

        self.logger.info(f"分析完成，共 {len(signals)} 个信号")
        self.logger.info("=" * 50)

    def start(self):
        """启动调度器"""
        schedule_config = self.config.get('schedule', {})

        if not schedule_config.get('enabled', True):
            print("⚠️  调度器已禁用")
            return

        run_times = schedule_config.get('run_times', ['14:57'])

        # 注册定时任务
        for run_time in run_times:
            schedule.every().day.at(run_time).do(self.run_analysis)
            print(f"⏰ 已设置定时任务: 每天 {run_time}")

        self.logger.info(f"调度器已启动，运行时间: {', '.join(run_times)}")
        print(f"\n✓ 调度器已启动")
        print(f"📋 监控股票列表: {len(self._load_watch_list())} 只")
        print(f"⏰ 运行时间: {', '.join(run_times)}")
        print(f"\n💡 提示: 按 Ctrl+C 停止调度器\n")

        # 循环执行
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 调度器已停止")
            self.logger.info("调度器已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stock Trading Scheduler - 股票交易定时提醒工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动调度器（使用默认配置）
  python scheduler.py

  # 使用自定义配置
  python scheduler.py -c config/my_scheduler.yaml

  # 立即运行一次（不等待定时）
  python scheduler.py --run-now

  # 立即运行并启动调度器
  python scheduler.py --run-now --start
        """
    )

    parser.add_argument('-c', '--config', type=str,
                       default='config/scheduler_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--run-now', action='store_true',
                       help='立即运行一次分析')
    parser.add_argument('--start', action='store_true',
                       help='启动定时调度器')

    args = parser.parse_args()

    # 创建调度器
    scheduler = TradingScheduler(config_path=args.config)

    # 立即运行
    if args.run_now:
        print("🚀 立即执行分析任务...")
        scheduler.run_analysis()

    # 启动调度器
    if args.start or not args.run_now:
        scheduler.start()


if __name__ == '__main__':
    main()
