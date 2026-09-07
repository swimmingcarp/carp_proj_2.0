#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Trading Advisor - 定时调度器
在固定时间自动运行股票分析并生成买卖提醒
"""

import sys
import yaml
import json
import logging
import argparse
import schedule
import time
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
PROJECT_ROOT = BASE_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher import DataFetcher
from src.new_strategy import RSITrendStrategy
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

        # 实时数据获取失败的股票列表（在每次 run_analysis 内重置）
        self.network_failed_stocks = []

        # 实盘持仓追踪：解决盘中价格和收盘价不一致的问题
        self.position_file = self.script_dir / 'data' / 'realtime_positions.json'
        self.realtime_positions = self._load_positions()

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """
        获取默认配置

        注意：这只是 fallback 配置，仅在配置文件加载失败时使用
        实际生产环境应使用 config/scheduler_config.yaml
        """
        return {
            'schedule': {
                'enabled': True,
                'run_times': [],  # 默认为空，强制用户在配置文件中设置
                'check_trading_day': True,
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

    # ========== 实盘持仓追踪功能 ==========
    # 解决盘中价格（如15:57）和收盘价不一致导致的信号漂移问题
    
    def _load_positions(self) -> Dict:
        """加载实盘持仓状态"""
        try:
            if self.position_file.exists():
                with open(self.position_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"加载持仓状态失败: {e}")
        return {}

    def _save_positions(self):
        """保存实盘持仓状态"""
        try:
            self.position_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.position_file, 'w', encoding='utf-8') as f:
                json.dump(self.realtime_positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存持仓状态失败: {e}")

    def _get_position(self, code: str) -> Optional[Dict]:
        """获取某只股票的实盘持仓状态"""
        return self.realtime_positions.get(code)

    def _set_position(self, code: str, buy_price: float, buy_date: str):
        """记录买入持仓"""
        self.realtime_positions[code] = {
            'buy_price': buy_price,
            'buy_date': buy_date,
            'status': 'holding'
        }
        self.logger.info(f"[持仓追踪] 记录买入: {code} @ ¥{buy_price:.2f} ({buy_date})")

    def _clear_position(self, code: str):
        """清除持仓记录"""
        if code in self.realtime_positions:
            del self.realtime_positions[code]
            self.logger.info(f"[持仓追踪] 清除持仓: {code}")

    def _apply_position_tracking(self, signals: List[Dict]) -> List[Dict]:
        """
        应用实盘持仓追踪逻辑，修正因盘中价格和收盘价不一致导致的信号问题
        
        规则：
        1. 今天发出买入信号，但实盘已持有 → 显示"持有"而非"买入"
        2. 实盘已持有，但策略显示"观望"（昨天收盘没买入信号）→ 提示"止损卖出"
        3. 策略显示"卖出"且实盘持有 → 正常卖出并清除持仓
        4. 策略显示"买入"且实盘未持有 → 正常买入并记录持仓
        """
        today = date.today().isoformat()
        adjusted_signals = []
        
        for signal in signals:
            code = signal.get('code', '')
            action = signal.get('action', '')
            price = signal.get('price', 0)
            position = self._get_position(code)
            
            original_action = action
            adjusted_reason = None
            
            if action == '买入':
                if position:
                    # 规则1: 已持仓，买入信号改为持有
                    signal['action'] = '持有'
                    signal['signal'] = 'HOLD_BUY'
                    adjusted_reason = f"实盘已于{position['buy_date']}买入，继续持有"
                    # 更新原因
                    original_reason = signal.get('reason', '')
                    signal['reason'] = f"[持仓追踪] {adjusted_reason}（原信号: 买入 - {original_reason}）"
                    self.logger.info(f"[持仓追踪] {code}: 买入→持有 ({adjusted_reason})")
                else:
                    # 规则4: 新买入，记录持仓
                    self._set_position(code, price, today)
                    
            elif action == '卖出':
                if position:
                    # 规则3: 正常卖出，清除持仓
                    self._clear_position(code)
                    
            elif action == '观望':
                if position:
                    # 规则2: 实盘持有但策略显示观望，说明盘中买入后收盘价不满足条件
                    # 检查是否是当天买入的（当天买入的不需要止损提示）
                    if position.get('buy_date') != today:
                        signal['action'] = '卖出'
                        signal['signal'] = 'SELL'
                        buy_price = position.get('buy_price', 0)
                        loss_pct = ((price - buy_price) / buy_price * 100) if buy_price > 0 else 0
                        adjusted_reason = f"实盘于{position['buy_date']}以¥{buy_price:.2f}买入，但策略信号已消失，建议止损"
                        signal['reason'] = f"[持仓追踪] {adjusted_reason}（当前价¥{price:.2f}，浮盈{loss_pct:+.2f}%）"
                        self.logger.info(f"[持仓追踪] {code}: 观望→卖出止损 ({adjusted_reason})")
                        self._clear_position(code)
                        
            elif action == '持有':
                # 策略本身显示持有，确保持仓记录存在
                if not position:
                    # 可能是之前没有追踪到的持仓，补记录
                    self._set_position(code, price, today)
                    self.logger.info(f"[持仓追踪] {code}: 补记录持仓 @ ¥{price:.2f}")
            
            adjusted_signals.append(signal)
        
        # 保存持仓状态
        self._save_positions()
        
        return adjusted_signals

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
            fetcher = DataFetcher(
                source=data_config.get('provider', app_config.DATA_SOURCE),
                # 微信推送场景：为了确保信号完全基于最新网络数据，这里强制关闭缓存
                # 即便网络失败也不回退到本地缓存，而是直接视为无信号
                cache_enabled=False,
                max_retries=app_config.MAX_RETRIES,
                retry_delay=app_config.RETRY_DELAY,
                is_backtest_mode=False,  # 实时模式
                default_adjust=data_config.get('adjust', app_config.DEFAULT_ADJUST),
            )

            # 检测市场类型
            market = fetcher._detect_market(stock_code)
            strategy = RSITrendStrategy(market=market, stock_code=stock_code)

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

            # 检查网络/实时数据状态（在交易时间内，避免用过期数据发出信号）
            net_status = None
            if validation_report and isinstance(validation_report, dict):
                net_status = validation_report.get('net_status')
            if net_status == 'REALTIME_FAILED':
                self.logger.warning(f"{stock_code} 实时分钟数据获取失败，在交易时间内跳过信号计算")
                self.network_failed_stocks.append(stock_code)
                return None

            # 检查数据验证（仅针对数据质量）
            if validation_report and validation_report.get('status') == 'FAILED':
                self.logger.warning(f"{stock_code} 数据验证失败")
                return None

            if df is None or len(df) == 0:
                self.logger.warning(f"{stock_code} 无数据")
                # 视为网络或接口获取失败，在推送中提示
                self.network_failed_stocks.append(stock_code)
                return None

            # 执行策略分析
            df_analyzed, _ = strategy.analyze(df)

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
                name = signal.get('name', code)
                action = signal.get('action', '')
                price = signal.get('price', 0)
                reason = signal.get('reason', '')
                k = signal.get('k', 0)
                macd = signal.get('macd', 0)

                # 显示格式：【代码】 名称（当名称与代码不同时）或 【代码】（相同时）
                if name and name != code:
                    display_title = f"【{code}】 {name}"
                else:
                    display_title = f"【{code}】"
                lines.append(f"  {display_title}")
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
                name = signal.get('name', code)
                action = signal.get('action', '')
                price = signal.get('price', 0)
                reason = signal.get('reason', '')
                k = signal.get('k', 0)
                macd = signal.get('macd', 0)

                # 显示格式：【代码】 名称（当名称与代码不同时）或 【代码】（相同时）
                if name and name != code:
                    display_title = f"【{code}】 {name}"
                else:
                    display_title = f"【{code}】"
                lines.append(f"  {display_title}")
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
                name = signal.get('name', code)
                price = signal.get('price', 0)
                # 显示格式：代码 名称（当名称与代码不同时）或 代码（相同时）
                if name and name != code:
                    display_name = f"{code} {name}"
                else:
                    display_name = code
                lines.append(f"  {display_name}  |  价格: ¥{price:.2f}")

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
                name = signal.get('name', code)
                price = signal.get('price', 0)
                # 显示格式：代码 名称（当名称与代码不同时）或 代码（相同时）
                if name and name != code:
                    display_name = f"{code} {name}"
                else:
                    display_name = code
                lines.append(f"  {display_name}  |  价格: ¥{price:.2f}")
        elif watch_signals:
            # 有其他信号时，简化显示观望股票
            lines.append(f"\n{'='*20}")
            lines.append(f"⚪ 观望 ({len(watch_signals)}只)")
            lines.append("=" * 20)
            for signal in watch_signals:
                code = signal.get('code', '')
                name = signal.get('name', code)
                price = signal.get('price', 0)
                # 显示格式：代码 名称（当名称与代码不同时）或 代码（相同时）
                if name and name != code:
                    display_name = f"{code} {name}"
                else:
                    display_name = code
                lines.append(f"  {display_name}  |  价格: ¥{price:.2f}")

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

    def run_analysis(self, market_filter: List[str] = None):
        """
        执行定时分析任务

        Args:
            market_filter: 市场过滤列表，如 ['CN-A'] 或 ['HK']，None表示所有市场
        """
        schedule_config = self.config.get('schedule', {})

        # 检查是否需要验证交易日
        if schedule_config.get('check_trading_day', True):
            if not MarketHours.is_trading_day():
                self.logger.info("今天不是交易日，跳过分析")
                print("📅 今天不是交易日，跳过分析")
                return

        self.logger.info("=" * 50)
        if market_filter:
            self.logger.info(f"开始定时分析任务（市场: {', '.join(market_filter)}）")
        else:
            self.logger.info("开始定时分析任务（所有市场）")

        # 每次分析前重置网络失败列表
        self.network_failed_stocks = []

        # 加载监控列表
        watch_list = self._load_watch_list()

        if not watch_list:
            self.logger.warning("监控列表为空")
            print("⚠️  监控列表为空，请在配置文件中添加股票代码")
            return

        # 根据市场过滤股票
        if market_filter:
            from src.data_fetcher import DataFetcher
            fetcher = DataFetcher(source='akshare', cache_enabled=True, default_adjust=app_config.DEFAULT_ADJUST)

            filtered_watch_list = []
            for code in watch_list:
                market = fetcher._detect_market(code)
                if market in market_filter:
                    filtered_watch_list.append(code)

            watch_list = filtered_watch_list

            if not watch_list:
                market_names = {'CN-A': 'A股', 'HK': '港股', 'US': '美股'}
                market_str = ', '.join([market_names.get(m, m) for m in market_filter])
                self.logger.info(f"当前时段没有{market_str}股票需要推送")
                print(f"ℹ️  当前时段没有{market_str}股票需要推送")
                return

        self.logger.info(f"监控股票数量: {len(watch_list)}")
        print(f"\n🔍 开始分析 {len(watch_list)} 只股票...")
        if market_filter:
            market_names = {'CN-A': 'A股', 'HK': '港股', 'US': '美股'}
            market_str = ', '.join([market_names.get(m, m) for m in market_filter])
            print(f"   市场类型: {market_str}")

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

        # 应用实盘持仓追踪逻辑（解决盘中价格和收盘价不一致问题）
        filtered_signals = self._apply_position_tracking(filtered_signals)

        # 生成提醒文本
        alert_text = self._format_alert(filtered_signals)

        # 如果存在网络/实时数据获取失败的股票，在推送中追加提示
        if self.network_failed_stocks:
            unique_codes = sorted(set(self.network_failed_stocks))
            alert_text += (
                "\n\n⚠️ 实时数据获取失败的股票（可能是网络或接口问题）:\n  "
                + ", ".join(unique_codes)
            )

        # 发送提醒
        self._send_notification(alert_text)

        self.logger.info(f"分析完成，共 {len(signals)} 个信号")
        self.logger.info("=" * 50)

    def start(self):
        """启动调度器"""
        schedule_config = self.config.get('schedule', {})

        if not schedule_config.get('enabled', True):
            print("⚠️  调度器已禁用")
            return

        run_times = schedule_config.get('run_times', [])

        # 兼容两种配置格式
        # 格式1: ['14:57', '15:57'] - 字符串列表（所有市场）
        # 格式2: [{'time': '14:57', 'markets': ['CN-A']}, ...] - 字典列表（分市场）

        if not run_times:
            print("⚠️  未配置运行时间")
            return

        # 注册定时任务
        task_count = 0
        for item in run_times:
            if isinstance(item, str):
                # 格式1: 简单字符串，所有市场
                run_time = item
                markets = None
                schedule.every().day.at(run_time).do(self.run_analysis, market_filter=markets)
                print(f"⏰ 已设置定时任务: 每天 {run_time} (所有市场)")
                task_count += 1

            elif isinstance(item, dict):
                # 格式2: 字典，包含时间和市场信息
                run_time = item.get('time')
                markets = item.get('markets', None)

                if not run_time:
                    self.logger.warning(f"任务配置缺少时间: {item}")
                    continue

                # 使用 lambda 捕获 markets 的值（避免闭包问题）
                schedule.every().day.at(run_time).do(
                    lambda m=markets: self.run_analysis(market_filter=m)
                )

                # 格式化显示
                if markets:
                    market_names = {'CN-A': 'A股', 'HK': '港股', 'US': '美股'}
                    market_str = ', '.join([market_names.get(m, m) for m in markets])
                    print(f"⏰ 已设置定时任务: 每天 {run_time} ({market_str})")
                else:
                    print(f"⏰ 已设置定时任务: 每天 {run_time} (所有市场)")
                task_count += 1

        if task_count == 0:
            print("⚠️  未能设置任何定时任务")
            return

        self.logger.info(f"调度器已启动，共 {task_count} 个定时任务")
        print(f"\n✓ 调度器已启动")
        print(f"📋 监控股票列表: {len(self._load_watch_list())} 只")
        print(f"📅 定时任务数: {task_count} 个")
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
