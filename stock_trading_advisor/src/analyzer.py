"""
买卖信号分析器
提供友好的信号展示和建议
"""

import pandas as pd
from typing import Dict, List
from colorama import Fore, Back, Style, init
import logging

# 初始化 colorama（用于彩色输出）
init(autoreset=True)

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """交易信号分析器"""

    def __init__(self):
        """初始化分析器"""
        pass

    def format_signal(self, signal_data: Dict, stock_info: Dict = None) -> str:
        """
        格式化信号输出

        Args:
            signal_data: 信号数据字典
            stock_info: 股票基本信息

        Returns:
            格式化后的字符串
        """
        output = []
        output.append("\n" + "=" * 60)

        # 股票信息
        if stock_info:
            output.append(f"股票代码: {stock_info.get('code', 'N/A')}")
            output.append(f"股票名称: {stock_info.get('name', 'N/A')}")
        else:
            output.append(f"分析日期: {signal_data.get('date', 'N/A')}")

        output.append(f"当前价格: {signal_data.get('price', 0):.2f}")
        output.append("=" * 60)

        # 信号类型
        signal = signal_data.get('signal', 'HOLD')
        strength = signal_data.get('strength', 0)

        if signal == 'BUY':
            color = Fore.GREEN
            signal_text = "【买入信号】"
            emoji = "📈"
        elif signal == 'SELL':
            color = Fore.RED
            signal_text = "【卖出信号】"
            emoji = "📉"
        elif signal == 'HOLD_BUY':
            color = Fore.YELLOW
            signal_text = "【持有信号】"
            emoji = "✋"
        else:
            color = Fore.WHITE
            signal_text = "【无信号】"
            emoji = "➖"

        output.append(color + f"\n{emoji} {signal_text} {self._get_strength_bar(strength)}")
        output.append(color + f"信号强度: {strength}/5")
        output.append(Style.RESET_ALL)

        # 信号原因
        reason = signal_data.get('reason', '无')
        output.append(f"\n理由: {reason}")

        # 技术指标
        output.append(f"\n--- 技术指标 ---")
        output.append(f"KDJ - K: {signal_data.get('k', 0):.2f}, D: {signal_data.get('d', 0):.2f}")
        output.append(f"MACD: {signal_data.get('macd', 0):.4f}, DIFF: {signal_data.get('diff', 0):.4f}")
        output.append(f"MA16: {signal_data.get('ma_16', 0):.2f}")
        output.append(f"MA45: {signal_data.get('ma_45', 0):.2f}")

        output.append("=" * 60 + "\n")

        return "\n".join(output)

    def _get_strength_bar(self, strength: int) -> str:
        """生成信号强度条"""
        filled = "█" * strength
        empty = "░" * (5 - strength)
        return f"[{filled}{empty}]"

    def format_backtest_result(self, result: Dict) -> str:
        """
        格式化回测结果

        Args:
            result: 回测结果字典

        Returns:
            格式化后的字符串
        """
        if result is None:
            return "回测失败"

        output = []
        output.append("\n" + "=" * 60)
        output.append("📊 回测结果")
        output.append("=" * 60)

        # 资金指标
        initial_capital = result.get('initial_capital', 0)
        final_capital = result.get('final_capital', 0)
        total_return = result.get('total_return', 0)
        profit_amount = final_capital - initial_capital

        output.append(f"\n💰 资金变化:")
        output.append(f"  初始资金: ¥{initial_capital:,.2f}")
        output.append(f"  最终资金: ¥{final_capital:,.2f}")

        # 盈亏显示
        if profit_amount > 0:
            profit_color = Fore.GREEN
            profit_symbol = "+"
        else:
            profit_color = Fore.RED
            profit_symbol = ""

        output.append(
            f"  盈亏金额: "
            f"{profit_color}{profit_symbol}¥{profit_amount:,.2f}{Style.RESET_ALL}"
        )

        # 收益率显示
        if total_return > 0:
            return_color = Fore.GREEN
        else:
            return_color = Fore.RED

        output.append(
            return_color +
            f"  总收益率: {total_return:.2f}%" +
            Style.RESET_ALL
        )

        # 显示手续费信息（如果有）
        total_commission = result.get('total_commission', 0)
        gross_return = result.get('gross_return', 0)
        commission_rate = result.get('commission_rate', 0)

        if total_commission > 0:
            output.append(f"\n💸 交易成本:")
            output.append(f"  累计手续费: ¥{total_commission:,.2f}")
            output.append(f"  手续费率: {commission_rate:.2f}%")
            if gross_return > 0:
                output.append(
                    Fore.YELLOW +
                    f"  毛收益率: {gross_return:.2f}% (扣费前)" +
                    Style.RESET_ALL
                )


        # 风险指标
        output.append(f"\n📉 风险指标:")
        max_dd = result.get('max_drawdown', 0)
        dd_color = Fore.RED if max_dd < -10 else Fore.YELLOW
        output.append(
            dd_color +
            f"  最大回撤: {max_dd:.2f}%" +
            Style.RESET_ALL
        )
        output.append(f"  夏普比率: {result.get('sharpe_ratio', 0):.2f}")

        # 交易统计
        output.append(f"\n📈 交易统计:")
        output.append(f"  交易次数: {result.get('total_trades', 0)}")

        win_rate = result.get('win_rate', 0)
        win_rate_color = Fore.GREEN if win_rate >= 60 else (Fore.YELLOW if win_rate >= 40 else Fore.RED)
        output.append(
            f"  胜率: " +
            win_rate_color +
            f"{win_rate:.2f}%" +
            Style.RESET_ALL
        )
        output.append(f"  交易天数: {result.get('trading_days', 0)}")

        # 如果有资金曲线数据，显示关键节点
        if 'capital_curve' in result and result['capital_curve']:
            curve = result['capital_curve']
            output.append(f"\n📊 资金曲线关键节点:")

            # 最高点
            max_capital = max(curve, key=lambda x: x['capital'])
            output.append(
                f"  峰值: ¥{max_capital['capital']:,.2f} "
                f"({max_capital['date']}, "
                f"+{(max_capital['capital']/initial_capital - 1)*100:.2f}%)"
            )

            # 最低点
            min_capital = min(curve, key=lambda x: x['capital'])
            if min_capital['capital'] < initial_capital:
                output.append(
                    Fore.RED +
                    f"  谷底: ¥{min_capital['capital']:,.2f} "
                    f"({min_capital['date']}, "
                    f"{(min_capital['capital']/initial_capital - 1)*100:.2f}%)" +
                    Style.RESET_ALL
                )

        output.append("=" * 60 + "\n")

        return "\n".join(output)

    def generate_recommendation(self, signal_data: Dict, backtest_result: Dict = None) -> str:
        """
        生成投资建议

        Args:
            signal_data: 信号数据
            backtest_result: 回测结果

        Returns:
            投资建议文本
        """
        output = []
        output.append("\n" + "💡 投资建议")
        output.append("-" * 60)

        signal = signal_data.get('signal', 'HOLD')
        strength = signal_data.get('strength', 0)

        if signal == 'BUY':
            if strength >= 4:
                output.append("✅ 强烈推荐买入")
                output.append("  - 多个技术指标共振，信号强度高")
                output.append("  - 建议: 可考虑分批建仓")
            elif strength >= 2:
                output.append("⚠️ 谨慎买入")
                output.append("  - 有买入信号，但强度一般")
                output.append("  - 建议: 小仓位试探，观察后续走势")
            else:
                output.append("❓ 信号较弱")
                output.append("  - 建议: 继续观察，等待更强信号")

        elif signal == 'SELL':
            output.append("🚫 建议卖出")
            output.append("  - 出现卖出信号，注意风险控制")
            output.append("  - 建议: 及时止盈或止损")

        elif signal == 'HOLD_BUY':
            output.append("✋ 持有观望")
            output.append("  - 当前处于持仓状态")
            output.append("  - 建议: 注意止损位置")

        else:
            output.append("➖ 无明确信号")
            output.append("  - 建议: 继续观察")

        # 基于回测结果的建议
        if backtest_result:
            output.append("\n基于历史回测:")
            total_return = backtest_result.get('total_return', 0)
            max_dd = backtest_result.get('max_drawdown', 0)
            win_rate = backtest_result.get('win_rate', 0)

            if total_return > 20 and max_dd > -15 and win_rate > 50:
                output.append("  ✅ 该策略历史表现良好")
            elif total_return < 0 or max_dd < -30:
                output.append("  ⚠️ 该策略历史回撤较大，需谨慎")
            else:
                output.append("  ℹ️ 该策略历史表现一般")

            output.append("-" * 60)

        return "\n".join(output)

    def batch_analyze(self, stock_signals: List[Dict]) -> str:
        """
        批量分析多只股票

        Args:
            stock_signals: 股票信号列表

        Returns:
            批量分析结果
        """
        output = []
        output.append("\n" + "=" * 80)
        output.append("📋 批量股票分析")
        output.append("=" * 80)

        # 分类统计
        buy_signals = [s for s in stock_signals if s.get('signal') == 'BUY']
        sell_signals = [s for s in stock_signals if s.get('signal') == 'SELL']
        hold_signals = [s for s in stock_signals if s.get('signal') == 'HOLD_BUY']

        output.append(f"\n总分析股票数: {len(stock_signals)}")
        output.append(f"买入信号: {len(buy_signals)}")
        output.append(f"卖出信号: {len(sell_signals)}")
        output.append(f"持有信号: {len(hold_signals)}")

        # 买入信号详情
        if buy_signals:
            output.append(f"\n{Fore.GREEN}--- 买入机会 ---{Style.RESET_ALL}")
            # 按强度排序
            buy_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
            for i, signal in enumerate(buy_signals[:10], 1):  # 只显示前10个
                output.append(
                    f"{i}. {signal.get('code', 'N/A')} - "
                    f"强度: {signal.get('strength', 0)}/5 - "
                    f"价格: {signal.get('price', 0):.2f} - "
                    f"{signal.get('reason', '')}"
                )

        # 卖出信号详情
        if sell_signals:
            output.append(f"\n{Fore.RED}--- 卖出提醒 ---{Style.RESET_ALL}")
            for i, signal in enumerate(sell_signals[:10], 1):
                output.append(
                    f"{i}. {signal.get('code', 'N/A')} - "
                    f"价格: {signal.get('price', 0):.2f} - "
                    f"{signal.get('reason', '')}"
                )

        output.append("=" * 80 + "\n")

        return "\n".join(output)

    def format_trading_signals(self, signals_data: Dict, show_limit: int = 10, df: pd.DataFrame = None) -> str:
        """
        格式化历史买卖点信号

        Args:
            signals_data: 包含买卖点的字典
            show_limit: 显示的最大数量
            df: 包含position和capital列的数据框（用于计算准确的收益）

        Returns:
            格式化后的字符串
        """
        output = []
        output.append("\n" + "=" * 80)
        output.append("📍 历史买卖点")
        output.append("=" * 80)

        buy_points = signals_data.get('buy_points', [])
        sell_points = signals_data.get('sell_points', [])
        total_trades = signals_data.get('total_trades', 0)

        # 分离未平仓和已平仓的卖出点
        open_positions = [sp for sp in sell_points if sp.get('is_open', False)]
        closed_sells = [sp for sp in sell_points if not sp.get('is_open', False)]

        output.append(f"\n总交易次数: {total_trades}")
        output.append(f"已完成交易: {len(closed_sells)}")
        output.append(f"当前持仓: {len(open_positions)}")

        # 买入点详情
        if buy_points:
            output.append(f"\n{Fore.GREEN}━━━ 买入点 (最近 {min(show_limit, len(buy_points))} 次) ━━━{Style.RESET_ALL}")
            output.append(f"{'序号':<6} {'日期':<14} {'价格':<10} {'手续费':<16}   {'原因'}")
            output.append("-" * 80)

            # 显示最近的买入点
            for i, point in enumerate(buy_points[-show_limit:], 1):
                # 格式化数据，确保与标题对齐
                seq = f"{i:<6}"
                date_str = f"{point['date']:<14}"
                price_str = f"{point['price']:>10.2f}"
                commission_str = f"{Fore.YELLOW}{point.get('commission', 0):>16.2f}{Style.RESET_ALL}"
                reason = point['reason']

                output.append(
                    f"{seq}"
                    f"{date_str}"
                    f"{price_str}"
                    f"{commission_str}"
                    f"      {reason}"
                )

        # 卖出点详情（只显示已完成的交易）
        if closed_sells:
            output.append(f"\n{Fore.RED}━━━ 卖出点 (最近 {min(show_limit, len(closed_sells))} 次) ━━━{Style.RESET_ALL}")
            output.append(f"{'序号':<6} {'日期':<14} {'价格':<10} {'手续费':<16}   {'原因'}")
            output.append("-" * 80)

            # 显示最近的卖出点
            for i, point in enumerate(closed_sells[-show_limit:], 1):
                # 格式化数据，确保与标题对齐
                seq = f"{i:<6}"
                date_str = f"{point['date']:<14}"
                price_str = f"{point['price']:>10.2f}"
                commission_str = f"{Fore.YELLOW}{point.get('commission', 0):>16.2f}{Style.RESET_ALL}"
                reason = point['reason']

                output.append(
                    f"{seq}"
                    f"{date_str}"
                    f"{price_str}"
                    f"{commission_str}"
                    f"      {reason}"
                )

        # 当前持仓详情（单独显示）
        if open_positions:
            output.append(f"\n{Fore.YELLOW}━━━ 当前持仓状态 ━━━{Style.RESET_ALL}")
            output.append(f"{'序号':<6} {'日期':<12} {'价格':<10} {'K值':<8} {'MACD':<10} {'状态说明'}")
            output.append("-" * 80)

            for i, point in enumerate(open_positions, 1):
                k_color = Fore.CYAN if point['k'] < 45 else ''
                macd_color = Fore.GREEN if point['macd'] > 0 else Fore.RED

                seq = f"{i:<6}"
                date_str = f"{point['date']:<12}"
                price_str = f"{point['price']:<10.2f}"
                k_str = f"{point['k']:<8.1f}"
                macd_str = f"{point['macd']:<10.4f}"
                reason = point['reason']

                output.append(
                    f"{seq}"
                    f"{date_str}"
                    f"{price_str}"
                    f"{k_color}{k_str}{Style.RESET_ALL}"
                    f"{macd_color}{macd_str}{Style.RESET_ALL}"
                    f"{reason}"
                )

        # 计算交易对收益
        if buy_points and sell_points:
            output.append(f"\n{Fore.CYAN}━━━ 交易对收益分析 ━━━{Style.RESET_ALL}")
            output.append(f"{Fore.YELLOW}说明: 买入价和卖出价均为次日开盘价{Style.RESET_ALL}")
            output.append(f"{Fore.YELLOW}      剩余本金已扣除交易手续费和印花税{Style.RESET_ALL}")

            # 获取初始资金和trades数据
            initial_capital = signals_data.get('initial_capital', 10000.0)
            trades = signals_data.get('trades', [])  # 获取trades数据

            output.append(f"{'序号':<6} {'买入日期':<12} {'买入价':<12} {'卖出日期':>10} {'卖出价':>10} {'收益率':>7} {'剩余本金':>10}")
            output.append("-" * 100)

            # 使用trades中的实际资金
            profit_rates = []
            winning_trades = 0
            cumulative_commission = 0.0  # 累计手续费

            for i, (buy, sell) in enumerate(zip(buy_points, sell_points), 1):
                is_open = sell.get('is_open', False)

                # 格式化数据，确保与标题对齐
                seq = f"{i:<6}"
                # 将日期转换为字符串格式
                buy_date_str = str(buy['date']).split()[0] if hasattr(buy['date'], 'strftime') else str(buy['date'])
                buy_date = f"{buy_date_str:<12}"
                buy_price = f"{buy['price']:>12.2f}"

                if is_open:
                    # 未平仓：统一显示"持仓中"
                    # 风险警告和原因说明已在"当前持仓状态"部分详细展示
                    # 这里不需要重复显示"卖出信号"
                    status_text = "持仓中"

                    sell_date = f"{Fore.YELLOW}{status_text:<20}{Style.RESET_ALL}"
                    sell_price = f"{Fore.YELLOW}{'--':<12}{Style.RESET_ALL}"

                    # 计算浮盈（价格收益率）
                    profit_rate = (sell['price'] - buy['price']) / buy['price'] * 100

                    # 获取实际资金（如果有trades数据）
                    if i <= len(trades):
                        # 使用上一笔交易后的实际资金
                        prev_capital = trades[i-2]['capital'] if i > 1 else initial_capital
                        float_profit_capital = prev_capital * (1 + profit_rate / 100)
                    else:
                        # 回退到简单计算
                        float_profit_capital = initial_capital * (1 + profit_rate / 100)

                    # 持仓中的累计手续费和毛收益率（基于已完成的交易）
                    gross_return_rate = (float_profit_capital + cumulative_commission - initial_capital) / initial_capital * 100

                    # 根据收益率显示浮盈或浮亏
                    if profit_rate >= 0:
                        profit_str = f"{Fore.YELLOW}(浮盈){profit_rate:>7.2f}%{Style.RESET_ALL}"
                    else:
                        profit_str = f"{Fore.YELLOW}(浮亏){profit_rate:>7.2f}%{Style.RESET_ALL}"

                    commission_str = f"{Fore.YELLOW}¥{cumulative_commission:>11,.2f}{Style.RESET_ALL}"
                    gross_return_str = f"{Fore.YELLOW}{gross_return_rate:>9.2f}%{Style.RESET_ALL}"
                    capital_str = f"{Fore.YELLOW}¥{float_profit_capital:>13,.2f}{Style.RESET_ALL}"
                else:
                    # 已平仓：正常显示
                    # 将日期转换为字符串格式
                    sell_date_str = str(sell['date']).split()[0] if hasattr(sell['date'], 'strftime') else str(sell['date'])
                    sell_date = f"{sell_date_str:>20}"
                    sell_price = f"{sell['price']:>12.2f}"

                    profit_rate = (sell['price'] - buy['price']) / buy['price'] * 100
                    profit_rates.append(profit_rate)

                    # 使用trades中的实际资金（已扣除手续费）
                    if i <= len(trades):
                        current_capital = trades[i-1]['capital']
                        # 累加手续费
                        current_commission = trades[i-1].get('commission', 0)
                        cumulative_commission += current_commission
                    else:
                        # 回退到简单计算（不应该发生）
                        current_capital = initial_capital * (1 + profit_rate / 100)

                    # 计算毛收益率（不含手续费的收益率）
                    gross_return_rate = (current_capital + cumulative_commission - initial_capital) / initial_capital * 100

                    if profit_rate > 0:
                        winning_trades += 1

                    profit_color = Fore.GREEN if profit_rate > 0 else Fore.RED
                    capital_color = Fore.GREEN if current_capital > initial_capital else Fore.RED

                    profit_str = f"{profit_rate:>11.2f}%"
                    commission_str = f"¥{cumulative_commission:>11,.2f}"
                    gross_return_str = f"{gross_return_rate:>9.2f}%"
                    capital_str = f"¥{current_capital:>13,.2f}"
                    profit_str = f"{profit_color}{profit_str}{Style.RESET_ALL} "
                    capital_str = f"{capital_color}{capital_str}{Style.RESET_ALL}"

                output.append(
                    f"{seq}"
                    f"{buy_date} "
                    f"{buy_price} "
                    f"{sell_date} "
                    f"{sell_price} "
                    f"{profit_str} "
                    f"{capital_str}"
                )

            final_capital = current_capital

            # 计算统计数据（只计算已完成的交易）
            if profit_rates:
                avg_profit = sum(profit_rates) / len(profit_rates)
                win_rate = (winning_trades / len(profit_rates) * 100)
            else:
                avg_profit = 0
                win_rate = 0

            avg_color = Fore.GREEN if avg_profit > 0 else Fore.RED
            final_return = (final_capital - initial_capital) / initial_capital * 100
            final_color = Fore.GREEN if final_return > 0 else Fore.RED

            # 从signals_data中获取准确的毛收益率（通过无手续费回测计算）
            gross_return = signals_data.get('gross_return', 0)
            # 如果没有获取到，使用简单计算作为fallback
            if gross_return == 0:
                gross_capital = final_capital + cumulative_commission
                gross_return = (gross_capital - initial_capital) / initial_capital * 100

            output.append("-" * 100)
            output.append(f"初始资金: ¥{initial_capital:,.2f}")
            output.append(f"最终资金: {final_color}¥{final_capital:,.2f}{Style.RESET_ALL} (已完成交易)")
            if len(open_positions) > 0:
                output.append(f"{Fore.YELLOW}当前持仓浮盈未计入最终资金{Style.RESET_ALL}")
            output.append(f"累计收益率: {final_color}{final_return:+.2f}%{Style.RESET_ALL}")
            output.append(f"累计手续费: {Fore.YELLOW}¥{cumulative_commission:,.2f}{Style.RESET_ALL}")
            output.append(f"毛收益率: {Fore.GREEN}{gross_return:.2f}%{Style.RESET_ALL} (扣费前)")
            output.append(f"平均单次收益: {avg_color}{avg_profit:.2f}%{Style.RESET_ALL}")
            output.append(f"盈利交易占比: {win_rate:.1f}%")

        output.append("=" * 80 + "\n")

        return "\n".join(output)
