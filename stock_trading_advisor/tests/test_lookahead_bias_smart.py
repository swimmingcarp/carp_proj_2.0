"""
未来函数检测测试 - 智能采样版本

核心思路：
1. 先全局回测，找出所有买卖点（信号点）
2. 针对每个信号点，逐一覆盖未来数据测试（100%覆盖关键点）
3. 在信号点之间的平静期，随机采样测试（应该无信号）

优势：
- 关键点100%覆盖（买卖点一个不漏）
- 平静期随机采样（确保平静期也正确）
- 测试数量大幅减少（只测试关键点+采样点）
- 性能提升10-20倍

示例：
    200天数据，只有10个买点+10个卖点
    传统方法：测试200天
    智能方法：测试20个关键点 + 10个随机采样 = 30天
    性能提升：200/30 = 6.7倍
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import random
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.new_strategy import RSITrendStrategy


class TestLookAheadBiasSmart(unittest.TestCase):
    """未来函数检测 - 智能采样版本（使用真实数据）"""

    def _extract_signal_points(self, result: pd.DataFrame,
                              signal_cols: List[str]) -> Dict[str, List[int]]:
        """
        提取所有信号点的索引

        Args:
            result: 全局回测结果
            signal_cols: 信号列名列表

        Returns:
            {signal_name: [索引列表]}
        """
        signal_points = {}

        for col in signal_cols:
            if col not in result.columns:
                continue

            # 找出该列中所有信号点（值为True或1的位置）
            indices = result[result[col] == 1].index.tolist()

            # 过滤掉前180天（策略需要的最少数据）
            indices = [i for i in indices if i >= 180]

            signal_points[col] = indices

        return signal_points

    def _get_quiet_periods(self, signal_points: Dict[str, List[int]],
                          total_length: int,
                          sample_count: int = None) -> List[int]:
        """
        获取平静期（无信号期间）的随机采样点

        Args:
            signal_points: 所有信号点
            total_length: 数据总长度
            sample_count: 采样数量（默认等于信号点总数）

        Returns:
            随机采样的索引列表
        """
        # 合并所有信号点
        all_signal_indices = set()
        for indices in signal_points.values():
            all_signal_indices.update(indices)

        # 如果未指定采样数量，默认等于信号点总数
        if sample_count is None:
            sample_count = len(all_signal_indices)

        # 找出平静期（无信号的时期）
        min_idx = 180
        quiet_periods = []

        # 按信号点排序，找出间隔
        sorted_signals = sorted(all_signal_indices)

        if not sorted_signals:
            # 如果没有信号点，所有天都是平静期
            quiet_periods = list(range(min_idx, total_length))
        else:
            # 第一个信号点之前
            if sorted_signals[0] > min_idx:
                quiet_periods.extend(range(min_idx, sorted_signals[0]))

            # 信号点之间的间隔
            for i in range(len(sorted_signals) - 1):
                start = sorted_signals[i] + 1
                end = sorted_signals[i + 1]
                if end - start > 1:  # 至少间隔2天
                    quiet_periods.extend(range(start, end))

            # 最后一个信号点之后
            if sorted_signals[-1] < total_length - 1:
                quiet_periods.extend(range(sorted_signals[-1] + 1, total_length))

        # 随机采样：采样数量等于信号点数量，但不超过平静期总数
        sample_size = min(sample_count, len(quiet_periods))
        sample_size = max(5, sample_size)  # 至少采样5个点

        if sample_size > 0:
            return random.sample(quiet_periods, sample_size)
        else:
            return []

    def _test_stock(self, stock_code: str, market: str = 'CN'):
        """
        通用的股票测试方法

        Args:
            stock_code: 股票代码
            market: 市场 ('CN' 或 'HK')
        """
        print("\n" + "="*80)
        print(f"真实数据智能采样测试 ({stock_code})")
        print("="*80)

        # 读取真实数据
        data_path = os.path.join(
            os.path.dirname(__file__),
            f'../data/cache/{stock_code}_qfq.csv'
        )

        if not os.path.exists(data_path):
            print(f"⚠️  未找到{stock_code}数据文件，跳过测试")
            return

        real_data = pd.read_csv(data_path)
        real_data['date'] = pd.to_datetime(real_data['date'])

        print(f"   数据量: {len(real_data)}天")

        # 步骤1：全局回测
        print("\n步骤1: 全局回测...")
        strategy_full = RSITrendStrategy(market=market, stock_code=stock_code)
        result_full, _ = strategy_full.analyze(real_data.copy())

        if result_full is None:
            self.fail("真实数据回测失败")

        # 用于提取信号点的列（只包含离散的买卖信号）
        signal_cols = ['entry_signal', 'exit_signal', 'w_bottom_signal',
                      'bullish_divergence_signal', 'sideways_entry']  # 新增：震荡入场信号

        # 用于比较的列（包括中间状态，用于检测未来函数）
        comparison_cols = signal_cols + ['mtf_bias', 'direction',
                                          'is_sideways', 'aroon_osc']  # 新增：震荡状态和Aroon指标

        # 提取信号点
        signal_points = self._extract_signal_points(result_full, signal_cols)

        # 提取震荡区间边界点（is_sideways 从 False→True 和 True→False 的转换点）
        if 'is_sideways' in result_full.columns:
            sideways_vals = result_full['is_sideways'].astype(int).values
            boundary_indices = []
            for i in range(181, len(sideways_vals)):
                if sideways_vals[i] != sideways_vals[i - 1]:
                    boundary_indices.append(i)
            signal_points['sideways_boundary'] = boundary_indices

        total_signal_points = sum(len(indices) for indices in signal_points.values())

        print(f"   找到信号点: {total_signal_points}个")
        for col, indices in signal_points.items():
            if indices:
                print(f"      {col}: {len(indices)}个")

        # 计算平静期总数
        all_signal_indices = set()
        for indices in signal_points.values():
            all_signal_indices.update(indices)
        total_quiet_days = len(real_data) - 180 - len(all_signal_indices)

        # 平静期采样（数量等于信号点数）
        quiet_samples = self._get_quiet_periods(
            signal_points=signal_points,
            total_length=len(real_data)
            # 默认采样数量 = 信号点总数
        )

        print(f"\n   数据统计:")
        print(f"      总数据量: {len(real_data)}天")
        print(f"      策略启动: 第180天")
        print(f"      有效测试范围: {len(real_data) - 180}天")
        print(f"      信号点总数: {len(all_signal_indices)}个")
        print(f"      平静期总数: {total_quiet_days}天")
        print(f"      平静期采样: {len(quiet_samples)}个 (等于信号点数量)")
        print(f"   总测试点数: {len(all_signal_indices) + len(quiet_samples)}个")

        # 构建测试列表
        test_indices = set()
        for indices in signal_points.values():
            test_indices.update(indices)
        test_indices.update(quiet_samples)
        test_indices = sorted(test_indices)

        total_tests = len(test_indices)
        print(f"\n步骤2: 开始测试 {total_tests} 个关键点...")

        # 预先提取全局信号值（包括所有要比较的列）
        full_signals = {}
        for col in comparison_cols:
            if col in result_full.columns:
                full_signals[col] = result_full[col].values

        # 逐一测试
        discrepancies = []
        signal_point_errors = []
        quiet_point_errors = []

        for idx, test_idx in enumerate(test_indices):
            if idx % 10 == 0:
                print(f"\r   进度: {idx}/{total_tests}", end='', flush=True)

            # 截取数据
            partial_data = real_data.iloc[:test_idx + 1].copy()
            strategy = RSITrendStrategy(market=market, stock_code=stock_code)
            result_partial, _ = strategy.analyze(partial_data)

            if result_partial is None:
                continue

            # 对比所有列（包括中间状态）
            differences = {}
            for col in comparison_cols:
                if col not in result_partial.columns:
                    continue

                partial_signal = result_partial[col].iloc[-1]
                full_signal = full_signals[col][test_idx]

                if partial_signal != full_signal:
                    differences[col] = {
                        'partial': partial_signal,
                        'full': full_signal
                    }

            if differences:
                result = {
                    'index': test_idx,
                    'date': real_data['date'].iloc[test_idx],
                    'differences': differences
                }
                discrepancies.append(result)

                is_signal_point = any(test_idx in indices
                                     for indices in signal_points.values())
                if is_signal_point:
                    signal_point_errors.append(result)
                else:
                    quiet_point_errors.append(result)

        print(f"\r   完成: {total_tests}/{total_tests} 个点测试    ")

        # 输出结果
        print("\n" + "="*80)
        print("测试结果")
        print("="*80)

        if not discrepancies:
            print("✅ 真实数据未检测到未来函数")
            print(f"   - 信号点测试: {total_signal_points}个 ✅")
            print(f"   - 平静期采样: {len(quiet_samples)}个 ✅")
        else:
            print(f"⚠️  发现 {len(discrepancies)} 处差异")

            if signal_point_errors:
                print(f"\n❌ 信号点差异: {len(signal_point_errors)}个")
                print("   【严重】这些关键买卖点的判断使用了未来数据！\n")

                for i, err in enumerate(signal_point_errors[:10], 1):
                    print(f"   [{i}] 日期: {err['date']} (索引: {err['index']})")

                    for col, diff in err['differences'].items():
                        partial_val = '有信号(1)' if diff['partial'] == 1 else '无信号(0)'
                        full_val = '有信号(1)' if diff['full'] == 1 else '无信号(0)'

                        print(f"       {col}:")
                        print(f"         逐日回测: {partial_val}")
                        print(f"         全局回测: {full_val}")
                        print(f"         ⚠️  说明: 只用到{err['date']}的数据时{partial_val}，")
                        print(f"             但使用全部数据（包括未来）时{full_val}")
                        print(f"             → 该信号可能用到了{err['date']}之后的数据！")
                    print()

                if len(signal_point_errors) > 10:
                    print(f"   ... 还有 {len(signal_point_errors) - 10} 个信号点差异未显示\n")

            if quiet_point_errors:
                print(f"\n⚠️  平静期差异: {len(quiet_point_errors)}个")
                print("   【提示】这些非信号点出现差异，可能是策略计算的累积效应\n")

                for i, err in enumerate(quiet_point_errors[:5], 1):
                    print(f"   [{i}] 日期: {err['date']} (索引: {err['index']})")
                    for col, diff in err['differences'].items():
                        print(f"       {col}: 逐日={diff['partial']} vs 全局={diff['full']}")

                if len(quiet_point_errors) > 5:
                    print(f"\n   ... 还有 {len(quiet_point_errors) - 5} 个平静期差异未显示")

            # 构建详细的错误信息
            error_details = []
            if signal_point_errors:
                error_details.append(f"信号点差异: {len(signal_point_errors)}个")
            if quiet_point_errors:
                error_details.append(f"平静期差异: {len(quiet_point_errors)}个")

            self.fail(f"真实数据检测到未来函数！{' | '.join(error_details)}")

    def test_02367(self):
        """测试港股02367"""
        self._test_stock('02367', market='HK')

    def test_300750(self):
        """测试A股300750"""
        self._test_stock('300750', market='CN')

    def test_300274(self):
        """测试A股300274"""
        self._test_stock('300274', market='CN')


def run_tests():
    """运行测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLookAheadBiasSmart)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("未来函数检测 - 智能采样版本")
    print("="*80)
    print("策略：测试所有信号点 + 随机采样平静期")
    print("="*80 + "\n")

    success = run_tests()
    sys.exit(0 if success else 1)
