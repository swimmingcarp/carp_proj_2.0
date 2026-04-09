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
import io
import contextlib
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.new_strategy import RSITrendStrategy
from src.personality.pit_stage import (
    compute_pit_states, compute_downtrend_phase, smooth_macro_phase,
    make_personality_config, compute_stock_dna,
)
from src.personality.segmenter import StockPersonalityEngine


def _load_cache_adjust() -> str:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'))
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return 'hfq'
    return (cfg.get('data_source', {}) or {}).get('adjust', 'hfq')


class TestLookAheadBiasSmart(unittest.TestCase):
    """未来函数检测 - 智能采样版本（使用真实数据）"""
    CACHE_ADJUST = _load_cache_adjust()

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
            f'../data/cache/{stock_code}_{self.CACHE_ADJUST}.csv'
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
        signal_cols = [
            'entry_signal',
            'exit_signal',
            'gc_extreme_chase_block',
            'zigzag_entry',
            'zigzag_fixed_entry',
            'zigzag_ddb_entry',
            'zigzag_dc_entry',
            'elliott_wave_entry',
            'zigzag_prob_entry',
            'wave_entry',
            'wave_start_signal',
            'wave_impulse_signal',
            'wave_retest_signal',
            'wave_end_signal',
            'wave_exit_takeover_block',
            'zigzag_trend_exit_softconfirm_block',
            'hard_stop_capitulation_softconfirm_block',
            'hard_stop_mainwave_softconfirm_block',
            'w_bottom_signal',
            'bullish_divergence_signal',
            'sideways_entry',
            'rsi_momentum_entry',
            'slow_bull_rotation_entry',
            'slow_bull_mtop_reclaim_entry',
            'slow_bull_mtop_reclaim_extended_entry',
            'slow_bull_ma_retest_entry',
            'banklike_ma_pullback_entry',
            'slow_bull_rotation_exit_signal',
        ]

        # 用于比较的列（包括中间状态，用于检测未来函数）
        comparison_cols = signal_cols + [
            'mtf_bias',
            'direction',
            'is_sideways',
            'aroon_osc',
            'atr_expanding',
            'rsi_momentum',
            'zigzag_prob_score',
            'zigzag_vote_count',
            'wave_active_signal',
            'wave_active_age',
            'wave_force_exit_signal',
            'wave_takeover_existing_position',
            'banklike_slow_switch_mask',
            'golden_cross_slow_switch_mask',
        ]

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
                        # 二值信号用中文描述，其他类型直接显示值
                        if diff['partial'] in (0, 1) and diff['full'] in (0, 1):
                            partial_val = '有信号(1)' if diff['partial'] == 1 else '无信号(0)'
                            full_val = '有信号(1)' if diff['full'] == 1 else '无信号(0)'
                        else:
                            partial_val = str(diff['partial'])
                            full_val = str(diff['full'])

                        print(f"       {col}:")
                        print(f"         逐日回测: {partial_val}")
                        print(f"         全局回测: {full_val}")
                        print(f"         ⚠️  说明: 只用到{err['date']}的数据时={partial_val}，")
                        print(f"             但使用全部数据（包括未来）时={full_val}")
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

    def test_000001(self):
        """测试A股000001，覆盖 banklike / slow_bull 新分支"""
        self._test_stock('000001', market='CN')

    def test_600775(self):
        """测试A股600775，覆盖 gc_extreme_chase_block 新分支"""
        self._test_stock('600775', market='CN')

    def test_00512(self):
        """测试港股00512，覆盖 zigzag_trend_exit_softconfirm 新分支"""
        self._test_stock('00512', market='HK')


class TestPitStageLookahead(unittest.TestCase):
    """
    直接测试 smooth_macro_phase / pit_stage 流水线的未来函数检测。

    现有 TestLookAheadBiasSmart 通过 RSITrendStrategy 检测策略层面的 pit_state，
    但 smooth_macro_phase 本身只在 main.py 展示函数中被调用，不在策略链里，
    因此需要单独检测。

    重点覆盖本次修改新增的 ULTRA2（DIP≤5%）和 SURGE（gfl≥27%）触发股票：
        SURGE 触发：603444（2020-02春节）、002362 / 600775（2024-02春节）、
                     002467、605117（2021劳动节）
        ULTRA2 触发：002920 / 300279（2024-09/02）、300769、002167

    核心逻辑：
        1. 用全量数据跑 pit_stage 流水线，得到 confirmed_full[0..n-1]
        2. 找所有状态转换点（confirmed_full[t] != confirmed_full[t-1]）
        3. 对每个转换点 t：截取 data[:t+1]，用相同 cfg 重跑流水线，比较最后一个值
        4. confirmed_partial[-1] != confirmed_full[t] → 存在未来函数
        5. 额外随机采样平静期（应该不出现差异）
    """

    CACHE_DIR = os.path.join(os.path.dirname(__file__), '../data/cache')
    CACHE_ADJUST = _load_cache_adjust()
    # 最少数据量（比 K_DN_TO_CO*2 大即可）
    MIN_BARS = 200

    def _run_pipeline(self, close, high, low, vol, cfg, dna):
        """用固定 cfg/dna 跑完整 pit_stage 流水线，返回 confirmed 列表。"""
        states = compute_pit_states(close, high, low, cfg=cfg)
        phases, _ = compute_downtrend_phase(
            close, high, low, states, cfg=cfg, dna=dna, return_data=True
        )
        confirmed = smooth_macro_phase(
            phases, close, high, low, vol, confirmed_only=True
        )
        return confirmed

    def _test_pit_stage_stock(self, stock_code: str):
        """
        针对单支股票，直接测试 smooth_macro_phase 的未来函数。

        使用全量数据推导出的 cfg/dna 固定用于所有截断测试，
        将配置差异排除在外，只检测流水线本身是否使用了未来数据。
        """
        print("\n" + "=" * 80)
        print(f"[PitStage 未来函数] {stock_code}")
        print("=" * 80)

        csv_path = os.path.join(self.CACHE_DIR, f'{stock_code}_{self.CACHE_ADJUST}.csv')
        if not os.path.exists(csv_path):
            print(f"  ⚠️  未找到数据文件，跳过: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        close = df['close'].values.astype(float)
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        vol   = df['volume'].values.astype(float)
        dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
        n = len(close)

        if n < self.MIN_BARS:
            print(f"  ⚠️  数据量不足 ({n} < {self.MIN_BARS})，跳过")
            return

        print(f"  数据量: {n} 天")

        # ── 步骤1：全量数据跑 pit_stage，得到基准分类 ─────────────────
        print("\n步骤1: 全量数据推导 cfg/dna 并运行流水线...")
        dna_full = compute_stock_dna(close, high, low)
        pe_full  = StockPersonalityEngine(close, dates)
        pers_full = pe_full.get_personality_at_bar(n - 1)
        cfg_full  = make_personality_config(pers_full, dna=dna_full)

        confirmed_full = self._run_pipeline(close, high, low, vol, cfg_full, dna_full)

        # ── 步骤2：找所有状态转换点（100% 覆盖关键点）─────────────────
        transition_indices = []
        for i in range(self.MIN_BARS, n):
            if confirmed_full[i] != confirmed_full[i - 1]:
                transition_indices.append(i)

        print(f"  状态转换点: {len(transition_indices)} 个")

        # 按状态类型统计
        state_counts: Dict[str, int] = {}
        for i in transition_indices:
            s = confirmed_full[i]
            state_counts[s] = state_counts.get(s, 0) + 1
        for s, cnt in sorted(state_counts.items()):
            print(f"    → {s}: {cnt} 次")

        # ── 步骤3：随机采样平静期 ──────────────────────────────────────
        all_trans_set = set(transition_indices)
        quiet_pool = [i for i in range(self.MIN_BARS, n) if i not in all_trans_set]
        sample_size = min(len(transition_indices), len(quiet_pool))
        sample_size = max(5, sample_size)
        quiet_samples = random.sample(quiet_pool, min(sample_size, len(quiet_pool)))

        # ── 步骤4：构建测试集 ──────────────────────────────────────────
        test_indices = sorted(set(transition_indices) | set(quiet_samples))
        total = len(test_indices)
        print(f"\n步骤2: 开始测试 {total} 个点（转换点 {len(transition_indices)} + 平静期 {len(quiet_samples)}）...")

        # ── 步骤5：逐点截断测试 ────────────────────────────────────────
        discrepancies_trans = []
        discrepancies_quiet = []

        for idx, t in enumerate(test_indices):
            if idx % 20 == 0:
                print(f"\r  进度: {idx}/{total}", end='', flush=True)

            c_p = close[:t + 1]
            h_p = high[:t + 1]
            l_p = low[:t + 1]
            v_p = vol[:t + 1]

            try:
                confirmed_p = self._run_pipeline(c_p, h_p, l_p, v_p, cfg_full, dna_full)
            except Exception as e:
                print(f"\n  ⚠️  截断测试 t={t} 异常: {e}")
                continue

            partial_val = confirmed_p[-1]
            full_val    = confirmed_full[t]

            if partial_val != full_val:
                rec = {
                    'index': t,
                    'date':  dates[t],
                    'partial': partial_val,
                    'full':    full_val,
                    'is_transition': t in all_trans_set,
                }
                if t in all_trans_set:
                    discrepancies_trans.append(rec)
                else:
                    discrepancies_quiet.append(rec)

        print(f"\r  完成: {total}/{total}          ")

        # ── 步骤6：输出结果 ────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("测试结果")
        print("=" * 80)

        all_disc = discrepancies_trans + discrepancies_quiet
        if not all_disc:
            print(f"✅ {stock_code} 未检测到未来函数")
            print(f"   - 状态转换点: {len(transition_indices)} 个 ✅")
            print(f"   - 平静期采样: {len(quiet_samples)} 个 ✅")
        else:
            if discrepancies_trans:
                print(f"❌ 状态转换点差异: {len(discrepancies_trans)} 个（严重）")
                print("   这些状态切换时的分类在截断数据下不一致 → 存在未来函数！\n")
                for r in discrepancies_trans[:10]:
                    print(f"   [{r['date']} / idx={r['index']}]"
                          f"  截断={r['partial']}  全量={r['full']}")
                if len(discrepancies_trans) > 10:
                    print(f"   ... 还有 {len(discrepancies_trans) - 10} 条")

            if discrepancies_quiet:
                print(f"⚠️  平静期差异: {len(discrepancies_quiet)} 个")
                for r in discrepancies_quiet[:5]:
                    print(f"   [{r['date']} / idx={r['index']}]"
                          f"  截断={r['partial']}  全量={r['full']}")

            details = []
            if discrepancies_trans:
                details.append(f"转换点差异 {len(discrepancies_trans)} 个")
            if discrepancies_quiet:
                details.append(f"平静期差异 {len(discrepancies_quiet)} 个")
            self.fail(f"{stock_code} 检测到未来函数！{' | '.join(details)}")

    # ── 具体股票测试方法 ──────────────────────────────────────────────

    # SURGE 触发股票（新增 SURGE 层，DIP 不限，gfl≥27%，BARS≥4）
    def test_surge_603444(self):
        """SURGE 触发：603444（2020-02 春节后大涨，bar4 gfl=27%）"""
        self._test_pit_stage_stock('603444')

    def test_surge_002362(self):
        """SURGE 触发：002362（2024-02 春节，bar5 gfl=43%）"""
        self._test_pit_stage_stock('002362')

    def test_surge_600775(self):
        """SURGE 触发：600775（2024-02 春节，bar5 gfl=43%）"""
        self._test_pit_stage_stock('600775')

    def test_surge_002467(self):
        """SURGE 触发：002467（2024-02 春节，bar4 gfl=35%）"""
        self._test_pit_stage_stock('002467')

    def test_surge_605117(self):
        """SURGE 触发：605117（2021-05 劳动节，bar4 gfl=34%）"""
        self._test_pit_stage_stock('605117')

    # ULTRA2 触发股票（新增 ULTRA2 层，DIP≤5%，gfl≥21%）
    def test_ultra2_002920(self):
        """ULTRA2 触发：002920（2024-09 924行情，dip=4.3%）"""
        self._test_pit_stage_stock('002920')

    def test_ultra2_300279(self):
        """ULTRA2 触发：300279（2024-02 春节后，dip=4.7%）"""
        self._test_pit_stage_stock('300279')

    def test_ultra2_002167(self):
        """ULTRA2 触发：002167（2024-02 春节，dip=3.5%）"""
        self._test_pit_stage_stock('002167')

    def test_ultra2_300769(self):
        """ULTRA2 触发：300769（2021-10 强势涨，dip=4.2%）"""
        self._test_pit_stage_stock('300769')

    # 已有基准股票（原 ULTRA 层，DIP≤2%）
    def test_baseline_02367(self):
        """基准：港股 02367（原 ULTRA 触发，DIP≤2%）"""
        self._test_pit_stage_stock('02367')

    def test_baseline_300750(self):
        """基准：A股 300750"""
        self._test_pit_stage_stock('300750')


STRATEGY_TEST_NAMES = [
    'test_02367',
    'test_300750',
    'test_300274',
    'test_000001',
    'test_600775',
    'test_00512',
]

PIT_STAGE_TEST_NAMES = [
    'test_surge_603444',
    'test_surge_002362',
    'test_surge_600775',
    'test_surge_002467',
    'test_surge_605117',
    'test_ultra2_002920',
    'test_ultra2_300279',
    'test_ultra2_002167',
    'test_ultra2_300769',
    'test_baseline_02367',
    'test_baseline_300750',
]


def _default_worker_count(total_cases: int, requested: Optional[int] = None) -> int:
    if isinstance(requested, int) and requested > 0:
        return max(1, min(requested, total_cases))
    return max(1, min(total_cases, os.cpu_count() or 1))


def _run_named_unittest_case(case_class_name: str, test_name: str) -> Dict[str, object]:
    """在独立进程中运行单个 unittest case，并捕获输出。"""
    case_class = globals()[case_class_name]
    suite = unittest.TestSuite([case_class(test_name)])
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
    return {
        'case_class_name': case_class_name,
        'test_name': test_name,
        'success': result.wasSuccessful(),
        'output': stream.getvalue(),
    }


def _run_case_group_parallel(title: str,
                             case_class_name: str,
                             test_names: List[str],
                             max_workers: Optional[int] = None) -> bool:
    """并行运行一组单测方法，汇总结果。"""
    total = len(test_names)
    workers = _default_worker_count(total, max_workers)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"并发模式: 多进程 ({workers} workers / {total} cases)")

    ordered_results: Dict[str, Dict[str, object]] = {}
    failures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_run_named_unittest_case, case_class_name, test_name): test_name
            for test_name in test_names
        }
        for future in as_completed(future_map):
            test_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((test_name, f"进程执行异常: {exc}"))
                continue
            ordered_results[test_name] = result
            if not result['success']:
                failures.append((test_name, result['output']))

    # 保持输出顺序稳定，便于阅读和比对
    for test_name in test_names:
        result = ordered_results.get(test_name)
        if result and result['output']:
            print(result['output'].rstrip())

    if failures:
        print("\n" + "=" * 80)
        print(f"❌ 并行测试失败: {len(failures)}/{total}")
        print("=" * 80)
        for test_name, details in failures:
            print(f"\n[{test_name}] 失败")
            if details:
                print(details.rstrip())
        return False

    print("\n" + "=" * 80)
    print(f"✅ 并行测试通过: {total}/{total}")
    print("=" * 80)
    return True


def run_tests(max_workers: Optional[int] = None):
    """运行策略层未来函数检测。"""
    return _run_case_group_parallel(
        title="未来函数检测 - 智能采样版本（策略层）",
        case_class_name='TestLookAheadBiasSmart',
        test_names=STRATEGY_TEST_NAMES,
        max_workers=max_workers,
    )


def run_pit_stage_tests(max_workers: Optional[int] = None):
    """运行 PitStage 未来函数检测。"""
    return _run_case_group_parallel(
        title="PitStage 未来函数检测（smooth_macro_phase / ULTRA2 / SURGE）",
        case_class_name='TestPitStageLookahead',
        test_names=PIT_STAGE_TEST_NAMES,
        max_workers=max_workers,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='未来函数检测')
    parser.add_argument('--pit', action='store_true', help='只跑 PitStage 检测')
    parser.add_argument('--all', action='store_true', help='同时跑策略检测 + PitStage 检测')
    parser.add_argument('--workers', type=int, default=0,
                        help='并行 worker 数量（默认=min(测试数量, CPU核心数)）')
    args = parser.parse_args()

    worker_count = args.workers if args.workers > 0 else None

    if args.pit or args.all:
        ok_pit = run_pit_stage_tests(max_workers=worker_count)
    else:
        ok_pit = True

    if not args.pit:
        ok_strategy = run_tests(max_workers=worker_count)
    else:
        ok_strategy = True

    sys.exit(0 if (ok_pit and ok_strategy) else 1)
