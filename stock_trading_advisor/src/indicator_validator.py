"""
技术指标验证模块
验证 KDJ、MACD、MA 等技术指标的合理性
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class IndicatorValidator:
    """技术指标验证器 - 检查指标计算的合理性"""

    def __init__(self, config: Dict = None):
        """
        初始化指标验证器

        Args:
            config: 验证配置参数
        """
        # 先加载默认配置
        self.config = self._default_config()

        # 如果有自定义配置，合并到默认配置
        if config:
            self.config.update(config)

    def _default_config(self) -> Dict:
        """默认验证配置"""
        return {
            # KDJ 验证阈值
            'kdj_min': 0.0,              # KDJ 最小值
            'kdj_max': 100.0,            # KDJ 最大值
            'kdj_extreme_ratio': 0.3,    # KDJ 极值（>90 或 <10）占比阈值
            'max_kdj_consecutive_same': 10,  # KDJ 最大连续相同值

            # MACD 验证阈值
            'max_macd_abs': 10.0,        # MACD 绝对值上限（相对于价格）
            'macd_zero_crossing_min': 2, # 最少零轴穿越次数
            'diff_dea_correlation': 0.8, # DIFF 和 DEA 相关性下限

            # MA 验证阈值
            'ma_price_deviation': 0.5,   # MA 与价格的最大偏离倍数
            'ma_order_violation_ratio': 0.1,  # MA 顺序违背比率
            'ma_smoothness': 0.95,       # MA 平滑度（相邻差异）

            # 通用
            'min_valid_data_ratio': 0.8, # 最小有效数据比率
        }

    def validate(self, df: pd.DataFrame, stock_code: str = "") -> Tuple[bool, Dict]:
        """
        完整的指标验证流程

        Args:
            df: 包含技术指标的 DataFrame
            stock_code: 股票代码（用于日志）

        Returns:
            (是否通过验证, 验证报告字典)
        """
        if df is None or len(df) == 0:
            return False, {'status': 'FAILED', 'reason': '数据为空'}

        report = {
            'stock_code': stock_code,
            'total_rows': len(df),
            'issues': [],
            'warnings': [],
            'status': 'PASSED',
            'indicators_checked': []
        }

        # 1. 检查 KDJ 指标
        if all(col in df.columns for col in ['k', 'd']):
            kdj_passed, kdj_issues = self._validate_kdj(df)
            report['indicators_checked'].append('KDJ')
            if not kdj_passed:
                report['issues'].extend(kdj_issues)
                report['status'] = 'WARNING'
            elif kdj_issues:
                report['warnings'].extend(kdj_issues)

        # 2. 检查 MACD 指标
        if all(col in df.columns for col in ['diff', 'dea', 'macd']):
            macd_passed, macd_issues = self._validate_macd(df)
            report['indicators_checked'].append('MACD')
            if not macd_passed:
                report['issues'].extend(macd_issues)
                report['status'] = 'WARNING'
            elif macd_issues:
                report['warnings'].extend(macd_issues)

        # 3. 检查移动平均线
        ma_columns = [col for col in df.columns if col.endswith('_ma') and col[:-3].isdigit()]
        if ma_columns:
            ma_passed, ma_issues = self._validate_ma(df, ma_columns)
            report['indicators_checked'].append(f'MA({len(ma_columns)}条)')
            if not ma_passed:
                report['issues'].extend(ma_issues)
                report['status'] = 'WARNING'
            elif ma_issues:
                report['warnings'].extend(ma_issues)

        # 4. 检查数据有效性
        validity_issues = self._check_data_validity(df, report['indicators_checked'])
        if validity_issues:
            report['warnings'].extend(validity_issues)

        # 判断最终状态
        if len(report['issues']) > 3:
            report['status'] = 'FAILED'

        return report['status'] != 'FAILED', report

    def _validate_kdj(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证 KDJ 指标合理性"""
        issues = []
        passed = True

        # 检查 K 值
        if 'k' in df.columns:
            k_valid = df['k'].dropna()
            if len(k_valid) > 0:
                # 1. 范围检查
                k_out_of_range = ((k_valid < self.config['kdj_min']) |
                                  (k_valid > self.config['kdj_max'])).sum()
                if k_out_of_range > 0:
                    issues.append(f"CRITICAL: K 值有 {k_out_of_range} 条超出范围 [0, 100]")
                    passed = False

                # 2. 极值分布检查
                k_extreme = ((k_valid > 90) | (k_valid < 10)).sum()
                extreme_ratio = k_extreme / len(k_valid)
                if extreme_ratio > self.config['kdj_extreme_ratio']:
                    issues.append(
                        f"K 值极值占比过高: {extreme_ratio:.1%} (阈值: {self.config['kdj_extreme_ratio']:.1%})"
                    )

                # 3. 连续相同值检查
                k_consecutive = (k_valid == k_valid.shift(1)).astype(int)
                max_consecutive = k_consecutive.groupby(
                    (k_consecutive != k_consecutive.shift()).cumsum()
                ).sum().max()

                if max_consecutive >= self.config['max_kdj_consecutive_same']:
                    issues.append(
                        f"K 值存在连续 {max_consecutive} 天相同（可能计算异常）"
                    )

        # 检查 D 值（类似逻辑）
        if 'd' in df.columns:
            d_valid = df['d'].dropna()
            if len(d_valid) > 0:
                d_out_of_range = ((d_valid < self.config['kdj_min']) |
                                  (d_valid > self.config['kdj_max'])).sum()
                if d_out_of_range > 0:
                    issues.append(f"CRITICAL: D 值有 {d_out_of_range} 条超出范围 [0, 100]")
                    passed = False

        # 检查 K 和 D 的关系
        if 'k' in df.columns and 'd' in df.columns:
            valid_idx = df[['k', 'd']].dropna().index
            if len(valid_idx) > 10:
                # K 和 D 应该高度相关
                correlation = df.loc[valid_idx, 'k'].corr(df.loc[valid_idx, 'd'])
                if correlation < 0.7:
                    issues.append(
                        f"K 和 D 相关性较低: {correlation:.2f} (可能计算错误)"
                    )

        return passed, issues

    def _validate_macd(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证 MACD 指标合理性"""
        issues = []
        passed = True

        diff_valid = df['diff'].dropna()
        dea_valid = df['dea'].dropna()
        macd_valid = df['macd'].dropna()

        if len(diff_valid) == 0 or len(dea_valid) == 0:
            issues.append("CRITICAL: MACD 指标全部为空")
            return False, issues

        # 1. 检查 DIFF 和 DEA 的相关性
        valid_idx = df[['diff', 'dea']].dropna().index
        if len(valid_idx) > 10:
            correlation = df.loc[valid_idx, 'diff'].corr(df.loc[valid_idx, 'dea'])
            if correlation < self.config['diff_dea_correlation']:
                issues.append(
                    f"CRITICAL: DIFF 和 DEA 相关性异常低: {correlation:.2f}"
                )
                passed = False

        # 2. 检查 MACD 柱状图的计算正确性
        # MACD = (DIFF - DEA) * 2
        valid_idx = df[['diff', 'dea', 'macd']].dropna().index
        if len(valid_idx) > 0:
            expected_macd = (df.loc[valid_idx, 'diff'] - df.loc[valid_idx, 'dea']) * 2
            macd_error = np.abs(df.loc[valid_idx, 'macd'] - expected_macd).max()
            if macd_error > 0.01:
                issues.append(
                    f"MACD 计算可能有误，最大误差: {macd_error:.4f}"
                )

        # 3. 检查零轴穿越次数
        if len(macd_valid) > 20:
            zero_crossings = ((macd_valid > 0) != (macd_valid.shift(1) > 0)).sum()
            if zero_crossings < self.config['macd_zero_crossing_min']:
                issues.append(
                    f"MACD 零轴穿越次数过少: {zero_crossings} (数据可能单调或计算异常)"
                )

        # 4. 检查 MACD 极值是否合理（相对于价格）
        if 'close' in df.columns and len(macd_valid) > 0:
            price_mean = df['close'].mean()
            macd_max = macd_valid.abs().max()
            macd_ratio = macd_max / price_mean

            if macd_ratio > self.config['max_macd_abs']:
                issues.append(
                    f"MACD 绝对值过大: {macd_max:.2f} (价格均值: {price_mean:.2f})"
                )

        return passed, issues

    def _validate_ma(self, df: pd.DataFrame, ma_columns: List[str]) -> Tuple[bool, List[str]]:
        """验证移动平均线合理性"""
        issues = []
        passed = True

        if 'close' not in df.columns:
            return passed, issues

        close = df['close']

        for ma_col in ma_columns:
            ma_data = df[ma_col].dropna()
            if len(ma_data) == 0:
                continue

            # 提取周期数
            period = int(ma_col.split('_')[0])

            # 1. 检查 MA 与价格的偏离
            valid_idx = df[[ma_col, 'close']].dropna().index
            if len(valid_idx) > 0:
                ma_price_ratio = df.loc[valid_idx, ma_col] / df.loc[valid_idx, 'close']

                # MA 不应该偏离价格太远
                max_deviation = ma_price_ratio.max()
                min_deviation = ma_price_ratio.min()

                if max_deviation > (1 + self.config['ma_price_deviation']):
                    issues.append(
                        f"{ma_col} 超出价格 {(max_deviation-1)*100:.1f}% (可能计算错误)"
                    )

                if min_deviation < (1 - self.config['ma_price_deviation']):
                    issues.append(
                        f"{ma_col} 低于价格 {(1-min_deviation)*100:.1f}% (可能计算错误)"
                    )

            # 2. 检查 MA 的平滑性
            if len(ma_data) > 2:
                ma_changes = ma_data.pct_change().abs()
                # 移动平均线应该比价格更平滑
                price_changes = close.pct_change().abs()

                valid_compare_idx = ma_changes.dropna().index.intersection(
                    price_changes.dropna().index
                )

                if len(valid_compare_idx) > 10:
                    ma_volatility = ma_changes.loc[valid_compare_idx].mean()
                    price_volatility = price_changes.loc[valid_compare_idx].mean()

                    # MA 的波动应该小于价格波动
                    if ma_volatility > price_volatility * self.config['ma_smoothness']:
                        issues.append(
                            f"{ma_col} 波动过大，平滑性不足 "
                            f"(MA波动: {ma_volatility:.4f}, 价格波动: {price_volatility:.4f})"
                        )

        # 3. 检查不同周期 MA 的顺序关系
        ma_dict = {}
        for ma_col in ma_columns:
            period = int(ma_col.split('_')[0])
            ma_dict[period] = ma_col

        # 按周期排序
        sorted_periods = sorted(ma_dict.keys())

        if len(sorted_periods) >= 2:
            # 检查长期 MA 是否比短期 MA 更平滑
            short_ma_col = ma_dict[sorted_periods[0]]
            long_ma_col = ma_dict[sorted_periods[-1]]

            valid_idx = df[[short_ma_col, long_ma_col]].dropna().index
            if len(valid_idx) > 20:
                # 在上升趋势中，短期MA应该在长期MA上方的概率更高
                # 这里只检查它们是否有交叉（正常现象）
                crossings = (
                    (df.loc[valid_idx, short_ma_col] > df.loc[valid_idx, long_ma_col]) !=
                    (df.loc[valid_idx, short_ma_col].shift(1) > df.loc[valid_idx, long_ma_col].shift(1))
                ).sum()

                if crossings == 0:
                    issues.append(
                        f"{short_ma_col} 和 {long_ma_col} 从未交叉（可能异常）"
                    )

        return passed, issues

    def _check_data_validity(self, df: pd.DataFrame, indicators: List[str]) -> List[str]:
        """检查指标数据的有效性"""
        warnings = []

        for indicator in indicators:
            if indicator == 'KDJ':
                cols = ['k', 'd']
            elif indicator == 'MACD':
                cols = ['diff', 'dea', 'macd']
            elif indicator.startswith('MA'):
                cols = [col for col in df.columns if col.endswith('_ma')]
            else:
                continue

            for col in cols:
                if col not in df.columns:
                    continue

                valid_count = df[col].notna().sum()
                total_count = len(df)
                valid_ratio = valid_count / total_count

                if valid_ratio < self.config['min_valid_data_ratio']:
                    warnings.append(
                        f"{col} 有效数据比率过低: {valid_ratio:.1%} "
                        f"(阈值: {self.config['min_valid_data_ratio']:.1%})"
                    )

        return warnings

    def format_report(self, report: Dict) -> str:
        """格式化验证报告"""
        if report.get('status') == 'FAILED':
            result = "❌ 指标验证失败"
        elif report.get('status') == 'WARNING':
            result = "⚠️  指标验证通过（有警告）"
        else:
            result = "✓ 指标验证通过"

        indicators_str = ', '.join(report.get('indicators_checked', []))

        output = [
            "\n" + "=" * 50,
            f"技术指标验证报告 - {report.get('stock_code', 'Unknown')}",
            "=" * 50,
            result,
            f"数据行数: {report.get('total_rows', 0)}",
            f"已检查指标: {indicators_str}"
        ]

        if report.get('issues'):
            output.append("\n问题列表:")
            for issue in report['issues']:
                output.append(f"  • {issue}")

        if report.get('warnings'):
            output.append("\n警告信息:")
            for warning in report['warnings']:
                output.append(f"  • {warning}")

        output.append("=" * 50)

        return "\n".join(output)
