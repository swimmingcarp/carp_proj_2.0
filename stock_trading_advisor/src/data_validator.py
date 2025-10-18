"""
数据验证模块
提供数据完整性检查和异常值过滤功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器 - 检查数据质量和完整性"""

    def __init__(self, config: Dict = None):
        """
        初始化数据验证器

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
            # 异常值阈值（A股默认 - 主板）
            'max_price_change_pct': 10.0,      # 单日最大涨跌幅 (%) - 主板默认10%
            'max_consecutive_same': 5,          # 最大连续相同值
            'min_volume': 100,                  # 最小成交量
            'max_volume_ratio': 50.0,           # 最大成交量比率（与均值比）
            'min_price': 0.01,                  # 最小价格
            'max_price': 10000.0,               # 最大价格
            'min_data_points': 60,              # 最小数据点数量（约3个月）
            'max_missing_ratio': 0.1,           # 最大缺失率 (10%)
            'price_decimal_places': 2,          # 价格小数位数

            # A股不同板块的配置
            'CN-A-Main': {
                'max_price_change_pct': 10.5,   # 主板（60/00开头）：10% + 容差
            },
            'CN-A-ChiNext': {
                'max_price_change_pct': 20.5,   # 创业板（300开头）：20% + 容差
            },
            'CN-A-STAR': {
                'max_price_change_pct': 20.5,   # 科创板（688开头）：20% + 容差
            },
            'CN-A-Beijing': {
                'max_price_change_pct': 30.5,   # 北交所（8/4开头）：30% + 容差
            },

            # 港股特定配置
            'HK': {
                'max_price_change_pct': 50.0,   # 港股无涨跌停限制
                'min_price': 0.001,             # 港股最低价格（仙股）
                'max_price': 100000.0,          # 港股最高价格
                'price_decimal_places': 3,      # 港股价格可能有3位小数
            }
        }

    def validate(self, df: pd.DataFrame, stock_code: str = "", market: str = 'CN-A') -> Tuple[pd.DataFrame, Dict]:
        """
        完整的数据验证流程

        Args:
            df: 原始数据框
            stock_code: 股票代码（用于日志和判断板块）
            market: 市场类型 ('CN-A'-A股, 'HK'-港股, 'US'-美股)

        Returns:
            (清洗后的数据框, 验证报告字典)
        """
        if df is None or len(df) == 0:
            return None, {'status': 'FAILED', 'reason': '数据为空'}

        # 根据市场类型和股票代码调整配置
        effective_config = self._get_market_config(market, stock_code)

        report = {
            'stock_code': stock_code,
            'market': market,
            'original_rows': len(df),
            'issues': [],
            'warnings': [],
            'cleaned_rows': 0,
            'status': 'PASSED'
        }

        # 1. 基础完整性检查
        df, completeness_issues = self._check_completeness(df)
        report['issues'].extend(completeness_issues)

        if df is None:
            report['status'] = 'FAILED'
            return None, report

        # 2. 数据类型验证
        df, type_issues = self._check_data_types(df)
        report['issues'].extend(type_issues)

        # 3. 检查重复数据
        df, duplicate_count = self._remove_duplicates(df)
        if duplicate_count > 0:
            report['warnings'].append(f"删除了 {duplicate_count} 条重复数据")

        # 4. 异常值检测（使用市场特定配置）
        df, outlier_issues = self._detect_outliers(df, effective_config)
        report['issues'].extend(outlier_issues)

        # 5. 价格合理性检查（使用市场特定配置）
        df, price_issues = self._check_price_validity(df, effective_config)
        report['issues'].extend(price_issues)

        # 6. 成交量异常检测
        df, volume_issues = self._check_volume_validity(df)
        report['warnings'].extend(volume_issues)

        # 7. 连续性检查
        continuity_warnings = self._check_continuity(df)
        report['warnings'].extend(continuity_warnings)

        # 8. 最终数据量检查
        if len(df) < effective_config['min_data_points']:
            report['status'] = 'FAILED'
            report['issues'].append(
                f"数据量不足: {len(df)} < {effective_config['min_data_points']}"
            )
            return None, report

        report['cleaned_rows'] = len(df)
        report['data_quality_score'] = self._calculate_quality_score(report)

        # 判断最终状态
        if len(report['issues']) > 0:
            # 区分"已删除异常数据"和"未解决的严重问题"
            critical_issues = []
            for issue in report['issues']:
                if 'CRITICAL' in issue:
                    # 如果是已删除的异常数据，不算严重问题
                    if '删除' not in issue and '已自动修正' not in issue:
                        critical_issues.append(issue)

            if critical_issues:
                # 只有存在未解决的严重问题才标记为失败
                report['status'] = 'FAILED'
            else:
                # 异常数据已被删除或修正，标记为警告
                report['status'] = 'WARNING'

        return df, report

    def _get_market_config(self, market: str, stock_code: str = "") -> Dict:
        """
        获取市场特定的配置参数

        Args:
            market: 市场类型
            stock_code: 股票代码（用于判断A股板块）

        Returns:
            合并后的配置字典
        """
        base_config = self.config.copy()

        # 如果是港股，覆盖特定参数
        if market == 'HK' and 'HK' in self.config:
            hk_config = self.config['HK']
            base_config.update(hk_config)
        # 如果是A股，根据股票代码判断板块
        elif market == 'CN-A' and stock_code:
            # 判断A股板块
            if stock_code.startswith('300'):
                # 创业板
                if 'CN-A-ChiNext' in self.config:
                    base_config.update(self.config['CN-A-ChiNext'])
            elif stock_code.startswith('688'):
                # 科创板
                if 'CN-A-STAR' in self.config:
                    base_config.update(self.config['CN-A-STAR'])
            elif stock_code.startswith(('8', '4')) and len(stock_code) == 6:
                # 北交所（6位数字，8或4开头）
                if 'CN-A-Beijing' in self.config:
                    base_config.update(self.config['CN-A-Beijing'])
            else:
                # 主板（60/00开头或其他）
                if 'CN-A-Main' in self.config:
                    base_config.update(self.config['CN-A-Main'])

        return base_config

    def _check_completeness(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """检查数据完整性"""
        issues = []

        # 检查必需列
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            issues.append(f"CRITICAL: 缺少必需列: {missing_columns}")
            return None, issues

        # 检查缺失值比例
        for col in required_columns:
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio > self.config['max_missing_ratio']:
                issues.append(
                    f"CRITICAL: 列 '{col}' 缺失率过高: {missing_ratio:.2%}"
                )
                return None, issues
            elif missing_ratio > 0:
                issues.append(
                    f"列 '{col}' 存在 {df[col].isna().sum()} 个缺失值"
                )

        # 删除有缺失值的行
        original_len = len(df)
        df = df.dropna(subset=required_columns)
        removed = original_len - len(df)
        if removed > 0:
            issues.append(f"删除了 {removed} 行包含缺失值的数据")

        return df, issues

    def _check_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """检查并转换数据类型"""
        issues = []

        # 转换数值列
        numeric_columns = ['open', 'close', 'high', 'low', 'volume']
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                issues.append(f"列 '{col}' 类型转换失败: {e}")

        # 确保日期列是字符串格式
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)

        return df, issues

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """删除重复数据"""
        original_len = len(df)
        df = df.drop_duplicates(subset=['date'], keep='first')
        removed = original_len - len(df)
        return df, removed

    def _detect_outliers(self, df: pd.DataFrame, config: Dict = None) -> Tuple[pd.DataFrame, List[str]]:
        """检测异常值（价格涨跌幅）"""
        issues = []
        if config is None:
            config = self.config

        if len(df) < 2:
            return df, issues

        # 计算单日涨跌幅
        df = df.sort_values('date').reset_index(drop=True)
        df['daily_change_pct'] = df['close'].pct_change() * 100

        # 检测异常涨跌幅
        max_change = config['max_price_change_pct']
        outliers = df[abs(df['daily_change_pct']) > max_change]

        if len(outliers) > 0:
            for idx, row in outliers.iterrows():
                # ST股票、新股、港股仙股可能出现大幅波动，只记录警告
                change = row['daily_change_pct']
                issues.append(
                    f"异常涨跌幅: {row['date']} ({change:.2f}%)"
                )

        return df, issues

    def _check_price_validity(self, df: pd.DataFrame, config: Dict = None) -> Tuple[pd.DataFrame, List[str]]:
        """检查价格合理性"""
        issues = []
        if config is None:
            config = self.config

        # 检查价格范围
        for col in ['open', 'close', 'high', 'low']:
            min_price = df[col].min()
            max_price = df[col].max()

            if min_price < config['min_price']:
                issues.append(f"CRITICAL: {col} 存在异常低价: {min_price}")

            if max_price > config['max_price']:
                issues.append(f"CRITICAL: {col} 存在异常高价: {max_price}")

        # 检查 OHLC 逻辑关系并自动修正小误差
        # 1. 检查最高价
        max_ohlc = df[['open', 'close', 'low']].max(axis=1)
        invalid_high = df['high'] < max_ohlc

        if invalid_high.any():
            # 计算误差
            error_amount = max_ohlc - df['high']

            # 如果误差很小（<= 当前价格的1%），自动修正
            auto_fix_mask = invalid_high & (error_amount <= df['high'] * 0.01)
            if auto_fix_mask.any():
                df.loc[auto_fix_mask, 'high'] = max_ohlc.loc[auto_fix_mask]
                issues.append(
                    f"已自动修正 {auto_fix_mask.sum()} 条最高价小误差（<1%）"
                )

            # 误差较大的删除
            large_error_mask = invalid_high & ~auto_fix_mask
            if large_error_mask.any():
                large_errors = df[large_error_mask]
                issues.append(
                    f"已删除 {len(large_errors)} 条最高价异常数据（误差>1%）"
                )
                df = df.drop(large_errors.index)

        # 2. 检查最低价（在删除异常最高价后重新计算索引）
        df = df.reset_index(drop=True)
        min_ohlc = df[['open', 'close', 'high']].min(axis=1)
        invalid_low = df['low'] > min_ohlc

        if invalid_low.any():
            # 计算误差
            error_amount = df['low'] - min_ohlc

            # 如果误差很小（<= 当前价格的1%），自动修正
            auto_fix_mask = invalid_low & (error_amount <= df['low'] * 0.01)
            if auto_fix_mask.any():
                df.loc[auto_fix_mask, 'low'] = min_ohlc.loc[auto_fix_mask]
                issues.append(
                    f"已自动修正 {auto_fix_mask.sum()} 条最低价小误差（<1%）"
                )

            # 误差较大的删除
            large_error_mask = invalid_low & ~auto_fix_mask
            if large_error_mask.any():
                large_errors = df[large_error_mask]
                issues.append(
                    f"已删除 {len(large_errors)} 条最低价异常数据（误差>1%）"
                )
                df = df.drop(large_errors.index)

        # 检查连续相同价格（可能是停牌）
        for col in ['close']:
            consecutive_same = (df[col] == df[col].shift(1)).astype(int)
            max_consecutive = consecutive_same.groupby(
                (consecutive_same != consecutive_same.shift()).cumsum()
            ).sum().max()

            if max_consecutive >= config.get('max_consecutive_same', self.config['max_consecutive_same']):
                issues.append(
                    f"警告: {col} 存在连续 {max_consecutive} 天相同值（可能停牌）"
                )

        return df.reset_index(drop=True), issues

    def _check_volume_validity(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """检查成交量合理性"""
        warnings = []

        # 检查零成交量（可能是停牌）
        zero_volume = df[df['volume'] == 0]
        if len(zero_volume) > 0:
            warnings.append(
                f"存在 {len(zero_volume)} 天零成交量数据（可能停牌）"
            )

        # 检查异常大成交量
        if len(df) > 10:
            mean_volume = df['volume'].mean()
            std_volume = df['volume'].std()

            if mean_volume > 0:
                df['volume_ratio'] = df['volume'] / mean_volume
                high_volume = df[df['volume_ratio'] > self.config['max_volume_ratio']]

                if len(high_volume) > 0:
                    warnings.append(
                        f"存在 {len(high_volume)} 天异常大成交量"
                    )

                df = df.drop(columns=['volume_ratio'])

        return df, warnings

    def _check_continuity(self, df: pd.DataFrame) -> List[str]:
        """检查时间连续性"""
        warnings = []

        if len(df) < 2:
            return warnings

        # 转换日期并排序
        df_sorted = df.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(df_sorted['date'])

        # 检查日期间隔（排除周末）
        gaps = []
        for i in range(1, len(dates)):
            date_diff = (dates.iloc[i] - dates.iloc[i-1]).days
            # 正常情况下应该是1-3天（考虑周末）
            if date_diff > 7:
                gaps.append((dates.iloc[i-1].strftime('%Y-%m-%d'),
                           dates.iloc[i].strftime('%Y-%m-%d'),
                           date_diff))

        if len(gaps) > 0:
            warnings.append(
                f"存在 {len(gaps)} 个时间间隔 > 7天（可能是假期或停牌）"
            )
            if len(gaps) <= 3:
                for start, end, days in gaps:
                    warnings.append(f"  - {start} 到 {end}: {days} 天")

        return warnings

    def _calculate_quality_score(self, report: Dict) -> float:
        """
        计算数据质量评分 (0-100)

        扣分项:
        - 每个 CRITICAL 问题扣 30 分
        - 每个普通问题扣 10 分
        - 每个警告扣 5 分
        """
        score = 100.0

        for issue in report['issues']:
            if 'CRITICAL' in issue:
                score -= 30
            else:
                score -= 10

        score -= len(report['warnings']) * 5

        return max(0.0, score)

    def format_report(self, report: Dict) -> str:
        """格式化验证报告"""
        if report.get('status') == 'FAILED':
            result = "❌ 数据验证失败"
        elif report.get('status') == 'WARNING':
            result = "⚠️  数据验证通过（有警告）"
        else:
            result = "✓ 数据验证通过"

        output = [
            "\n" + "=" * 50,
            f"数据验证报告 - {report.get('stock_code', 'Unknown')}",
            "=" * 50,
            result,
            f"原始数据行数: {report.get('original_rows', 0)}",
            f"清洗后行数: {report.get('cleaned_rows', 0)}",
            f"数据质量评分: {report.get('data_quality_score', 0):.1f}/100"
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
