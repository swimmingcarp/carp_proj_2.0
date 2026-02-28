"""
多尺度滑动窗口特征提取 + 分段合并

类CNN方案：
  Layer 1: 多尺度(5/10/20/40/60/120天)滚动线性回归，逐bar计算slope/R²/volatility
  Layer 2: 综合多尺度特征（加权），判断每bar的局部regime
  Layer 3: 按方向(up/down/sideways)合并连续bar为segments

【无未来函数保证】
  - rolling_linear_regression: bar[i]的slope只用[i-W+1, i]范围的数据
  - detect_regimes: bar[i]的regime只用bar[i]时刻的多尺度特征
  - get_personality_at_bar: 取regimes[bar_idx-window+1 : bar_idx+1]切片
  - 回测时在每次入场前调用，只看过去，不看未来
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# 分析尺度（交易日天数）
SCALES = [5, 10, 20, 40, 60, 120]

# 各尺度权重：中长期尺度权重更高，抑制短期噪声
SCALE_WEIGHTS = {5: 0.3, 10: 0.5, 20: 1.0, 40: 1.5, 60: 2.0, 120: 2.0}

# 斜率阈值（年化log空间，×252）
SLOPE_STEEP = 0.40      # >40% 年化 = 陡峭
SLOPE_MODERATE = 0.15   # >15% 年化 = 温和
SLOPE_FLAT = 0.05       # <5% 年化 = 横盘

# R²阈值
R2_HIGH = 0.60
R2_LOW = 0.25

# 最小段长度（合并后短于此的段被吸收）
MIN_SEGMENT_BARS = 15


@dataclass
class Segment:
    start_idx: int
    end_idx: int
    start_date: str = ''
    end_date: str = ''
    duration: int = 0
    regime: str = ''
    avg_slope: float = 0.0          # 年化斜率
    avg_r_squared: float = 0.0
    avg_volatility: float = 0.0
    slope_by_scale: Dict[int, float] = field(default_factory=dict)


def rolling_linear_regression(log_close: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化滚动线性回归，O(N)计算。
    slopes[i]只用log_close[i-W+1 : i+1]，纯回望，无未来数据。
    """
    N = len(log_close)
    slopes = np.full(N, np.nan)
    r_squared = np.full(N, np.nan)

    if N < window:
        return slopes, r_squared

    W = window
    sum_x = W * (W - 1) / 2.0
    sum_x2 = W * (W - 1) * (2 * W - 1) / 6.0
    denom = W * sum_x2 - sum_x * sum_x

    if abs(denom) < 1e-12:
        return slopes, r_squared

    y = log_close
    cumsum_y = np.concatenate([[0.0], np.cumsum(y)])
    roll_sum_y = cumsum_y[W:] - cumsum_y[:-W]

    idx = np.arange(N, dtype=np.float64)
    wy = idx * y
    cumsum_wy = np.concatenate([[0.0], np.cumsum(wy)])
    roll_sum_wy = cumsum_wy[W:] - cumsum_wy[:-W]

    starts = np.arange(N - W + 1, dtype=np.float64)
    roll_sum_xy = roll_sum_wy - starts * roll_sum_y
    slope_vals = (W * roll_sum_xy - sum_x * roll_sum_y) / denom

    cumsum_y2 = np.concatenate([[0.0], np.cumsum(y * y)])
    roll_sum_y2 = cumsum_y2[W:] - cumsum_y2[:-W]
    ss_tot = roll_sum_y2 - roll_sum_y * roll_sum_y / W
    ss_x = sum_x2 - sum_x * sum_x / W

    r2_vals = np.where(
        ss_tot > 1e-12,
        np.clip(slope_vals * slope_vals * ss_x / ss_tot, 0, 1),
        0.0
    )

    # slopes[W-1]对应窗口[0, W-1]，即bar W-1看过去W天。无未来数据。
    slopes[W - 1:] = slope_vals
    r_squared[W - 1:] = r2_vals

    return slopes, r_squared


def rolling_volatility(close: np.ndarray, window: int) -> np.ndarray:
    """滚动收益率波动率（向量化），vol[i]只用close[i-W:i+1]。"""
    N = len(close)
    vol = np.full(N, np.nan)
    if N < window + 1:
        return vol

    returns = np.diff(np.log(np.clip(close, 1e-6, None)))
    W = window
    n_valid = len(returns) - W + 1
    if n_valid <= 0:
        return vol

    cumsum_r = np.concatenate([[0.0], np.cumsum(returns)])
    cumsum_r2 = np.concatenate([[0.0], np.cumsum(returns * returns)])

    roll_sum = cumsum_r[W:W + n_valid] - cumsum_r[:n_valid]
    roll_sum2 = cumsum_r2[W:W + n_valid] - cumsum_r2[:n_valid]

    variance = (roll_sum2 - roll_sum * roll_sum / W) / max(W - 1, 1)
    variance = np.clip(variance, 0, None)

    vol[W:W + n_valid] = np.sqrt(variance)
    return vol


def multi_scale_features(close: np.ndarray, scales: List[int] = None) -> Dict[int, Dict[str, np.ndarray]]:
    """
    计算多尺度滑窗特征（全量序列，每个bar只用过去数据）。
    返回: {scale: {'slope': array(N), 'r2': array(N), 'vol': array(N), 'strength': array(N)}}
    """
    if scales is None:
        scales = SCALES

    log_close = np.log(np.clip(close, 1e-6, None))
    features = {}

    for scale in scales:
        slope, r2 = rolling_linear_regression(log_close, scale)
        vol = rolling_volatility(close, scale)
        slope_annual = slope * 252
        strength = np.abs(slope_annual) * np.where(np.isnan(r2), 0, r2)

        features[scale] = {
            'slope': slope_annual,
            'r2': r2,
            'vol': vol,
            'strength': strength,
        }

    return features


def detect_regimes(features: Dict[int, Dict[str, np.ndarray]], N: int) -> np.ndarray:
    """
    Layer 2: 每个bar的regime只用该bar的多尺度特征（加权平均，无未来数据）。
    中长期尺度(20-120天)权重更高，抑制短期噪声。
    """
    regimes = np.array(['unknown'] * N, dtype=object)

    available_scales = sorted(features.keys())

    for i in range(N):
        weighted_slope_sum = 0.0
        weighted_r2_sum = 0.0
        total_weight = 0.0

        short_slopes = []
        short_r2_vals = []
        long_slopes = []

        for s in available_scales:
            slope_val = features[s]['slope'][i]
            r2_val = features[s]['r2'][i]

            if np.isnan(slope_val):
                continue

            w = SCALE_WEIGHTS.get(s, 1.0)
            r2_clean = r2_val if not np.isnan(r2_val) else 0.0
            weighted_slope_sum += slope_val * w
            weighted_r2_sum += r2_clean * w
            total_weight += w

            if s <= 10:
                short_slopes.append(slope_val)
                short_r2_vals.append(r2_clean)
            elif s >= 60:
                long_slopes.append(slope_val)

        if total_weight == 0:
            continue

        avg_slope = weighted_slope_sum / total_weight
        avg_r2 = weighted_r2_sum / total_weight
        avg_short_slope = np.mean(short_slopes) if short_slopes else 0
        avg_short_r2 = np.mean(short_r2_vals) if short_r2_vals else 0
        avg_long_slope = np.mean(long_slopes) if long_slopes else 0

        # breakout: 短期陡峭+短R²高 + 长期横盘（更严格）
        if (short_slopes and long_slopes
                and avg_short_slope > SLOPE_STEEP
                and avg_short_r2 > R2_LOW
                and abs(avg_long_slope) < SLOPE_FLAT):
            regimes[i] = 'breakout_up'
            continue

        # reversal: 短期和长期强烈分歧（更严格：短期需要STEEP级别）
        if (short_slopes and long_slopes
                and avg_short_slope * avg_long_slope < 0
                and abs(avg_short_slope) > SLOPE_STEEP
                and abs(avg_long_slope) > SLOPE_MODERATE):
            regimes[i] = 'reversal_up' if avg_short_slope > 0 else 'reversal_down'
            continue

        # 主分类：使用加权平均
        if avg_r2 < R2_LOW and abs(avg_slope) < SLOPE_MODERATE:
            regimes[i] = 'sideways'
        elif abs(avg_slope) < SLOPE_FLAT:
            regimes[i] = 'sideways'
        elif avg_slope > SLOPE_STEEP and avg_r2 > R2_LOW:
            regimes[i] = 'strong_up'
        elif avg_slope > SLOPE_MODERATE:
            regimes[i] = 'moderate_up'
        elif avg_slope > SLOPE_FLAT:
            regimes[i] = 'weak_up'
        elif avg_slope < -SLOPE_STEEP and avg_r2 > R2_LOW:
            regimes[i] = 'strong_down'
        elif avg_slope < -SLOPE_MODERATE:
            regimes[i] = 'moderate_down'
        elif avg_slope < -SLOPE_FLAT:
            regimes[i] = 'weak_down'
        else:
            regimes[i] = 'sideways'

    return regimes


def _regime_direction(regime: str) -> str:
    """Map fine-grained regime to direction group (up/down/sideways)."""
    if regime in ('strong_up', 'moderate_up', 'weak_up', 'breakout_up', 'reversal_up'):
        return 'up'
    elif regime in ('strong_down', 'moderate_down', 'weak_down', 'reversal_down'):
        return 'down'
    elif regime == 'sideways':
        return 'sideways'
    return 'other'


def _dominant_regime(regimes_slice) -> str:
    """Find the most common regime in a slice."""
    counts = Counter(regimes_slice)
    counts.pop('unknown', None)
    if not counts:
        return 'unknown'
    return counts.most_common(1)[0][0]


def merge_segments(regimes: np.ndarray, features: Dict[int, Dict[str, np.ndarray]],
                   dates: Optional[np.ndarray] = None,
                   offset: int = 0,
                   min_bars: int = MIN_SEGMENT_BARS) -> List[Segment]:
    """
    Layer 3: 按方向(up/down/sideways)合并连续bar为segments。

    流程:
    1. 逐bar映射为方向(up/down/sideways)
    2. 合并连续同方向的bar
    3. 吸收短段(<min_bars)到相邻段
    4. 再次合并相邻同方向段
    5. 用dominant regime标记每段

    offset: regimes[0]对应全局features中的第offset个bar（用于切片场景）。
    """
    N = len(regimes)
    if N == 0:
        return []

    # Step 1: Map each bar to direction group
    directions = np.array([_regime_direction(r) for r in regimes])

    # Step 2: Initial merge by direction
    raw = []
    seg_start = 0
    for i in range(1, N):
        if directions[i] != directions[seg_start]:
            raw.append((seg_start, i))
            seg_start = i
    raw.append((seg_start, N))

    # Step 3: Absorb short segments into neighbors
    merged = list(raw)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            start, end = merged[i]
            duration = end - start
            if duration < min_bars and len(new_merged) > 0:
                # Absorb into previous segment
                prev_start, prev_end = new_merged[-1]
                new_merged[-1] = (prev_start, end)
                changed = True
            elif duration < min_bars and i + 1 < len(merged):
                # Absorb into next segment
                next_start, next_end = merged[i + 1]
                merged[i + 1] = (start, next_end)
                changed = True
            else:
                new_merged.append((start, end))
            i += 1
        merged = new_merged

    # Step 4: Re-merge adjacent same-direction segments
    final = []
    for start, end in merged:
        dom = _dominant_regime(regimes[start:end])
        direction = _regime_direction(dom)
        if final:
            prev_start, prev_end, prev_dir = final[-1]
            if direction == prev_dir:
                final[-1] = (prev_start, end, prev_dir)
                continue
        final.append((start, end, direction))

    # Step 5: Build Segment objects
    scales = sorted(features.keys())
    segments = []
    for start, end, direction in final:
        duration = end - start
        regime = _dominant_regime(regimes[start:end])

        slope_by_scale = {}
        all_slopes = []
        all_r2 = []
        all_vol = []

        for scale in scales:
            g_start = offset + start
            g_end = offset + end
            feat_len = len(features[scale]['slope'])
            g_start = max(0, min(g_start, feat_len))
            g_end = max(0, min(g_end, feat_len))

            s_slice = features[scale]['slope'][g_start:g_end]
            r2_slice = features[scale]['r2'][g_start:g_end]
            vol_slice = features[scale]['vol'][g_start:g_end]

            valid_s = s_slice[~np.isnan(s_slice)]
            valid_r2 = r2_slice[~np.isnan(r2_slice)]
            valid_vol = vol_slice[~np.isnan(vol_slice)]

            if len(valid_s) > 0:
                slope_by_scale[scale] = float(np.mean(valid_s))
                all_slopes.extend(valid_s)
            if len(valid_r2) > 0:
                all_r2.extend(valid_r2)
            if len(valid_vol) > 0:
                all_vol.extend(valid_vol)

        seg = Segment(
            start_idx=offset + start,
            end_idx=offset + end,
            start_date=str(dates[start]) if dates is not None and start < len(dates) else '',
            end_date=str(dates[end - 1]) if dates is not None and end - 1 < len(dates) else '',
            duration=duration,
            regime=regime,
            avg_slope=float(np.mean(all_slopes)) if all_slopes else 0.0,
            avg_r_squared=float(np.mean(all_r2)) if all_r2 else 0.0,
            avg_volatility=float(np.mean(all_vol)) if all_vol else 0.0,
            slope_by_scale=slope_by_scale,
        )
        segments.append(seg)

    return segments


class StockPersonalityEngine:
    """
    股性分析引擎 — 预计算全量特征+regime，支持点时间查询。

    用法:
        engine = StockPersonalityEngine(close, dates)
        # 回测中，在bar_idx时刻查询过去400天的股性:
        segments = engine.get_segments_at_bar(bar_idx, window=400)
        personality = engine.get_personality_at_bar(bar_idx, window=200)
    """

    def __init__(self, close: np.ndarray, dates: np.ndarray = None):
        self.close = close
        self.dates = dates
        self.N = len(close)

        # 一次性计算全量多尺度特征（每个bar只用过去数据，无未来函数）
        self.features = multi_scale_features(close)

        # 一次性计算全量regime（每个bar只用该bar的特征，无未来函数）
        self.regimes = detect_regimes(self.features, self.N)

        # 一次性计算全量stage（含min_duration过滤，更稳定）
        self.stages, self.stage_slopes = self._precompute_stages()

    def _precompute_stages(self):
        """
        预计算所有bar的stage，含min_duration过滤。
        阶段比regime更稳定：使用中长期尺度(40/60/120d)平滑slope，
        并要求每个阶段至少持续STAGE_MIN_DURATION天。
        """
        from .classifier import detect_current_stage, STAGE_MIN_DURATION

        stages = np.array(['uncertain'] * self.N, dtype=object)
        slopes = np.zeros(self.N)

        # Step 1: 计算每bar的raw stage
        raw_stages = []
        raw_slopes = []
        for i in range(self.N):
            pp_start = max(0, i - 119)
            recent = self.close[pp_start:i + 1]
            high, low = np.max(recent), np.min(recent)
            pp = (self.close[i] - low) / (high - low) if high > low else 0.5
            stage, slope = detect_current_stage(self.features, i, pp)
            raw_stages.append(stage)
            raw_slopes.append(slope)

        # Step 2: 应用min_duration过滤
        current_stage = raw_stages[0]
        current_start = 0

        for i in range(self.N):
            if raw_stages[i] != current_stage:
                duration = i - current_start
                if duration >= STAGE_MIN_DURATION:
                    # 当前阶段已持续足够久，接受新阶段
                    current_stage = raw_stages[i]
                    current_start = i
                # else: 保持当前阶段，忽略短暂偏离

            stages[i] = current_stage
            slopes[i] = raw_slopes[i]

        return stages, slopes

    def get_segments_at_bar(self, bar_idx: int, window: int = 200) -> List[Segment]:
        """
        获取bar_idx时刻、回看window天的分段结果。
        只使用[bar_idx-window+1, bar_idx]范围的数据，无未来函数。
        """
        if bar_idx < 0 or bar_idx >= self.N:
            return []

        start = max(0, bar_idx - window + 1)
        end = bar_idx + 1  # inclusive

        regime_slice = self.regimes[start:end]
        date_slice = self.dates[start:end] if self.dates is not None else None

        return merge_segments(regime_slice, self.features, date_slice, offset=start)

    def get_personality_at_bar(self, bar_idx: int) -> str:
        """
        获取bar_idx时刻的股性分类。
        使用全量可用历史（到bar_idx为止），数据越多越稳定。
        """
        from .classifier import classify_personality

        full_lookback = bar_idx + 1
        long_term_segments = self.get_segments_at_bar(bar_idx, full_lookback)

        # 多窗口分段作为回退
        segments_by_window = {}
        for w in [90, 200, 400]:
            segments_by_window[w] = self.get_segments_at_bar(bar_idx, w)

        lookback_start = max(0, bar_idx - full_lookback + 1)
        net_return = float(self.close[bar_idx] / self.close[lookback_start] - 1) \
            if self.close[lookback_start] > 0 else 0.0

        return classify_personality(
            segments_by_window,
            long_term_segments=long_term_segments,
            net_return=net_return,
        )

    def get_stage_at_bar(self, bar_idx: int) -> str:
        """获取bar_idx时刻的当前阶段（预计算+min_duration过滤）。"""
        if 0 <= bar_idx < self.N:
            return str(self.stages[bar_idx])
        return 'uncertain'

    def get_stage_slope_at_bar(self, bar_idx: int) -> float:
        """获取bar_idx时刻的stage slope值（用于描述趋势强度）。"""
        if 0 <= bar_idx < self.N:
            return float(self.stage_slopes[bar_idx])
        return 0.0

    def get_full_analysis_at_bar(self, bar_idx: int) -> Dict:
        """
        获取bar_idx时刻的完整分析（多窗口分段+股性+阶段）。
        所有数据都是点时间(point-in-time)，无未来函数。

        股性: 基于全量可用历史（到bar_idx为止）分类，确保长期稳定
        阶段: 基于短期窗口(90d)检测，反映近期变化
        """
        windows = [90, 200, 400]
        segments_by_window = {}
        for w in windows:
            segments_by_window[w] = self.get_segments_at_bar(bar_idx, w)

        # 全量历史分段：用于股性分类（长期稳定）
        full_lookback = bar_idx + 1  # 使用所有可用数据
        long_term_segments = self.get_segments_at_bar(bar_idx, full_lookback)

        from .classifier import classify_personality, detect_current_stage

        # 计算全量历史的净收益率（用于区分震荡股/均值回归股/趋势股）
        lookback_start = max(0, bar_idx - full_lookback + 1)
        net_return = float(self.close[bar_idx] / self.close[lookback_start] - 1) \
            if self.close[lookback_start] > 0 else 0.0

        personality = classify_personality(
            segments_by_window,
            long_term_segments=long_term_segments,
            net_return=net_return,
        )

        pp_start = max(0, bar_idx - 119)
        recent = self.close[pp_start:bar_idx + 1]
        high = np.max(recent)
        low = np.min(recent)
        price_pos = (self.close[bar_idx] - low) / (high - low) if high > low else 0.5

        stage = self.get_stage_at_bar(bar_idx)
        stage_slope = self.get_stage_slope_at_bar(bar_idx)

        return {
            'bar_idx': bar_idx,
            'date': str(self.dates[bar_idx]) if self.dates is not None else '',
            'personality': personality,
            'stage': stage,
            'stage_slope': stage_slope,
            'price_position': price_pos,
            'segments_by_window': segments_by_window,
            'long_term_segments': long_term_segments,
        }


def analyze_stock(close: np.ndarray, dates: np.ndarray = None,
                  windows: List[int] = None) -> Dict[int, List[Segment]]:
    """
    便捷函数：分析股票最新时刻的多窗口分段。
    等价于 engine.get_segments_at_bar(N-1, window) for each window。
    """
    if windows is None:
        windows = [90, 200, 400]

    engine = StockPersonalityEngine(close, dates)
    results = {}
    for w in windows:
        results[w] = engine.get_segments_at_bar(len(close) - 1, w)
    return results
