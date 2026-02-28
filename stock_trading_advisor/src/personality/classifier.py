"""
股性分类 + 当前阶段检测

股性基于全量历史数据分类（长期稳定特征），方向无关。
阶段基于短期窗口检测（近期变化状态），包含方向信息。

6种方向无关股性类型 × 7种阶段。

设计原则：
- 股性描述的是股票的行为模式（HOW it moves），不涉及方向
- 方向（涨/跌）由"阶段"描述
- 数据不足（<90天）→ 返回 'insufficient'，使用默认参数
- 分类可能随数据积累而修正（90天→400天），这是预期行为
"""

import numpy as np
from typing import List, Dict, Optional
from .segmenter import Segment, SLOPE_MODERATE, R2_LOW, SCALE_WEIGHTS


# 最少需要的数据量（交易日），低于此值不做股性判断
MIN_BARS_FOR_CLASSIFICATION = 90

# 6种方向无关的股性类型
PERSONALITY_TYPES = [
    'trend_persistent',    # 趋势持续型（某方向长期主导，高R²，不限涨跌）
    'staircase_mover',     # 阶梯型（交替趋势+横盘，不限方向）
    'mean_reverter',       # 均值回归/箱体震荡（低振幅围绕均值波动）
    'volatile_oscillator', # 暴涨暴跌（频繁方向切换，高振幅）
    'breakout_runner',     # 突破跑动（长横盘后爆发，不限方向）
    'erratic',             # 无规律
]

# 当前阶段（5种 + uncertain）
# 基于40/60/120日多尺度特征，slope决定趋势方向，pp决定震荡位置
# slope强度（急/温和）作为metadata附带，不增加阶段种类以减少翻转
STAGE_TYPES = [
    'uptrend',             # 上涨（包含急涨和温和上涨）
    'downtrend',           # 下跌（包含急跌和温和下跌）
    'high_consolidation',  # 高位震荡（横盘 + pp > 0.65）
    'low_consolidation',   # 低位震荡/筑底（横盘 + pp < 0.35）
    'consolidation',       # 横盘整理
    'uncertain',           # 不确定
]

# 阶段检测参数
STAGE_SLOPE_FLAT = 0.10     # 年化|slope| < 10% 视为横盘
STAGE_SCALE_WEIGHTS = {40: 0.5, 60: 1.0, 120: 1.0}  # 侧重中长期
STAGE_SMOOTH_BARS = 20       # 平滑窗口（天）
STAGE_MIN_DURATION = 20      # 最少持续天数（防翻转）


def _is_up(regime: str) -> bool:
    return regime in ('strong_up', 'moderate_up', 'weak_up', 'breakout_up', 'reversal_up')


def _is_down(regime: str) -> bool:
    return regime in ('strong_down', 'moderate_down', 'weak_down', 'reversal_down')


def _is_sideways(regime: str) -> bool:
    return regime == 'sideways'


def classify_personality(segments_by_window: Dict[int, List[Segment]],
                         long_term_segments: List[Segment] = None,
                         hurst: float = 0.5,
                         net_return: float = None,
                         debug: bool = False) -> str:
    """
    基于长期分段结果分类股性（方向无关）。

    优先使用 long_term_segments（全量历史），回退到最长窗口。
    所有分类规则使用方向无关指标（abs、max/min对称）。

    Returns:
        股性类型字符串，或 'insufficient' 表示数据不足。
    """
    # 选择分段数据：优先全量历史，回退到最长可用窗口
    if long_term_segments and len(long_term_segments) > 0:
        segs = long_term_segments
    else:
        for w in sorted(segments_by_window.keys(), reverse=True):
            segs = segments_by_window.get(w, [])
            if segs:
                break
        else:
            return 'insufficient'

    if not segs:
        return 'insufficient'

    # ---- 基础统计 ----
    total_bars = sum(s.duration for s in segs)
    if total_bars < MIN_BARS_FOR_CLASSIFICATION:
        return 'insufficient'

    n_segments = len(segs)
    avg_segment_dur = total_bars / n_segments

    # 方向统计
    up_bars = sum(s.duration for s in segs if _is_up(s.regime))
    down_bars = sum(s.duration for s in segs if _is_down(s.regime))
    sideways_bars = sum(s.duration for s in segs if _is_sideways(s.regime))

    up_ratio = up_bars / total_bars
    down_ratio = down_bars / total_bars
    sideways_ratio = sideways_bars / total_bars

    # ---- 方向无关核心指标 ----
    # 方向偏向度：一个方向主导的程度（0=完全对称，1=单方向100%）
    direction_bias = abs(up_ratio - down_ratio)
    # 主导方向占比（不区分是涨还是跌）
    dominant_ratio = max(up_ratio, down_ratio)

    # ---- 趋势质量指标（按时长加权）----
    weighted_r2 = sum(s.avg_r_squared * s.duration for s in segs) / total_bars
    weighted_abs_slope = sum(abs(s.avg_slope) * s.duration for s in segs) / total_bars

    # 清晰趋势时间占比（slope超过阈值且R²尚可）
    trending_bars = sum(s.duration for s in segs
                        if abs(s.avg_slope) > SLOPE_MODERATE and s.avg_r_squared > R2_LOW)
    trending_ratio = trending_bars / total_bars

    # ---- 方向切换频率 ----
    dir_changes = 0
    for i in range(1, len(segs)):
        prev_up = _is_up(segs[i - 1].regime)
        prev_down = _is_down(segs[i - 1].regime)
        curr_up = _is_up(segs[i].regime)
        curr_down = _is_down(segs[i].regime)
        if (prev_up and curr_down) or (prev_down and curr_up):
            dir_changes += 1

    dc_per_200 = dir_changes / (total_bars / 200) if total_bars > 0 else 0

    # ---- 模式检测（方向无关）----
    # breakout模式：长横盘后接强方向性移动（上或下均可）
    has_breakout = any(
        _is_sideways(segs[i].regime) and segs[i].duration >= 30
        and (_is_up(segs[i + 1].regime) or _is_down(segs[i + 1].regime))
        and segs[i + 1].avg_r_squared > R2_LOW
        for i in range(len(segs) - 1)
    )

    # 阶梯模式：交替趋势+横盘（任一方向均可）
    staircase_count = sum(
        1 for i in range(len(segs) - 1)
        if (_is_sideways(segs[i].regime) and not _is_sideways(segs[i + 1].regime)) or
           (not _is_sideways(segs[i].regime) and _is_sideways(segs[i + 1].regime))
    )

    if debug:
        nr_str = f", net_ret={net_return:.1%}" if net_return is not None else ""
        print(f"    [DEBUG] n_segs={n_segments}, total_bars={total_bars}, avg_dur={avg_segment_dur:.0f}")
        print(f"    [DEBUG] up={up_ratio:.2f}, down={down_ratio:.2f}, sw={sideways_ratio:.2f}")
        print(f"    [DEBUG] dir_bias={direction_bias:.2f}, dominant={dominant_ratio:.2f}")
        print(f"    [DEBUG] trend_ratio={trending_ratio:.2f}, w_r2={weighted_r2:.2f}, w_|slope|={weighted_abs_slope:.2f}{nr_str}")
        print(f"    [DEBUG] dc={dir_changes}, dc/200={dc_per_200:.2f}, staircase_count={staircase_count}")
        print(f"    [DEBUG] breakout={has_breakout}")

    # ---- 预计算辅助指标 ----
    # 趋势段的平均时长（用于区分阶梯型 vs 暴涨暴跌）
    up_durs = [s.duration for s in segs if _is_up(s.regime)]
    down_durs = [s.duration for s in segs if _is_down(s.regime)]
    avg_up_dur = np.mean(up_durs) if up_durs else 0
    avg_down_dur = np.mean(down_durs) if down_durs else 0

    # ====== 层次化分类（两级：大类 → 子类）======
    #
    # 设计思路：
    #   先识别"结构特征"（横盘型/趋势型/震荡型），再细分子类型。
    #   横盘型（突破跑动）是独立模式，需在趋势判断之前检测。
    #
    # 大类判断逻辑：
    #   1. 突破型（breakout_runner）：以横盘为主，偶有方向性突破
    #   2. 趋势型（trend_persistent / staircase_mover）：有明确方向偏向，切换较少
    #   3. 震荡型（mean_reverter / volatile_oscillator）：无明确方向，频繁来回

    # ---- 预计算 ----
    abs_net_return = abs(net_return) if net_return is not None else 0.0

    # ======== 第一步：突破型检测（不受趋势/震荡分类限制）========
    # 突破型 = 有明确"长横盘→爆发"事件 + 横盘占比不低
    # 典型：银行/稳健成长股，长期横盘，偶有突破行情
    # 必须有 has_breakout=True，避免把"恰好横盘多"的趋势股误归入
    if has_breakout and sideways_ratio >= 0.20:
        return 'breakout_runner'
    if has_breakout and sideways_ratio >= 0.10 and dc_per_200 < 1.2:
        # 极低换向频率 + 有突破事件 + 一定横盘 → 经典突破型（银行类）
        return 'breakout_runner'

    # ======== 第二步：趋势型判断 ========
    # 趋势型两个路径（任一满足即为趋势型）：
    #   路径A（收益驱动）：净收益显著（> 50%）且换向不太频繁（dc < 2.0）
    #     捕捉上涨斜率>>下跌斜率、时间对半但净收益巨大的长牛股
    #   路径B（时间驱动）：时间上一方向明显主导 + 换向少
    #     捕捉稳定趋势股，方向偏向在时间上体现
    is_trending = (
        dc_per_200 < 2.0
        and (
            abs_net_return > 0.50                                   # 路径A: 净收益 > 50%
            or (direction_bias > 0.25 and dominant_ratio > 0.55)    # 路径B: 时间偏向强
        )
    )

    if is_trending:
        # ---- 趋势型子分类 ----

        # 斜坡型（严格趋势）：连续高R²、平均段长大、切换极少，几乎一条直线
        if (trending_ratio > 0.55 and weighted_r2 > 0.42 and avg_segment_dur > 60
                and direction_bias > 0.30 and dc_per_200 < 1.5):
            return 'trend_persistent'

        # 阶梯型：路径A —— 有横盘+阶梯计数（横盘与趋势交替）
        if staircase_count >= 3 and sideways_ratio >= 0.05:
            return 'staircase_mover'
        # 阶梯型：路径B —— 涨跌段时长不对称 + 方向未极度偏向
        #   约束 direction_bias < 0.45：极度偏向（如净跌90%的股）应归为trend_persistent，
        #   而非staircase（否则崩溃股因"下跌段远长于上涨段"被错误识别为阶梯型）
        if (avg_up_dur > 0 and avg_down_dur > 0
                and max(avg_up_dur, avg_down_dur) > min(avg_up_dur, avg_down_dur) * 2.5
                and direction_bias < 0.40):
            return 'staircase_mover'

        # 趋势持续型（宽松）：方向偏向较明显，不满足严格斜坡条件
        return 'trend_persistent'

    else:
        # ======== 第三步：震荡型子分类 ========

        # 均值回归型：振幅低，涨跌均衡，整体净收益小
        #   适合：区间做T，不追趋势，止盈快
        if (weighted_abs_slope < 0.45 and up_ratio > 0.15 and down_ratio > 0.15):
            return 'mean_reverter'

        # 高振幅震荡型：涨跌都显著，方向切换频繁
        #   适合：快进快出，不长持，禁EH
        if up_ratio > 0.20 and down_ratio > 0.20:
            return 'volatile_oscillator'

        return 'erratic'


def detect_current_stage(features: Dict[int, Dict[str, np.ndarray]],
                         bar_idx: int,
                         price_position: float = 0.5) -> tuple:
    """
    检测单bar的阶段（无min_duration过滤）。

    返回: (stage_str, slope_value)
        stage: uptrend / downtrend / high_consolidation / low_consolidation /
               consolidation / uncertain
        slope: 年化slope值（用于描述强度：急涨/温和等）
    """
    smooth = STAGE_SMOOTH_BARS
    start = max(0, bar_idx - smooth + 1)
    end = bar_idx + 1

    weighted_slope = 0.0
    total_w = 0.0

    for scale, w in STAGE_SCALE_WEIGHTS.items():
        if scale not in features:
            continue
        s_slice = features[scale]['slope'][start:end]
        valid_s = s_slice[~np.isnan(s_slice)]
        if len(valid_s) > 0:
            weighted_slope += float(np.mean(valid_s)) * w
            total_w += w

    if total_w == 0:
        return 'uncertain', 0.0

    avg_slope = weighted_slope / total_w

    if abs(avg_slope) < STAGE_SLOPE_FLAT:
        if price_position > 0.65:
            return 'high_consolidation', avg_slope
        elif price_position < 0.35:
            return 'low_consolidation', avg_slope
        return 'consolidation', avg_slope

    if avg_slope > 0:
        return 'uptrend', avg_slope
    return 'downtrend', avg_slope


def describe_segments(segments: List[Segment]) -> str:
    """生成分段的人类可读描述。"""
    if not segments:
        return "(no segments)"
    parts = []
    for seg in segments:
        slope_pct = seg.avg_slope * 100  # 转为百分比/年
        slope_str = f"{slope_pct:+.0f}%/yr"
        r2_str = f"R²={seg.avg_r_squared:.2f}"
        parts.append(f"[{seg.regime}({seg.duration}d,{slope_str},{r2_str})]")
    return " → ".join(parts)
