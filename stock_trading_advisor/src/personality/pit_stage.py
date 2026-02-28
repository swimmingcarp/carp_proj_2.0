"""
PIT（Point-in-Time）多周期状态检测器 v7

核心设计原则：
  1. 严格PIT：每个bar只用 [0..bar_idx] 的历史数据
  2. 大/中周期双重确认才能触发状态切换（大周期组+中周期组）
  3. 假设法：consolidation是默认状态
  4. 股性自适应确认天数（基于60天年化波动率）
  5. 步长采样（v6）：大窗口用粗粒度采样过滤日内噪声
  6. 可配置参数（v7）：PitConfig dataclass，支持系统化sweep
  7. 增量EMA（v7）：O(1)每bar，可选EMA对齐信号
  8. 成交量因子（v7）：可选成交量确认信号

默认参数行为与v6完全一致。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────
# PitConfig：所有可调参数集中于此
# ─────────────────────────────────────────────────────────────────

@dataclass
class PitConfig:
    # ── 窗口分组 ─────────────────────────────────────────────────
    # 扫描最优：W3_sm5（加入5天超短窗口）
    scales_large:  tuple = (80, 120)    # 大周期组：决定主格局
    scales_medium: tuple = (40, 60)     # 中周期组：趋势确认
    scales_small:  tuple = (5, 10, 20)  # 小周期组：早期预警（含5天超短）

    # ── 步长映射（窗口→采样步长） ────────────────────────────────
    # 扫描最优：S2_mild（60:2, 80:3, 120:4 温和步长）
    strides: dict = field(
        default_factory=lambda: {5:1, 10:1, 20:1, 40:1, 60:2, 80:3, 120:4}
    )

    # ── 斜率阈值（年化，单位≈ ln收益/年） ────────────────────────
    # 扫描最优：T2_hard（更硬门槛，防止弱信号触发状态切换）
    slope_strong: float = 0.28   # 强趋势阈值（原0.22）
    slope_weak:   float = 0.12   # 弱趋势阈值（原0.09）
    slope_flat:   float = 0.05   # 接近平坦阈值（原0.04）

    # ── 确认天数（基准） ─────────────────────────────────────────
    # 扫描最优：GC2_fast（更快确认，硬阈值过滤弱信号后快速响应）
    confirm_enter: int = 6     # consolidation → trend（原8）
    confirm_exit:  int = 4     # trend → consolidation（原6）
    confirm_rev:   int = 10    # 反转（原12）

    # ── 状态最短持续天数（grace period） ─────────────────────────
    # 扫描最优：GC2_fast（较短grace，但由硬阈值保证质量）
    grace_trend:  int = 12     # 趋势状态最少12天（原15）
    grace_consol: int = 6      # 整理状态最少6天（原8）

    # ── 回撤/反弹预警阈值 ────────────────────────────────────────
    drawdown_exit: float = 0.18   # 从60天高点回落>18% → 退出信号

    # ── S/R区间参数 ───────────────────────────────────────────────
    range_high_pct:  float = 85.0
    range_low_pct:   float = 15.0
    breakout_large:  float = 0.05
    breakout_medium: float = 0.03

    # ── 有意义移动量下限 ─────────────────────────────────────────
    min_meaningful_move: float = 0.04   # 窗口内移动 < 4% 则忽略

    # ── 波动率自适应分档 ─────────────────────────────────────────
    vol_high: float = 0.50
    vol_mid:  float = 0.32
    vol_low:  float = 0.20

    # ── 附加因子开关 ─────────────────────────────────────────────
    use_ema:    bool = True    # EMA20/60/120对齐信号
    use_volume: bool = False   # 成交量确认信号

    # EMA因子参数
    ema_confirm_enter: bool = True   # EMA对齐用于加强进入条件
    ema_allow_exit:    bool = True   # EMA背离用于提前退出

    # 成交量因子参数
    vol_surge_ratio:   float = 1.5   # 成交量/MA20 > 此值 = 放量
    vol_window:        int   = 20    # 成交量均线窗口

    # ── 斜率速度（Slope Velocity，解决滞后性）─────────────────────
    # slope_velocity = slope[t] - slope[t - vel_step]
    # 斜率还是负的但正在快速改善 → 早于斜率变号 20-30 天发出信号
    use_vel:       bool  = True    # 开启斜率速度信号
    vel_step:      int   = 5       # 比较几天前的斜率（默认5天）
    vel_threshold: float = 0.08    # 斜率改善速率阈值（年化，≈8%/yr per vel_step days）

    # ── 平滑度自适应大周期门槛 ──────────────────────────────────────
    # R² 衡量最近 120 天价格的线性拟合质量（0=杂乱，1=完美趋势）
    # 缓慢趋势型（R² > smooth_thresh）→ 要求大+中双确认（防假信号）
    # 急涨急跌型（R² < rough_thresh） → 只用中周期，大周期只作参考
    use_smoothness:  bool  = True   # 开启平滑度自适应
    smooth_thresh:   float = 0.50   # R² > 0.50 视为平滑趋势
    rough_thresh:    float = 0.25   # R² < 0.25 视为急涨急跌

    # ── 长周期价格区间（High/Low Level Detection）────────────────────
    # 斜率只告诉你"方向"，价格区间告诉你"高低位"。
    # 结合两者才能区分：
    #   - 斜率=+15%/yr + 处于2年高位 → 可能是顶部，谨慎做多
    #   - 斜率=+15%/yr + 处于2年低位 → 更可能是真反弹，放心做多
    use_long_range:    bool  = True   # 开启长周期价格位置信号
    long_range_window: int   = 250    # 主要长周期窗口（约1年/250交易日）
    long_range_low:    float = 0.20   # pp < 0.20 = 低位区间
    long_range_high:   float = 0.80   # pp > 0.80 = 高位区间

    # ── 股性标签（由外部 segmenter → classifier 传入）────────────────
    # 取值: '' / 'trend_persistent' / 'volatile_oscillator' /
    #       'mean_reverter' / 'staircase_mover' / 'breakout_runner'
    # 空字符串 = 不区分股性，使用通用逻辑（向后兼容）
    personality: str = ''


# 默认配置（全局，保持向后兼容）
DEFAULT_CFG = PitConfig()


# ─────────────────────────────────────────────────────────────────
# 纯函数：斜率计算
# ─────────────────────────────────────────────────────────────────

def _log_slope_annualized(prices: np.ndarray) -> float:
    """步长1（每日）时的斜率估计，X轴单位=交易日。"""
    n = len(prices)
    if n < 5:
        return 0.0
    lp = np.log(np.maximum(prices, 1e-8))
    x  = np.arange(n, dtype=float)
    xm = x - x.mean()
    ym = lp - lp.mean()
    d  = np.dot(xm, xm)
    if d < 1e-10:
        return 0.0
    return np.dot(xm, ym) / d * 252


def _log_slope_and_r2(prices: np.ndarray, stride: int = 1):
    """
    同时返回 (annualized_slope, R²)。
    R² 衡量线性拟合质量：1=完美线性趋势，0=完全随机。
    用于判断股票平滑度：高R²=缓慢趋势，低R²=急涨急跌。
    """
    n = len(prices)
    if n < 5:
        return 0.0, 0.0
    lp = np.log(np.maximum(prices, 1e-8))
    x  = np.arange(n, dtype=float) * stride
    xm = x - x.mean()
    ym = lp - lp.mean()
    ss_xx = np.dot(xm, xm)
    ss_xy = np.dot(xm, ym)
    ss_yy = np.dot(ym, ym)
    if ss_xx < 1e-10:
        return 0.0, 0.0
    slope = ss_xy / ss_xx * 252
    r2    = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-10 else 0.0
    return slope, float(np.clip(r2, 0.0, 1.0))


def _log_slope_strided(prices: np.ndarray, stride: int) -> float:
    """
    带步长的斜率估计。

    X轴使用实际交易日数（idx × stride），保证斜率单位与步长1可比：
      slope = Δlog_price / Δday × 252 = 年化对数收益率斜率

    例：120天窗口步长5 → prices有≈24个点，x=[0,5,10,...,120]
    """
    n = len(prices)
    if n < 3:
        return 0.0
    lp = np.log(np.maximum(prices, 1e-8))
    x  = np.arange(n, dtype=float) * stride   # 实际天数轴
    xm = x - x.mean()
    ym = lp - lp.mean()
    d  = np.dot(xm, xm)
    if d < 1e-10:
        return 0.0
    return np.dot(xm, ym) / d * 252


# ─────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────

def _get_range(high: np.ndarray, low: np.ndarray,
               end_idx: int, window: int,
               cfg: PitConfig = DEFAULT_CFG) -> Tuple[float, float]:
    s  = max(0, end_idx - window + 1)
    sh = high[s: end_idx + 1]
    sl = low[s: end_idx + 1]
    if len(sh) < 5:
        return float(sl.min()), float(sh.max())
    return (float(np.percentile(sl, cfg.range_low_pct)),
            float(np.percentile(sh, cfg.range_high_pct)))


def _estimate_vol(close: np.ndarray, bar_idx: int) -> float:
    medium_w = 60
    s = max(1, bar_idx - medium_w + 1)
    e = bar_idx + 1
    if e - s < 5:
        return 0.35
    r = np.diff(np.log(np.maximum(close[s - 1: e], 1e-8)))
    return float(np.std(r) * np.sqrt(252))


def _group_slope(close: np.ndarray, bar_idx: int,
                 scales, cfg: PitConfig) -> float:
    """
    计算一组窗口的平均斜率（有意义的窗口才纳入，等权平均）。

    步长策略：窗口越大步长越粗（见cfg.strides），
    X轴始终用实际交易日数，保证不同步长的斜率可直接比较。
    """
    end = bar_idx + 1
    slopes = []
    for w in scales:
        stride   = cfg.strides.get(w, 1)
        actual_w = min(w, end)
        if actual_w < 5:
            continue
        prices = close[end - actual_w: end: stride]
        if len(prices) < 3:
            continue
        move = abs(prices[-1] / prices[0] - 1)
        if move >= cfg.min_meaningful_move:
            slopes.append(_log_slope_strided(prices, stride))
    return float(np.mean(slopes)) if slopes else 0.0


# ─────────────────────────────────────────────────────────────────
# 信号计算
# ─────────────────────────────────────────────────────────────────

def _compute_data(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  bar_idx: int, cfg: PitConfig,
                  ema_state: Optional[Dict] = None,
                  vol_ma20: Optional[float] = None,
                  volume:   Optional[float] = None) -> Dict:
    """
    计算bar_idx处的全部多尺度信号数据（PIT）。

    ema_state: 由PITStateDetector增量维护的EMA字典（含ema20/ema60/ema120）
    vol_ma20:  成交量20日均值（PITStateDetector增量维护）
    volume:    当前bar成交量（若cfg.use_volume=True则必须传入）
    """
    end = bar_idx + 1
    cp  = close[bar_idx]

    large_w  = max(cfg.scales_large)   # 120
    medium_w = max(cfg.scales_medium)  # 60

    # ── 三组斜率 ──────────────────────────────────────────────────
    s_large  = _group_slope(close, bar_idx, cfg.scales_large,  cfg)
    s_medium = _group_slope(close, bar_idx, cfg.scales_medium, cfg)
    s_small  = _group_slope(close, bar_idx, cfg.scales_small,  cfg)

    # 单尺度别名（60/120天用步长采样）
    sl = _log_slope_strided(close[max(0, end - 20): end],  1)
    sm = _log_slope_strided(close[max(0, end - min(medium_w, end)): end], 3)
    sx = _log_slope_strided(close[max(0, end - min(large_w,  end)): end], 5)

    # ── 扩展斜率：22日（月线级短期方向）+ 200日（超大周期趋势）──────
    # 用户建议：22/30日填补短-中周期空白，200日作大周期方向参考
    _w22  = min(22, bar_idx + 1)
    s_22d = _log_slope_annualized(close[max(0, end - _w22): end]) if _w22 >= 5 else 0.0
    _w200 = min(200, bar_idx + 1)
    _p200 = close[max(0, end - _w200): end: 5]   # 步长5采样，与120d保持一致
    s_200d = _log_slope_strided(_p200, 5) if len(_p200) >= 3 else 0.0

    # ── 反弹失败关键节点评分（Key Node: Failed Rally Detection）────────
    # 用户洞察（2023-07-28 案例）：
    #   "7月28日大阳线后面几天没有延续反转，8月11日一根大阴线把反转的希望浇灭了"
    #   "这几天的影响不能被大周期平均稀释，要加大权重"
    #
    # 评分逻辑：
    #   对 10/15/20 天窗口，分别计算：
    #     rally_gain   = 从窗口起点到窗口最高点的涨幅（> 3% = 有过真实反弹）
    #     retrace_pct  = (最高点 - 当前价) / (最高点 - 窗口起点) — 本次涨幅已回吐的比例
    #                    注意：这不是"从高点跌了多少%"，而是"反弹幅度被吃掉了多少"
    #   score = max(rally_gain × retrace_pct) 三个窗口取最大
    #   score > 0.050（≈涨6%回吐83%，或涨10%回吐50%）→ 确认反弹失败关键节点
    #
    # 应用：在状态惯性（consolidation→downtrend，且cfg.personality已知）中：
    #   当确认反弹失败 + s_small < slope_weak（短期动量减弱）
    #   → 比普通快速路径更早返回 downtrend
    #   000001案例：8月11日score=0.066触发 → 8月17日完成确认（vs原8月23日）
    _failed_rally_score = 0.0
    for _kw in (10, 15, 20):
        _ke = bar_idx + 1
        _ks = max(0, _ke - _kw)
        if _ke - _ks >= 5:
            _kh  = float(high[_ks: _ke].max())
            _kc0 = float(close[_ks])
            if _kc0 > 0 and _kh > _kc0:
                _kg = (_kh - _kc0) / _kc0                           # 涨幅（起点→最高点）
                _kr = (_kh - cp)   / (_kh - _kc0) if _kh > _kc0 else 0.0  # 回吐比例（相对于本次涨幅）
                # 回吐比例：> 1.0 = 连起点都跌穿了（完全回撤+超跌）
                _kr = float(np.clip(_kr, 0.0, 1.5))
                if _kg > 0.03 and _kr > 0.50:
                    _failed_rally_score = max(_failed_rally_score, _kg * _kr)

    # ── S/R区间 ───────────────────────────────────────────────────
    lg_s, lg_r = _get_range(high, low, bar_idx, min(large_w,  bar_idx + 1), cfg)
    md_s, md_r = _get_range(high, low, bar_idx, min(medium_w, bar_idx + 1), cfg)

    # ── 价格位置 ──────────────────────────────────────────────────
    above_lg = cp > lg_r * (1 + cfg.breakout_large)
    below_lg = cp < lg_s * (1 - cfg.breakout_large)
    above_md = cp > md_r * (1 + cfg.breakout_medium)
    below_md = cp < md_s * (1 - cfg.breakout_medium)
    in_range = not above_md and not below_md

    # ── 近60天高/低点与回撤 ───────────────────────────────────────
    win60       = min(medium_w, bar_idx + 1)
    peak_60     = float(high[max(0, end - win60): end].max())
    trough_60   = float(low[max(0, end - win60): end].min())
    drawdown_60 = (peak_60 - cp) / peak_60    if peak_60  > 0 else 0.0
    recovery_60 = (cp - trough_60) / trough_60 if trough_60 > 0 else 0.0

    # ── 区间内位置（上/下半） ─────────────────────────────────────
    md_mid        = (md_s + md_r) / 2.0
    in_upper_half = cp >= md_mid
    in_lower_half = cp <= md_mid

    # ── 综合趋势判断 ──────────────────────────────────────────────
    # 进入趋势：大+中周期双重确认 + 价格位置一致
    enter_up   = (s_large > cfg.slope_weak  and s_medium > cfg.slope_weak
                  and (in_upper_half or above_md))
    enter_down = (s_large < -cfg.slope_weak and s_medium < -cfg.slope_weak
                  and (in_lower_half or below_md))

    # 强信号：单中期极强（处理大底/顶部急速反转）
    strong_medium_up   = s_medium > 3 * cfg.slope_strong and (in_upper_half or above_md)
    strong_medium_down = s_medium < -3 * cfg.slope_strong and (in_lower_half or below_md)

    # 小周期预警（仅用于退出，不用于进入）
    small_up   = s_small > cfg.slope_weak
    small_down = s_small < -cfg.slope_weak

    # ── 平滑度（R²）估计 ─────────────────────────────────────────
    # 用 120 天和 60 天回归 R² 的较大值，衡量"是否缓慢线性趋势"
    lw_prices = close[max(0, end - min(large_w,  end)): end: 5]   # 120天，步长5
    mw_prices = close[max(0, end - min(medium_w, end)): end: 2]   # 60天，步长2
    _, r2_large  = _log_slope_and_r2(lw_prices, 5)
    _, r2_medium = _log_slope_and_r2(mw_prices, 2)
    # 平滑度：用两者中较大值（任一周期有清晰趋势即视为平滑）
    smoothness = float(max(r2_large, r2_medium))

    # 根据平滑度决定是否需要大周期确认
    # 缓慢趋势（smooth）：保留大+中双确认 → 精确但有滞后
    # 急涨急跌（rough）：只用中周期 → 响应更快，接受更多噪声
    is_smooth = smoothness >= cfg.smooth_thresh if cfg.use_smoothness else True
    is_rough  = smoothness <= cfg.rough_thresh  if cfg.use_smoothness else False

    # ── 长周期价格区间位置 ───────────────────────────────────────────
    # 斜率给方向，价格区间位置给高低位。两者结合才有完整上下文。
    # 例：斜率+15% + 处于1年低位 = 真反弹；斜率+15% + 处于1年高位 = 可能是顶部
    if cfg.use_long_range:
        lrw = min(cfg.long_range_window, bar_idx + 1)   # 长周期窗口（默认250天）
        lrw2 = min(cfg.long_range_window * 2, bar_idx + 1)  # 两倍窗口（500天）
        lr_high  = float(high[max(0, end - lrw ): end].max())
        lr_low   = float(low [max(0, end - lrw ): end].min())
        lr2_high = float(high[max(0, end - lrw2): end].max())
        lr2_low  = float(low [max(0, end - lrw2): end].min())
        # 价格在1年区间内的相对位置 [0=1年低点, 1=1年高点]
        pp_long  = (cp - lr_low)  / (lr_high  - lr_low)  if lr_high  > lr_low  else 0.5
        pp_long2 = (cp - lr2_low) / (lr2_high - lr2_low) if lr2_high > lr2_low else 0.5
        # 创1年新高/新低
        at_1y_high = cp >= lr_high  * 0.99   # 接近或突破1年高点（≥99%）
        at_1y_low  = cp <= lr_low   * 1.01   # 接近或跌破1年低点（≤101%）
    else:
        pp_long = pp_long2 = 0.5
        at_1y_high = at_1y_low = False
        lr_high = lr_low = cp

    # ── 多尺度起涨点 + 价格区间位置 ─────────────────────────────────────
    # 核心思路（用户要求）：
    #   - 斜率告诉你"方向"，起涨点+区间位置告诉你"在周期哪个阶段"
    #   - 不固定单一窗口，用多尺度组合（40/80/120/200/300天）捕捉不同级别
    #   - 另外用全历史月均价（大周期）确定历史性高低位
    _cycle_d: Dict = {}
    _cycle_wins = (40, 80, 120, 200, 300)
    for _w in _cycle_wins:
        _act = min(_w, bar_idx + 1)
        if _act < 5:
            _cycle_d[f'gain_{_w}d'] = 0.0
            _cycle_d[f'loss_{_w}d'] = 0.0
            _cycle_d[f'pp_{_w}d']   = 0.5
        else:
            _wh = float(high[max(0, end - _act): end].max())
            _wl = float(low [max(0, end - _act): end].min())
            _wl = max(_wl, 1e-8)
            _cycle_d[f'gain_{_w}d'] = float((cp - _wl) / _wl)
            _cycle_d[f'loss_{_w}d'] = float((_wh - cp) / _wh if _wh > 0 else 0.0)
            _pp = (cp - _wl) / (_wh - _wl) if _wh > _wl else 0.5
            _cycle_d[f'pp_{_w}d']   = float(np.clip(_pp, 0.0, 1.0))

    # ── 大周期历史位置：全历史月均价重采样 ──────────────────────────────
    # 核心思想（用户指出）：每隔30天取该时段的平均价格，
    # 平滑短期噪音后得到月均价序列，当前价格在其中的百分位
    # 即"大周期历史位置"——比固定200/400天窗口更全面
    MONTHLY_STEP = 30
    _monthly_avgs: list = []
    _mi = 0
    while _mi + MONTHLY_STEP <= bar_idx + 1:
        _monthly_avgs.append(float(close[_mi: _mi + MONTHLY_STEP].mean()))
        _mi += MONTHLY_STEP
    if _mi < bar_idx + 1 and (bar_idx + 1 - _mi) >= 5:
        _monthly_avgs.append(float(close[_mi: bar_idx + 1].mean()))

    if len(_monthly_avgs) >= 3:
        _marr    = np.array(_monthly_avgs)
        _m_low   = float(_marr.min())
        _m_high  = float(_marr.max())
        pp_hist  = float((_marr < cp).sum()) / len(_marr)
        gain_hist = float((cp - _m_low)  / _m_low  if _m_low  > 0 else 0.0)
        loss_hist = float((_m_high - cp) / _m_high if _m_high > 0 else 0.0)
        _rm      = _marr[-12:] if len(_marr) >= 12 else _marr
        gain_monthly_12m = float((cp - float(_rm.min())) / float(_rm.min()) if _rm.min() > 0 else 0.0)
        loss_monthly_12m = float((float(_rm.max()) - cp) / float(_rm.max()) if _rm.max() > 0 else 0.0)
    else:
        pp_hist = 0.5
        gain_hist = loss_hist = 0.0
        gain_monthly_12m = loss_monthly_12m = 0.0

    _cycle_d['pp_hist']          = pp_hist
    _cycle_d['gain_hist']        = gain_hist
    _cycle_d['loss_hist']        = loss_hist
    _cycle_d['gain_monthly_12m'] = gain_monthly_12m
    _cycle_d['loss_monthly_12m'] = loss_monthly_12m

    # ── 中周期历史位置：最近N周的周均价（辅助，补充月均的中间层）────────
    # 5日均价（周线）在最近52周（约1年）内的百分位
    # 辅助作用：比40/80天滚动窗口更平滑，比月均价更新鲜
    WEEKLY_STEP = 5
    _weekly_avgs: list = []
    _wi = 0
    while _wi + WEEKLY_STEP <= bar_idx + 1:
        _weekly_avgs.append(float(close[_wi: _wi + WEEKLY_STEP].mean()))
        _wi += WEEKLY_STEP
    if _wi < bar_idx + 1 and (bar_idx + 1 - _wi) >= 2:
        _weekly_avgs.append(float(close[_wi: bar_idx + 1].mean()))

    if len(_weekly_avgs) >= 4:
        _warr_all = np.array(_weekly_avgs)
        # 最近52周（约1年）的百分位
        _warr52   = _warr_all[-52:] if len(_warr_all) >= 52 else _warr_all
        pp_weekly_52w = float((_warr52 < cp).sum()) / len(_warr52)
        # 最近26周（约半年）的起涨点
        _warr26   = _warr_all[-26:] if len(_warr_all) >= 26 else _warr_all
        _w26_low  = float(_warr26.min())
        _w26_high = float(_warr26.max())
        gain_weekly_26w = float((cp - _w26_low)  / _w26_low  if _w26_low  > 0 else 0.0)
        loss_weekly_26w = float((_w26_high - cp) / _w26_high if _w26_high > 0 else 0.0)
    else:
        pp_weekly_52w = 0.5
        gain_weekly_26w = loss_weekly_26w = 0.0

    _cycle_d['pp_weekly_52w']   = pp_weekly_52w
    _cycle_d['gain_weekly_26w'] = gain_weekly_26w
    _cycle_d['loss_weekly_26w'] = loss_weekly_26w

    # ── 趋势延伸度综合分数（四层：日线动能 + 月级起涨点 + 周级区间 + 大周期历史）─
    # 层级：40d日线动能 → 周线区间(52w) → 月均价起涨点(12m) → 全历史百分位
    _g40   = min(_cycle_d['gain_40d'],  1.0)
    _g120  = min(_cycle_d['gain_120d'], 1.0)
    _l40   = min(_cycle_d['loss_40d'],  1.0)
    _l120  = min(_cycle_d['loss_120d'], 1.0)
    _ppw   = pp_weekly_52w      # 52周周均百分位（中间层）
    _pph   = pp_hist            # 全历史月均百分位（大周期，权重最大）
    _cycle_d['up_extension']   = _g40 * 0.15 + _g120 * 0.20 + _ppw * 0.25 + _pph * 0.40
    _cycle_d['down_extension'] = _l40 * 0.15 + _l120 * 0.20 + (1.0 - _ppw) * 0.25 + (1.0 - _pph) * 0.40

    # ── 平台突破检测（前期盘整后的突破）──────────────────────────────
    # 思路（用户指出）："前期一直平台震荡，突破了前期平台高点/跌破低点，
    #   说明可能要进入下一个区间" → 高质量的入场信号
    # 实现：取最近90天（排除最近10天，避免把当前突破纳入平台计算）
    # 得到"前期平台"的高低点，判断当前价格是否已突破
    PLAT_WIN  = 90   # 平台参考窗口（天）
    PLAT_EXCL = 10   # 排除最近N天（避免当前走势污染平台测量）
    _plat_end   = max(0, end - PLAT_EXCL)
    _plat_start = max(0, _plat_end - PLAT_WIN)
    if _plat_end - _plat_start >= 20:
        _plat_h = float(high[_plat_start: _plat_end].max())
        _plat_l = float(low [_plat_start: _plat_end].min())
        _plat_w = (_plat_h - _plat_l) / _plat_l if _plat_l > 0 else 0.0
        # 平台宽度 < 30%：属于横盘整理（过宽说明已经是趋势段）
        _is_plat = _plat_w < 0.30
        platform_breakout_up   = _is_plat and cp > _plat_h * 1.015
        platform_breakout_down = _is_plat and cp < _plat_l * 0.985
    else:
        platform_breakout_up = platform_breakout_down = False
    _cycle_d['platform_breakout_up']   = platform_breakout_up
    _cycle_d['platform_breakout_down'] = platform_breakout_down

    # ── 急涨急跌/暴涨暴跌型的中周期入场信号（不要求大周期）────────
    # 两种情况下启用：
    #   A. 股性标签明确为 volatile_oscillator（由 segmenter 预分类，可靠）
    #   B. 无股性标签时：基于当前bar的R²估计（per-bar，不稳定，仅作后备）
    is_volatile_personality = (cfg.personality == 'volatile_oscillator')
    use_rough_path = is_volatile_personality or (not cfg.personality and is_rough)

    if use_rough_path:
        enter_up_rough   = s_medium > cfg.slope_weak  and (in_upper_half or above_md)
        enter_down_rough = s_medium < -cfg.slope_weak and (in_lower_half or below_md)
    else:
        enter_up_rough   = False
        enter_down_rough = False

    d = {
        's_large': s_large, 's_medium': s_medium, 's_small': s_small,
        's_22d': s_22d, 's_200d': s_200d,
        'sl': sl, 'sm': sm, 'sx': sx,
        'lg_s': lg_s, 'lg_r': lg_r,
        'md_s': md_s, 'md_r': md_r, 'md_mid': md_mid,
        'above_lg': above_lg, 'below_lg': below_lg,
        'above_md': above_md, 'below_md': below_md,
        'in_range': in_range,
        'in_upper_half': in_upper_half, 'in_lower_half': in_lower_half,
        'peak_60': peak_60, 'trough_60': trough_60,
        'drawdown_60': drawdown_60, 'recovery_60': recovery_60,
        'enter_up': enter_up, 'enter_down': enter_down,
        'strong_medium_up': strong_medium_up,
        'strong_medium_down': strong_medium_down,
        'small_up': small_up, 'small_down': small_down,
        # 平滑度相关
        'smoothness':      smoothness,
        'r2_large':        r2_large,
        'r2_medium':       r2_medium,
        'is_smooth':       is_smooth,
        'is_rough':        is_rough,
        'enter_up_rough':  enter_up_rough,
        'enter_down_rough':enter_down_rough,
        # 反弹失败关键节点（> 0.050 = 确认反弹失败；仅在cfg.personality已知时触发）
        'failed_rally_score': _failed_rally_score,
        # 附加因子（默认None，PITStateDetector根据cfg填入）
        'ema_bull': None, 'ema_bear': None,
        'vol_surge_up': None, 'vol_surge_down': None,
        # 斜率速度（由PITStateDetector填入）
        'sv_large':  0.0,   # s_large 在过去 vel_step 天的变化
        'sv_medium': 0.0,   # s_medium 在过去 vel_step 天的变化
        # 长周期价格区间位置
        'pp_long':   pp_long,    # 价格在1年(250d)区间的位置 [0=低点, 1=高点]
        'pp_long2':  pp_long2,   # 价格在2年(500d)区间的位置
        'at_1y_high': at_1y_high,
        'at_1y_low':  at_1y_low,
        'lr_high':   lr_high,
        'lr_low':    lr_low,
    }

    # ── 合并多尺度周期上下文 ──────────────────────────────────────────
    d.update(_cycle_d)

    # ── EMA对齐因子（若cfg.use_ema且有EMA状态） ──────────────────
    if cfg.use_ema and ema_state is not None:
        e20  = ema_state.get('ema20',  cp)
        e60  = ema_state.get('ema60',  cp)
        e120 = ema_state.get('ema120', cp)
        # 多头排列：EMA20 > EMA60 > EMA120 且价格 > EMA20
        d['ema_bull'] = (e20 > e60 > e120) and (cp > e20)
        # 空头排列：EMA20 < EMA60 < EMA120 且价格 < EMA20
        d['ema_bear'] = (e20 < e60 < e120) and (cp < e20)
        # EMA死叉/金叉（短期EMA穿越中期）
        d['ema_golden'] = e20 > e60 and cp > e60   # 金叉区域
        d['ema_death']  = e20 < e60 and cp < e60   # 死叉区域

    # ── 成交量确认因子 ────────────────────────────────────────────
    if cfg.use_volume and vol_ma20 is not None and volume is not None and vol_ma20 > 0:
        vol_ratio = volume / vol_ma20
        # 放量上涨：成交量 > 1.5倍均量 且 当日涨价
        d['vol_surge_up']   = vol_ratio > cfg.vol_surge_ratio and cp > close[bar_idx - 1] if bar_idx > 0 else False
        # 放量下跌：成交量 > 1.5倍均量 且 当日跌价
        d['vol_surge_down'] = vol_ratio > cfg.vol_surge_ratio and cp < close[bar_idx - 1] if bar_idx > 0 else False
        d['vol_ratio'] = vol_ratio

    return d


def _state_aware_signal(d: Dict, current_state: str, cfg: PitConfig) -> str:
    """
    状态感知信号生成 v8。

    核心改进：在斜率基础上引入"起涨点+价格区间"（多尺度）：
      - gain_from_low_Xd：从X天最低点（起涨点）涨了多少 → 上涨延伸度
      - loss_from_high_Xd：从X天最高点（起跌点）跌了多少 → 下跌延伸度
      - pp_Xd：在X天区间的相对位置 [0=底, 1=顶]
      - up_extension / down_extension：综合延伸度分数

    三阶段逻辑：
      早期（low extension）：降低进入门槛，快速响应
      中期（mid extension）：标准大+中双重确认
      后期（high extension）：降低退出门槛，不等斜率完全反转

    ┌──────────────────────────────────────────────────────────────┐
    │ uptrend 维持/退出                                            │
    │   直接反转：大+中双强烈负 → downtrend                       │
    │   标准退出：中期走弱 / 跌破支撑 / 大幅回撤                  │
    │   延伸度退出：上涨后期 → 中期稍弱就退出（不等斜率全负）     │
    ├──────────────────────────────────────────────────────────────┤
    │ downtrend 维持/退出（对称）                                  │
    ├──────────────────────────────────────────────────────────────┤
    │ consolidation 进入                                           │
    │   早期上涨：低延伸度 + 中小周期正 → 快速进入               │
    │   标准进入：大+中双重确认（若延伸度高则要求更强信号）        │
    │   创新高/突破/EMA对齐：辅助入场信号                         │
    └──────────────────────────────────────────────────────────────┘
    """
    ema_bull   = d.get('ema_bull')
    ema_bear   = d.get('ema_bear')
    ema_golden = d.get('ema_golden')
    ema_death  = d.get('ema_death')
    vol_up     = d.get('vol_surge_up')
    vol_down   = d.get('vol_surge_down')

    sw = cfg.slope_weak
    ss = cfg.slope_strong
    sf = cfg.slope_flat
    de = cfg.drawdown_exit

    # ── 斜率速度 ──────────────────────────────────────────────────
    sv_medium  = d.get('sv_medium', 0.0)
    sv_large   = d.get('sv_large',  0.0)
    vt         = cfg.vel_threshold

    # ── 扩展斜率：22日（月线级方向）+ 200日（超大周期趋势）────────────
    s_22d  = d.get('s_22d',  0.0)   # 22日短期方向（用户建议）
    s_200d = d.get('s_200d', 0.0)   # 200日超大周期方向（用户建议）

    # ── 长周期价格区间位置（现有）────────────────────────────────
    pp_long    = d.get('pp_long',  0.5)
    at_1y_low  = d.get('at_1y_low',  False)
    at_1y_high = d.get('at_1y_high', False)
    ll = cfg.long_range_low
    lh = cfg.long_range_high

    # ── 多尺度起涨点 + 区间位置 → 趋势延伸度 ────────────────────
    pp_200d          = d.get('pp_200d',          0.5)
    gain_120d        = d.get('gain_120d',        0.0)
    gain_40d         = d.get('gain_40d',         0.0)
    loss_120d        = d.get('loss_120d',        0.0)
    loss_40d         = d.get('loss_40d',         0.0)
    up_ext           = d.get('up_extension',     0.0)
    dn_ext           = d.get('down_extension',   0.0)
    # 大周期历史位置（全历史月均价百分位）& 中间层（周均价）
    pp_hist          = d.get('pp_hist',          0.5)
    gain_monthly_12m = d.get('gain_monthly_12m', 0.0)
    loss_monthly_12m = d.get('loss_monthly_12m', 0.0)
    pp_weekly_52w    = d.get('pp_weekly_52w',    0.5)
    gain_weekly_26w  = d.get('gain_weekly_26w',  0.0)
    loss_weekly_26w  = d.get('loss_weekly_26w',  0.0)
    # 平台突破
    plat_brk_up   = d.get('platform_breakout_up',   False)
    plat_brk_down = d.get('platform_breakout_down', False)

    # ── 趋势阶段判断（充分利用历史数据：日线+周线+月线+全历史）──────────
    # 上涨早期：历史低位（全历史月均百分位低）+ 周线起涨幅度小
    #   → 价格处于历史偏低区域，方向信号出现就快速进入
    early_up   = (pp_hist < 0.32 and gain_monthly_12m < 0.15
                  and gain_weekly_26w < 0.20)

    # 上涨后期：综合延伸度高，且历史高位 OR 月级大涨
    #   → 从起涨点走了很远，降低退出门槛
    late_up    = (up_ext > 0.36 and (pp_hist > 0.65 or gain_monthly_12m > 0.28
                                     or pp_weekly_52w > 0.75))

    # 下跌早期：历史高位 + 周线跌幅小
    early_down = (pp_hist > 0.68 and loss_monthly_12m < 0.15
                  and loss_weekly_26w < 0.20)

    # 下跌后期：综合延伸度高，且历史低位 OR 月级大跌
    late_down  = (dn_ext > 0.36 and (pp_hist < 0.32 or loss_monthly_12m > 0.28
                                     or pp_weekly_52w < 0.25))

    # ──────────────────────────────────────────────────────────────
    # mean_reverter：改用全历史月均百分位（pp_hist）作主信号
    # 原因：pp_long只看1年，pp_hist用全历史，对箱体股更准确
    #   整理期：历史低位（pp_hist<0.25）停止下跌→uptrend
    #           历史高位（pp_hist>0.75）停止上涨→downtrend
    #   上涨/下跌期：历史高/低位 + 短期减速 → 提前退出
    #   中间区域 fall-through 到通用逻辑（允许检测持续性趋势如 600547）
    # ──────────────────────────────────────────────────────────────
    if cfg.personality == 'mean_reverter' and cfg.use_long_range:
        if current_state == 'consolidation':
            # pp_hist < 0.25 = 历史低25%分位（比pp_long<0.20更稳健）
            if pp_hist < 0.25 and d['s_small'] > -sf:
                return 'uptrend'
            if pp_hist > 0.75 and d['s_small'] < sf:
                return 'downtrend'
        elif current_state == 'uptrend':
            # 历史高位（>68%分位）+ 短期减速 → 早退出
            if pp_hist > 0.68 and d['s_small'] < sf:
                return 'consolidation'
        elif current_state == 'downtrend':
            # 历史低位（<32%分位）+ 短期止跌 → 早退出
            if pp_hist < 0.32 and d['s_small'] > -sf:
                return 'consolidation'
        # fall-through to generic logic

    # ──────────────────────────────────────────────────────────────
    # uptrend：维持 or 退出
    # ──────────────────────────────────────────────────────────────
    if current_state == 'uptrend':
        # 直接反转（最强信号，无论阶段）
        if d['s_large'] < -ss and d['s_medium'] < -sw:
            return 'downtrend'

        # 价格跌破大周期支撑
        if d['below_lg']:
            return 'consolidation'

        # 中期明显走弱
        if d['s_medium'] < -sw:
            return 'consolidation'

        # 大幅回撤 + 小周期下行
        if d['drawdown_60'] > de and not d['small_up']:
            return 'consolidation'

        # ── 延伸度感知退出：上涨后期降低退出门槛 ──────────────────
        # 当 up_extension 高（已从起涨点走了很远），中期动能消失就退出。
        # 使用全历史月均价(pp_hist)后信号更准确，breakout_runner 也适用。
        # 仅排除 trend_persistent（创新高是其核心特征，不受历史高位限制）
        if late_up and cfg.personality != 'trend_persistent':
            if d['s_medium'] < sf:
                return 'consolidation'
            # 200日趋势转弱也是退出信号（大周期开始反向）
            if s_200d < -sf and d['s_medium'] < sw:
                return 'consolidation'

        # 1年高位 + 动能减弱（对 trend_persistent 不生效：创新高是其特征）
        if (cfg.use_long_range and at_1y_high
                and cfg.personality != 'trend_persistent'):
            if d['s_medium'] < sf:
                return 'consolidation'

        # EMA背离 + 中期失去动力
        if cfg.use_ema and cfg.ema_allow_exit and ema_bear:
            if d['s_medium'] < sf:
                return 'consolidation'

        # ── trend_persistent：突发事件早期退出 ───────────────────────
        # 用户洞察："小周期（如日线/周线），只用于检测突发事件——某一天或某几天
        # 的异动，结合具体位置，是不是可能导致趋势反转？"
        #
        # 实现：
        #   突发事件 = 22日年化斜率 < -50%/yr（22天内约 -4.3%）
        #           OR 小周期组（5/10/20日）平均 < -60%/yr（近期急跌）
        #   高位上下文 = 全历史月均百分位 > 65% OR 处于近1年高位
        #   两者同时成立 → 提前退出（不等大/中周期完全反转）
        #
        # 注：其他股性仍用现有退出逻辑；此处仅针对 trend_persistent
        # 原因：trend_persistent 跳过了 late_up / at_1y_high 退出，
        #       需要专属的"异动预警"机制补偿这一宽松性。
        if cfg.personality == 'trend_persistent' and cfg.use_long_range:
            _sudden_drop = d['s_22d'] < -0.50 or d['s_small'] < -0.60
            _at_hist_high = pp_hist > 0.65 or at_1y_high
            if _sudden_drop and _at_hist_high:
                return 'consolidation'

        return 'uptrend'

    # ──────────────────────────────────────────────────────────────
    # downtrend：维持 or 退出（对称）
    # ──────────────────────────────────────────────────────────────
    elif current_state == 'downtrend':
        # 直接反转
        if d['s_large'] > ss and d['s_medium'] > sw:
            return 'uptrend'

        # 价格突破大周期阻力
        if d['above_lg']:
            return 'consolidation'

        # 中期明显走强
        if d['s_medium'] > sw:
            return 'consolidation'

        # 大幅反弹 + 小周期上行
        if d['recovery_60'] > de and d['small_up']:
            return 'consolidation'

        # ── 延伸度感知退出：下跌后期降低退出门槛 ──────────────────
        if late_down:
            if d['s_medium'] > -sf:
                return 'consolidation'

        # 1年低位 + 中期轻微走强
        if (cfg.use_long_range and at_1y_low
                and cfg.personality != 'trend_persistent'):
            if d['s_medium'] > -sf:
                return 'consolidation'

        # 斜率速度早期预警
        if cfg.use_vel and sv_medium > vt:
            if d['small_up'] or sv_large > vt * 0.5:
                return 'consolidation'

        # EMA金叉区域 + 中期轻微走强
        if cfg.use_ema and cfg.ema_allow_exit and ema_golden:
            if d['s_medium'] > -sf:
                return 'consolidation'

        return 'downtrend'

    # ──────────────────────────────────────────────────────────────
    # consolidation：进入 uptrend or downtrend
    # ──────────────────────────────────────────────────────────────
    else:
        # ── 状态惯性：不明确时延续前一个方向 ──────────────────────────
        # 用户原则："如果不明确，还是要延续前面的判断"
        #
        # 适用场景：短暂的技术性反弹/回调结束后，应快速回到前一趋势
        #   而不是在 consolidation 滞留等待 confirm_enter（6天）的完整确认
        #
        # 保护条件（避免误伤）：
        #   1. 排除 mean_reverter：均值回归股在历史极值的反转是核心信号，不能用惯性阻断
        #   2. 不在历史极值位置：历史极低位（pp_hist<0.30）的反弹可能是真实反转，
        #      不应被强制拉回 downtrend；历史极高位同理
        #   3. 大期不支持反向：大周期斜率仍处于原趋势方向
        #   4. 价格位置一致：在下半区才延续 downtrend，上半区才延续 uptrend
        #
        # 注意：_confirm_len 中已将"回到前一方向"的确认天数从 confirm_enter
        #       降至 confirm_exit，此处只需发出信号，机器会更快确认。
        _last_trend = d.get('last_trend')
        _inertia_ok = (
            _last_trend is not None
            and cfg.personality != 'mean_reverter'   # 均值回归不用惯性
        )
        if _inertia_ok and _last_trend == 'downtrend':
            # 保护：处于近1年低位时不强制延续下跌（可能是真实底部）
            # 注意：不用 pp_hist（全历史），因为长期熊市股 pp_hist 全程偏低
            _not_at_1y_low = not d.get('at_1y_low', False)
            _no_bull_large = d['s_large'] < sw       # 大期不支持上涨
            if _not_at_1y_low and _no_bull_large:
                # ── 关键节点：反弹失败加速路径 ──────────────────────────
                # 用户洞察（2023-07-28 案例，000001平安银行）：
                #   "大阳线后没有延续反转，8月11日大阴线彻底浇灭希望"
                #   "这几天影响不能被大周期平均稀释，要加大权重"
                #
                # 检测（10/15/20天窗口内）：
                #   涨幅 > 3%（有真实反弹）且已回吐 > 50%（相对于本次涨幅）
                #   → failed_rally_score = 涨幅 × 回吐比例 > 0.050
                #   （score=0.066在8月11日触发 → 8月17日完成确认, vs 原8月23日）
                #
                # 触发条件（均须满足）：
                #   1. cfg.personality != ''
                #      仅在股性已知时激活：generic模式无法区分trend_persistent股票
                #      的正常回调（假阳性会导致generic ordering -3%/yr回归）
                #   2. failed_rally_score > 0.050
                #      阈值0.050: 排除score=0.0425(8月8日)的早期假触发
                #                 保留score=0.066(8月11日)的真实信号
                #   3. s_small < sw（短期动量 < slope_weak=0.12）
                #      防止volatile_oscillator型股票正常震荡期中触发
                #   4. not above_md（价格未在阻力位之上）
                _failed_rally = (
                    cfg.personality  # 仅在已知股性时激活（generic无法判断假阳性风险）
                    and d.get('failed_rally_score', 0.0) > 0.050
                )
                if _failed_rally and d['s_small'] < sw and not d.get('above_md', False):
                    if not (cfg.use_ema and ema_bull):
                        return 'downtrend'

                # 快速路径：短期已转负 + 价格在下半区 → 反弹失败，回下跌
                if (d['s_small'] < -sf
                        and (d['in_lower_half'] or d['below_md'])):
                    if not (cfg.use_ema and ema_bull):
                        return 'downtrend'
                # 慢速路径：中期偏负 → 趋势延续
                if (d['s_medium'] < -sf
                        and (d['in_lower_half'] or d['below_md'])):
                    if not (cfg.use_ema and ema_bull):
                        return 'downtrend'
        elif _inertia_ok and _last_trend == 'uptrend':
            # 保护：处于近1年高位时不强制延续上涨（可能是真实顶部）
            _not_at_1y_high = not d.get('at_1y_high', False)
            _no_bear_large  = d['s_large'] > -sw     # 大期不支持下跌
            if _not_at_1y_high and _no_bear_large:
                # V-recovery 动量覆盖：长+短+月三重确认上涨时，绕过中期和价格位置
                # 适用场景：上涨段回调后 V 形反弹，60d 区间被旧峰撑大导致 pp_60
                # 偏低，40/60d 斜率被峰→谷拖累。此时 s_large 已是正的（120d 仍含
                # 更早的上涨段），s_small 和 s_22d 都极强，唯独 s_medium 和 pp_60
                # 因窗口机械性地包含了旧高峰而卡住。三级别共振 = 高可信度。
                if (d['s_large'] > sw
                        and d['s_small'] > ss
                        and d['s_22d'] > ss):
                    if not (cfg.use_ema and ema_bear):
                        return 'uptrend'
                # 快速路径：短期已转正 + 价格在上半区 → 回调结束，回上涨
                if (d['s_small'] > sf
                        and (d['in_upper_half'] or d['above_md'])):
                    if not (cfg.use_ema and ema_bear):
                        return 'uptrend'
                # 慢速路径：中期偏正 → 趋势延续
                if (d['s_medium'] > sf
                        and (d['in_upper_half'] or d['above_md'])):
                    if not (cfg.use_ema and ema_bear):
                        return 'uptrend'

        # ── 早期上涨快速进入：低延伸度 + 中小周期均正 ──────────────
        # 在起涨初期（低位、小涨幅），不等大周期，中期+短期双正即可进入
        # 这是解决"斜率滞后"的核心改进：早期信号不等大周期回应
        if early_up and d['s_medium'] > sf and d['s_small'] > sf:
            if not (cfg.use_ema and ema_bear):
                return 'uptrend'

        # ── 标准进入：大+中双重确认 ───────────────────────────────
        if d['enter_up']:
            if late_up and cfg.personality != 'trend_persistent':
                # 上涨后期（非trend_persistent）：需要更强信号才进入（避免末段追高）
                if d['s_large'] > ss:
                    return 'uptrend'
            else:
                if not (cfg.use_ema and ema_bear):
                    return 'uptrend'
        if d['enter_down']:
            if late_down and cfg.personality != 'trend_persistent':
                if d['s_large'] < -ss:
                    return 'downtrend'
            else:
                if not (cfg.use_ema and ema_bull):
                    return 'downtrend'

        # ── 早期下跌快速进入（对称）──────────────────────────────
        # 仅排除 trend_persistent（创新高是其特征，高位不应被快速推入下跌）
        if early_down and d['s_medium'] < -sf and d['s_small'] < -sf:
            if cfg.personality != 'trend_persistent':
                if not (cfg.use_ema and ema_bull):
                    return 'downtrend'

        # ── 平台突破：高质量进场信号（用户建议：前期盘整突破高点/低点）─
        # 前提：之前是横盘整理（平台宽度<30%），现在价格突破平台高低点
        # 不需要等大+中双重确认，单中期方向确认即可
        if plat_brk_up and d['s_medium'] > sf:
            if cfg.personality != 'mean_reverter':   # 均值回归不追突破
                return 'uptrend'
        if plat_brk_down and d['s_medium'] < -sf:
            if cfg.personality != 'mean_reverter':
                return 'downtrend'

        # ── volatile_oscillator 快速进入 ────────────────────────
        if d['enter_up_rough']:
            if not (cfg.use_ema and ema_bear):
                return 'uptrend'
        if d['enter_down_rough']:
            if not (cfg.use_ema and ema_bull):
                return 'downtrend'

        # ── 创1年新高/新低：突破性强信号 ─────────────────────────
        if cfg.use_long_range and at_1y_high and d['s_medium'] > sf:
            if cfg.personality != 'mean_reverter':
                return 'uptrend'
        if cfg.use_long_range and at_1y_low and d['s_medium'] < -sf:
            if cfg.personality != 'mean_reverter':
                return 'downtrend'

        # ── 突破大/中期区间 ──────────────────────────────────────
        if d['above_lg'] and d['s_medium'] > sf:
            return 'uptrend'
        if d['below_lg'] and d['s_medium'] < -sf:
            return 'downtrend'

        if d['above_md'] and d['s_medium'] > sw:
            return 'uptrend'
        if d['below_md'] and d['s_medium'] < -sw:
            return 'downtrend'

        # 中期极强（大底反弹或顶部崩塌）
        if d['strong_medium_up'] and (d['above_md'] or d['in_range']):
            return 'uptrend'
        if d['strong_medium_down'] and (d['below_md'] or d['in_range']):
            return 'downtrend'

        # 成交量放量突破
        if cfg.use_volume and vol_up  and d['above_md'] and d['s_medium'] > -sf:
            return 'uptrend'
        if cfg.use_volume and vol_down and d['below_md'] and d['s_medium'] < sf:
            return 'downtrend'

        # EMA对齐 + 弱斜率信号
        if cfg.use_ema and cfg.ema_confirm_enter:
            if ema_bull and d['s_medium'] > sf and d['in_upper_half']:
                return 'uptrend'
            if ema_bear and d['s_medium'] < -sf and d['in_lower_half']:
                return 'downtrend'

        return 'consolidation'   # 假设法：保持整理


# ─────────────────────────────────────────────────────────────────
# 状态机
# ─────────────────────────────────────────────────────────────────

MIN_BARS = 20   # 最少20bar才开始计算

class PITStateDetector:
    """
    多周期PIT状态机 v7。

    支持可配置参数（PitConfig），向后兼容（无参数时使用默认配置）。
    新增：增量EMA追踪（O(1)/bar），可选成交量因子。
    """

    def __init__(self, cfg: Optional[PitConfig] = None):
        self.cfg = cfg or PitConfig()
        self._reset()

    def _reset(self):
        from collections import deque
        self._state      = 'consolidation'
        self._confirm_buf: List[str] = []
        self._state_age  = 0
        self._last_trend: Optional[str] = None   # 最近一次非consolidation状态

        # 增量EMA（多头/空头排列分析）
        self._ema20  = None
        self._ema60  = None
        self._ema120 = None

        # 增量成交量MA20（环形缓冲区）
        self._vol_buf:  List[float] = []
        self._vol_ma20: float = 0.0

        # 斜率速度：保存最近 vel_step+1 个bar的斜率值
        # slope_velocity[t] = slope[t] - slope[t - vel_step]
        step = self.cfg.vel_step if hasattr(self, 'cfg') else 5
        self._s_large_hist:  deque = deque(maxlen=step + 1)
        self._s_medium_hist: deque = deque(maxlen=step + 1)

    def _update_ema(self, price: float):
        """增量更新EMA20/60/120（O(1)，严格PIT）。"""
        if self._ema20 is None:
            self._ema20 = self._ema60 = self._ema120 = price
        else:
            a20  = 2.0 / 21
            a60  = 2.0 / 61
            a120 = 2.0 / 121
            self._ema20  = a20  * price + (1 - a20)  * self._ema20
            self._ema60  = a60  * price + (1 - a60)  * self._ema60
            self._ema120 = a120 * price + (1 - a120) * self._ema120

    def _update_vol_ma(self, volume: float):
        """增量更新成交量20日均值（O(1)）。"""
        self._vol_buf.append(volume)
        if len(self._vol_buf) > self.cfg.vol_window:
            self._vol_buf.pop(0)
        self._vol_ma20 = float(np.mean(self._vol_buf)) if self._vol_buf else 0.0

    def _grace_period(self, state: str, vol: float) -> int:
        """当前状态的最短持续天数（grace period）。"""
        cfg  = self.cfg
        base = cfg.grace_trend if state in ('uptrend', 'downtrend') else cfg.grace_consol
        if vol > cfg.vol_high:
            factor = 0.80
        elif vol < cfg.vol_low:
            factor = 1.25
        else:
            factor = 1.0
        min_days = 10 if state in ('uptrend', 'downtrend') else 8
        return max(min_days, round(base * factor))

    def _confirm_len(self, from_s: str, to_s: str, vol: float,
                     last_trend: Optional[str] = None) -> int:
        """股性自适应确认天数。

        状态惯性：从 consolidation 回到前一趋势方向，用较短的 confirm_exit
        而非 confirm_enter（用户原则："不明确时延续前面的判断"）。
        进入相反方向仍需完整的 confirm_enter。
        """
        cfg = self.cfg
        if from_s == 'consolidation':
            if last_trend is not None and to_s == last_trend:
                # 回到前一趋势方向：门槛降低（用 confirm_exit）
                base = cfg.confirm_exit
            else:
                base = cfg.confirm_enter
        elif to_s == 'consolidation':
            base = cfg.confirm_exit
        else:
            base = cfg.confirm_rev

        if vol > cfg.vol_high:
            factor = 0.70
        elif vol > cfg.vol_mid:
            factor = 0.85
        elif vol < cfg.vol_low:
            factor = 1.35
        elif vol < cfg.vol_mid:
            factor = 1.15
        else:
            factor = 1.0

        return max(3, round(base * factor))

    def update(self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
               bar_idx: int,
               volume: Optional[np.ndarray] = None) -> Tuple[str, Dict]:
        """
        在bar_idx处更新状态（PIT：只用[0..bar_idx]数据）。

        volume: 完整成交量数组（若cfg.use_volume=True则需传入）
        """
        if bar_idx < MIN_BARS:
            return self._state, {'insufficient': True}

        price = close[bar_idx]

        # 增量更新EMA和成交量MA
        self._update_ema(price)
        if self.cfg.use_volume and volume is not None:
            self._update_vol_ma(float(volume[bar_idx]))

        # EMA状态快照
        ema_state = None
        if self.cfg.use_ema:
            ema_state = {
                'ema20':  self._ema20,
                'ema60':  self._ema60,
                'ema120': self._ema120,
            }

        vol_cur = float(volume[bar_idx]) if (self.cfg.use_volume and volume is not None) else None

        d   = _compute_data(close, high, low, bar_idx, self.cfg,
                            ema_state=ema_state,
                            vol_ma20=self._vol_ma20 if self.cfg.use_volume else None,
                            volume=vol_cur)

        # ── 斜率速度（slope velocity）────────────────────────────
        # 将当前斜率存入历史队列
        self._s_large_hist.append(d['s_large'])
        self._s_medium_hist.append(d['s_medium'])
        # 若历史足够长，计算速度 = 当前斜率 - vel_step天前斜率
        if self.cfg.use_vel and len(self._s_large_hist) == self.cfg.vel_step + 1:
            d['sv_large']  = self._s_large_hist[-1]  - self._s_large_hist[0]
            d['sv_medium'] = self._s_medium_hist[-1] - self._s_medium_hist[0]

        vol = _estimate_vol(close, bar_idx)   # 波动率（用于自适应）
        # 将 _last_trend 注入 d，让 _state_aware_signal 可以使用
        d['last_trend'] = self._last_trend
        raw = _state_aware_signal(d, self._state, self.cfg)

        # 确认窗口更新
        max_c = max(self.cfg.confirm_enter, self.cfg.confirm_exit, self.cfg.confirm_rev)
        self._confirm_buf.append(raw)
        if len(self._confirm_buf) > max_c:
            self._confirm_buf.pop(0)

        self._state_age += 1

        # 状态切换判断：超过grace period才允许
        if raw != self._state:
            grace = self._grace_period(self._state, vol)
            if self._state_age >= grace:
                need   = self._confirm_len(self._state, raw, vol,
                                           last_trend=self._last_trend)
                recent = self._confirm_buf[-need:]
                if len(recent) >= need and all(r == raw for r in recent):
                    # 维护 _last_trend：记录离开时的方向
                    if self._state in ('uptrend', 'downtrend'):
                        self._last_trend = self._state
                    self._state     = raw
                    self._state_age = 0

        d.update({
            'state':      self._state,
            'raw_signal': raw,
            'vol':        vol,
            'state_age':  self._state_age,
        })
        return self._state, d

    def compute_all(self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
                    volume: Optional[np.ndarray] = None) -> List[str]:
        """逐bar计算全序列状态（PIT严格顺序）。"""
        self._reset()
        return [self.update(close, high, low, t, volume)[0] for t in range(len(close))]


# ─────────────────────────────────────────────────────────────────
# 快捷函数（向后兼容）
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# 股性DNA计算（从全历史数据提取每只股票的天然振幅特征）
# ─────────────────────────────────────────────────────────────────

def compute_stock_dna(close: np.ndarray,
                      high: Optional[np.ndarray] = None,
                      low:  Optional[np.ndarray] = None) -> Dict:
    """
    从全历史数据计算股票的"股性DNA"特征。

    目的：不同天然振幅的股票需要不同的斜率阈值。
      - 000001（平安银行）：月度振幅~6%，年化趋势斜率~8%/yr
        → 默认 slope_weak=0.12 永远触发不了，策略完全失效
      - 300274（陕西华泰）：月度振幅~20%，年化趋势斜率~60%/yr
        → 默认 slope_weak=0.12 完全合适

    参数：
      close: 完整收盘价序列（全历史，越长越准确）
      high/low: 可选，有则用真实高低价（更准确），无则用收盘价近似

    返回 dict 包含：
      volatility:            年化日收益率波动率（σ × √252）
      autocorr:              日收益率1日自相关（正=趋势动量，负=均值回归）
      typical_monthly_range: 典型月度振幅中位数（high-low range / low）
      swing_type:            'low_swing' / 'normal' / 'high_swing'
      slope_scale:           建议斜率阈值缩放因子（低振幅股 < 1.0）
    """
    n = len(close)

    # 数据不足：返回中性DNA（不做任何调整）
    if n < 60:
        return {
            'volatility':            0.30,
            'autocorr':              0.0,
            'typical_monthly_range': 0.15,
            'swing_type':            'normal',
            'slope_scale':           1.0,
        }

    # ── 年化波动率 ──────────────────────────────────────────────────
    log_rets = np.diff(np.log(np.maximum(close, 1e-8)))
    volatility = float(np.std(log_rets) * np.sqrt(252))

    # ── 日收益率1阶自相关（趋势性 vs 均值回归性）─────────────────────
    # 正值：今天涨明天也倾向于涨（趋势/动量型）
    # 负值：今天涨明天倾向于跌（均值回归型）
    if len(log_rets) >= 20:
        r = log_rets - log_rets.mean()
        std_r = float(np.std(r))
        if std_r > 1e-10:
            autocorr = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    # ── 典型月度振幅（天然振幅的直接度量）─────────────────────────
    # 每30天取该月的 (high - low) / low
    # 取中位数：避免极端牛熊月份的影响，代表"正常情况下一个月的波动幅度"
    # 这是判断股票天然振幅最稳健的指标：
    #   - 000001：月度振幅约5-7%（缓慢漂移型）
    #   - 300274：月度振幅约15-25%（大波段成长股）
    MONTH = 30
    monthly_ranges: list = []
    if high is not None and low is not None:
        # 用真实高低价（推荐）
        for i in range(0, n - MONTH + 1, MONTH):
            h = float(high[i: i + MONTH].max())
            l = float(low[i:  i + MONTH].min())
            if l > 0:
                monthly_ranges.append((h - l) / l)
    else:
        # 用收盘价近似（略偏低）
        for i in range(0, n - MONTH + 1, MONTH):
            seg = close[i: i + MONTH]
            h = float(seg.max())
            l = float(seg.min())
            if l > 0:
                monthly_ranges.append((h - l) / l)

    typical_monthly_range = float(np.median(monthly_ranges)) if monthly_ranges else 0.15

    # ── 振幅类型分类 ─────────────────────────────────────────────────
    # 阈值根据A股实际情况设定：
    #   < 0.08  → 低振幅（银行/公用事业/消费龙头等慢涨型）
    #   > 0.22  → 高振幅（科技成长/小市值/周期股等大波段型）
    #   中间    → 普通（大多数股票）
    if typical_monthly_range < 0.08:
        swing_type = 'low_swing'
    elif typical_monthly_range > 0.22:
        swing_type = 'high_swing'
    else:
        swing_type = 'normal'

    # ── 斜率缩放因子 ─────────────────────────────────────────────────
    # 参考振幅 0.15 对应默认 slope_weak=0.12（在A股正常股票中表现最好的配置）
    # 低振幅股：等比缩小阈值，让低速趋势也能被检测到
    # 高振幅股：保持默认（默认对高波动已经很好，不做向上调整）
    # 下限0.40：slope_weak 最低 = 0.12 × 0.40 = 0.048（太低则噪音太多）
    REF_RANGE  = 0.15
    raw_scale  = typical_monthly_range / REF_RANGE
    slope_scale = float(np.clip(raw_scale, 0.40, 1.0))   # 只向下缩放

    return {
        'volatility':            volatility,
        'autocorr':              autocorr,
        'typical_monthly_range': typical_monthly_range,
        'swing_type':            swing_type,
        'slope_scale':           slope_scale,
    }


def make_personality_config(personality: str,
                            dna: Optional[Dict] = None) -> PitConfig:
    """
    根据股性标签创建对应的 PitConfig。

    - trend_persistent:    大+中双重确认（当前默认行为）
    - volatile_oscillator: 只用中周期 + 速度信号（快速响应）
    - mean_reverter:       更短的确认窗口，重价格位置而非斜率
    - breakout_runner:     更宽的回撤退出，允许更长grace期
    - staircase_mover:     与 trend_persistent 类似但稍短
    - 其他:                默认配置

    dna: compute_stock_dna()的返回值（可选）。
         传入后会根据股票的天然振幅进一步调整斜率阈值：
           低振幅股（000001类）→ 降低slope_weak/strong，让低速趋势可被检测
           高振幅股（300274类）→ 保持默认（无需调整）
    """
    if personality == 'trend_persistent':
        cfg = PitConfig(
            personality=personality,
            # ── 使用默认参数 + 突发事件退出机制 ──────────────────────────────
            # 教训（实测）：
            #   (A) 窗口拉到 (120,200) + slope_weak=0.18 → -57%/yr（大滞后）
            #   (B) grace_trend=15, confirm_exit=5 → -37%/yr（更难退出，标签更坏）
            #   (C) 默认参数 (grace=12, exit=4) → -16%/yr（当前最优基准）
            #
            # 结论：对于这类正在震荡的股票，"让信号更快"而非"更慢"才是方向。
            # 因此恢复默认参数，唯一额外机制是 突发事件退出（见 _state_aware_signal）：
            #   短期急跌（22日斜率 < -50%/yr）+ 历史高位位置（pp_hist > 65%）
            #   → 提前退出，不等大周期完全反转。
        )

    elif personality == 'volatile_oscillator':
        cfg = PitConfig(
            personality=personality,
            # 只用中周期（enter_up_rough/down_rough in _compute_data）
            # 更快的确认：高波动股信号持续时间短，需要快速响应
            confirm_enter=4,
            confirm_exit=3,
            confirm_rev=7,
            grace_trend=8,
            grace_consol=5,
            # 斜率速度更重要：帮助提前退出滞后的下跌标签
            use_vel=True,
            vel_step=5,
            vel_threshold=0.08,
            # 不需要 per-bar R² 平滑度过滤（由 personality 标签决定路径）
            use_smoothness=False,
        )

    elif personality == 'mean_reverter':
        # 均值回归型：以价格区间位置为主信号（在 _state_aware_signal 中实现）
        # 需要较快的确认（区间底/顶不会持续很久）和适中的持续时间
        cfg = PitConfig(
            personality=personality,
            confirm_enter=3,   # 底部/顶部停留时间短，需要快速进入
            confirm_exit=3,
            confirm_rev=6,
            grace_trend=8,     # 适当持有（等待均值回归完成）
            grace_consol=5,
        )

    elif personality == 'breakout_runner':
        cfg = PitConfig(
            personality=personality,
            # 基线(generic)参数对这类股效果已很好(+18.1%/yr ordering)
            # 主要差异来自 _state_aware_signal 中的信号路径：
            #   - at_1y_high+弱动能时退出（与其他非trend_persistent相同）
            #   - at_1y_high+正动能时进入uptrend（突破检测）
            # 保持默认参数，不做 confirm/grace 特殊调整
            confirm_enter=6,
            confirm_exit=4,
            confirm_rev=10,
            grace_trend=12,
            grace_consol=6,
            drawdown_exit=0.18,
        )

    elif personality == 'staircase_mover':
        cfg = PitConfig(
            personality=personality,
            # 阶梯型：中等速度，介于 trend_persistent 和 volatile 之间
            confirm_enter=5,
            confirm_exit=3,
            confirm_rev=8,
            grace_trend=10,
            grace_consol=5,
        )

    else:
        # 默认/不足：通用配置
        cfg = PitConfig(personality=personality)

    # ── DNA自适应：按天然振幅缩放斜率阈值 ────────────────────────────
    # 目的：让低振幅股（月度振幅<默认参考值0.15）的缓慢趋势也能被检测到。
    #
    # 注意：breakout_runner 不做DNA缩放。
    #   原因：breakout_runner型股票（如银行股601398）虽然月度振幅偏低(~10%)，
    #   但这类股票本来就是趋势性强的品种，在当前阈值下已有+12%+/yr ordering。
    #   降低阈值反而引入噪音，使ordering下降。DNA缩放只对振幅低且策略表现差的股票有益。
    if dna is not None and personality != 'breakout_runner':
        scale = dna.get('slope_scale', 1.0)
        if abs(scale - 1.0) > 0.05:   # 变化超过5%才调整（避免无意义微调）
            cfg.slope_strong        = round(cfg.slope_strong        * scale, 3)
            cfg.slope_weak          = round(cfg.slope_weak          * scale, 3)
            cfg.slope_flat          = round(cfg.slope_flat          * scale, 3)
            cfg.min_meaningful_move = round(cfg.min_meaningful_move * scale, 3)
            # drawdown_exit也按振幅调整，但幅度保守（避免过激）
            # 低振幅股下限：0.18 × 0.70 = 0.126（约2个月的典型下跌）
            de_scale = 0.30 + 0.70 * scale   # 比slope更保守
            cfg.drawdown_exit = round(cfg.drawdown_exit * de_scale, 3)

    return cfg


def compute_pit_states(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       cfg: Optional[PitConfig] = None,
                       volume: Optional[np.ndarray] = None) -> List[str]:
    return PITStateDetector(cfg).compute_all(close, high, low, volume)


def get_current_state(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                      cfg: Optional[PitConfig] = None,
                      volume: Optional[np.ndarray] = None) -> Tuple[str, Dict]:
    det  = PITStateDetector(cfg)
    all_ = det.compute_all(close, high, low, volume)
    d    = _compute_data(close, high, low, len(close) - 1, cfg or PitConfig())
    d['state'] = all_[-1]
    return all_[-1], d


# ─────────────────────────────────────────────────────────────────
# 下跌阶段细分：compute_downtrend_phase
# ─────────────────────────────────────────────────────────────────

def compute_downtrend_phase(
    close:       np.ndarray,
    high:        np.ndarray,
    low:         np.ndarray,
    states:      List[str],
    cfg:         Optional[PitConfig] = None,
    dna:         Optional[Dict]      = None,
    return_data: bool                = False,
):
    """
    对 compute_pit_states() 的输出做下跌阶段细分。

    非 downtrend bar 原样保留；downtrend bar 替换为：

      'downtrend_steep'    急跌阶段：3个月净跌 > STEEP_THRESH，方向主导，回避入场。
                           如 688981 2020-07~10，2021-11~2022-04

      'downtrend_gradual'  缓跌阶段：有方向但不急，轻仓观望。
                           如 688981 2023-03~2024-09

      'downtrend_range'    震荡区间：净斜率接近零 + 振幅/净变化比高，高抛低吸。
                           如 688981 2020-10~2021-10，2022-05~2023-02

      'downtrend_bottom'   downtrend_range 条件 + pp_hist 历史低位 + s_22d 动能转正，
                           底部脱离，可试探建仓。

    参数：
        close, high, low : 与 compute_pit_states() 完全相同的价格数组
        states           : compute_pit_states() 的返回值（等长列表）
        cfg              : 同一个 PitConfig（默认 PitConfig()）
        dna              : compute_stock_dna() 的返回值（可选）。
                           传入后用 typical_monthly_range 归一化 amp22，
                           得到相对波动率 rel_vol（后期高抛低吸策略调参用）。
        return_data      : False（默认）→ 只返回 List[str]
                           True         → 返回 (List[str], List[Optional[dict]])
                           data[t] 对 downtrend bar 包含以下可调参特征：
                             ret_66    : 66日净回报（负=下跌）
                             amp22     : 22日H-L振幅 / close
                             atr_ratio : amp22 / |ret_66|（越大越震荡）
                             rel_vol   : amp22 / typical_monthly_range
                                         （相对于该股历史振幅的归一化值，
                                          >1.3=当前高波动，<0.7=当前低波动）
                                         需要传入 dna，否则为 None
                             s_22d     : 22日年化斜率（底部动能检测）
                             pp_hist   : 全历史月均价分位（0=历史最低，1=历史最高）

    阈值（来自 688981 实证数据，可通过 sweep 进一步校准）：
        STEEP_THRESH    = 0.12   66日净跌 > 12% = 急跌
        RANGE_THRESH    = 0.06   66日净变 < 6% + atr_ratio > 1.5 = 震荡
        BOTTOM_PP_MAX   = 0.30   历史分位 < 30% = 历史低位区
        BOTTOM_MOM_MIN  = 0.25   s_22d > 0.25/yr = 近期有上行动能
    """
    cfg = cfg or PitConfig()
    n   = len(close)

    # ── 阈值 ──────────────────────────────────────────────────────
    # 急跌阈值：使用回归斜率（比端点回报更准，能捕捉稳步下跌）
    #   s_medium（40/60日）< -0.30/yr  → 中周期趋势明确向下
    #   s_large（80/120日）< -0.25/yr  → 大周期确认
    #   OR逻辑：任一触发即为急跌（提高灵敏度）
    STEEP_SLOPE_MED  = 0.30   # s_medium 急跌阈值（年化）
    STEEP_SLOPE_LRG  = 0.25   # s_large  急跌阈值（年化）
    RANGE_THRESH     = 0.06   # 66日净变 < 6% + 振幅比 > 1.5 = 震荡
    BOTTOM_PP_MAX    = 0.30
    BOTTOM_MOM_MIN   = 0.25
    MONTHLY_STEP     = 30

    # dna 中提取典型月度振幅（用于 rel_vol 归一化）
    ref_amp: Optional[float] = None
    if dna is not None:
        ref_amp = float(dna.get('typical_monthly_range', 0.0)) or None

    result: List[str]               = list(states)
    feat:   List[Optional[Dict]]    = [None] * n   # 仅 return_data=True 时填充

    for t in range(n):
        if states[t] != 'downtrend':
            continue

        # ── 1. 回归斜率（急跌检测主力，复用PIT内部函数） ────────
        # 端点回报会低估稳步下跌的趋势强度；回归斜率拟合全程数据，更准确
        # s_medium：40/60日中周期（反应较快，1个月内能检测到新急跌）
        # s_large ：80/120日大周期（滞后更多，但更稳健）
        s_medium = _group_slope(close, t, cfg.scales_medium, cfg)
        s_large  = _group_slope(close, t, cfg.scales_large,  cfg)

        # ── 2. 66日净回报（震荡检测辅助） ────────────────────────
        w66    = min(66, t + 1)
        ret_66 = float(close[t] / close[t + 1 - w66] - 1) if w66 >= 10 else 0.0

        # ── 2. 22日振幅 + 22日净回报 ──────────────────────────────
        w22h   = min(22, t + 1)
        h22    = float(high[t + 1 - w22h: t + 1].max())
        l22    = float(low[t + 1 - w22h: t + 1].min())
        amp22  = (h22 - l22) / close[t] if close[t] > 0 else 0.0
        ret_22 = float(close[t] / close[t + 1 - w22h] - 1) if w22h >= 5 else 0.0

        # ── 3. 振幅/净变化比（信噪比逆数） ───────────────────────
        atr_ratio_66 = amp22 / max(abs(ret_66), 0.01)
        atr_ratio_22 = amp22 / max(abs(ret_22), 0.01)

        # ── 4. 60日价格区间分位（价格分布） ──────────────────────
        # 判断当前价格在近期高低点区间中的相对位置
        #   < 0.25 → 仍在60日底端（急跌或刚触底）
        #   > 0.35 → 已离开底部（震荡区、已从低点回升）
        # 这是区分 急跌 vs 震荡 的核心特征：
        #   急跌期价格持续创60日新低（pp_60 持续偏低）
        #   震荡期价格在60日区间内来回摆动（pp_60 在0.2~0.8之间波动）
        w60   = min(60, t + 1)
        h60   = float(high[t + 1 - w60: t + 1].max())
        l60   = float(low[t + 1 - w60: t + 1].min())
        r60   = max(h60 - l60, close[t] * 0.001)
        pp_60 = (close[t] - l60) / r60   # 0=60日低点, 1=60日高点

        # ── 5. 22日年化斜率 ───────────────────────────────────────
        s_22d = _log_slope_annualized(close[t + 1 - w22h: t + 1])

        # ── 6. pp_hist（全历史月均价分位）────────────────────────
        end  = t + 1
        mi   = 0
        marr: List[float] = []
        while mi + MONTHLY_STEP <= end:
            marr.append(float(close[mi: mi + MONTHLY_STEP].mean()))
            mi += MONTHLY_STEP
        if mi < end and (end - mi) >= 5:
            marr.append(float(close[mi: end].mean()))
        pp_hist = (float(np.sum(np.array(marr) < close[t])) / len(marr)
                   if len(marr) >= 3 else 0.5)

        # ── 7. 相对波动率 ────────────────────────────────────────
        rel_vol: Optional[float] = (amp22 / ref_amp) if ref_amp else None

        # ── 8. 分类（价格分布 + 多尺度回报联合判断）────────────
        #
        # ── 8a. 60日下跌bar密度（持续性检测） ──────────────────────
        dn_60 = sum(1 for s in states[max(0, t - 60): t] if s == 'downtrend')
        dn_density = dn_60 / max(min(60, t), 1)

        # ── 8b. 1年前低点位置（震荡区间识别）───────────────────────
        # 用户核心观察：震荡期价格"没走出震荡区间"—— 即价格明显高于历史底部
        #   急跌期：价格持续创年新低 → parm_1yr ≈ 0 或负数
        #   震荡期：价格高于1年前底部5%+ → 已在震荡区间内，不应标为steep
        # 典型案例：
        #   2024-02 close≈44，2022年底≈38 → parm_1yr=14% → 是震荡区间，非急跌
        #   2020-12 close≈57，2020-09底≈50 → parm_1yr=12% → 是震荡区间，非急跌
        W_YR = min(252, t)          # ~1年 ≈ 252 交易日
        if W_YR > 0:                # 用已有历史（即使不足60日）计算底部距离
            prior_min_1yr  = float(low[max(0, t - W_YR): t].min())
            parm_1yr       = (close[t] - prior_min_1yr) / close[t] if close[t] > 0 else 0.0
        else:
            parm_1yr = 0.0          # 首日无历史，视为在底部
        # 两档底部阈值（区分密度路径 vs 极端斜率旁路）：
        #   密度路径（dnd>0.75）使用宽松 7% 阈值：
        #     → 允许 急跌②/Dec2021（af1y=6%<7%，dnd=0.89）标为急跌
        #     → 阻断 震荡②/May2022（af1y=11%>7%，dnd=0.96）被误标
        #   极端旁路使用严格 5% 阈值：
        #     → 阻断 缓跌+震荡/Feb2024（af1y=8%>5%，s_med=-1.36极端短期斜率）被误标
        FLOOR_GAP_DENSE   = 0.07   # 密度路径使用的底部阈值
        FLOOR_GAP_EXTREME = 0.10   # 极端旁路使用的底部阈值（>10%=明确离底，非急跌区间）
        above_floor_dense   = parm_1yr > FLOOR_GAP_DENSE
        above_floor_extreme = parm_1yr > FLOOR_GAP_EXTREME

        # ── 核心判断逻辑 ────────────────────────────────────────────
        # 极端旁路加 not above_floor_extreme（parm_1yr < 5%）：
        #   急跌①/Sep2020（parm_1yr≈2%）→ 旁路可触发，steep ✓
        #   震荡①/Oct2020（parm_1yr≈14%）→ 旁路封堵，→ gradual（语义更准确）
        #   密度路径不受影响（急跌②/Dec2021 dnd=0.89 仍触发 steep）
        is_steep = (
            (s_medium < -STEEP_SLOPE_MED or s_large < -STEEP_SLOPE_LRG)
            and pp_60 < 0.35
            and (
                dn_density > 0.75                                  # 持续性：密度已充分积累
                or (s_medium < -0.80 and s_large < -0.60           # 极端旁路
                    and not above_floor_extreme)                    # 须接近1年底部（parm_1yr<5%）
            )
        )
        is_range_fast = (              # 快轨：22日震荡 + 价格已离底
            pp_60 > 0.25
            and abs(ret_22) < 0.05
            and atr_ratio_22 > 1.5
        )
        is_range_slow = (              # 慢轨：66日震荡确认
            abs(ret_66) < RANGE_THRESH
            and atr_ratio_66 > 1.5
        )
        is_range_zone = (              # 区间轨：价格在震荡区间内，放宽净变阈值
            # 核心场景：震荡期内的局部回摆（如震荡①2021-02，ret_66≈-10%但整体不跌）
            # 条件：1) 明确在震荡区间内（高于1年底部7%+）
            #        2) 3个月净变 < 15%（放宽覆盖 2021-10 的 ret_66≈13%）
            #        3) 振幅/净变比 > 1.2（仍有震荡特征）
            #        4) 中周期不强势方向（|s_med|<0.65，覆盖 2021-10 的 s_med≈-0.59）
            above_floor_dense               # parm_1yr > 7%：在震荡区间内
            and abs(ret_66) < 0.15          # 66日净变 < 15%（放宽，覆盖深度震荡回摆）
            and atr_ratio_66 > 1.2          # 振幅/净变比 > 1.2（震荡特征）
            and abs(s_medium) < 0.65        # 中周期不强势方向（稍放宽）
        )

        if is_steep:
            phase = 'downtrend_steep'
        elif is_range_fast or is_range_slow or is_range_zone:
            # 底部脱离要求价格非常接近1年底部（parm_1yr < 5%），防止震荡中期误判为底部
            if pp_hist < BOTTOM_PP_MAX and s_22d > BOTTOM_MOM_MIN and parm_1yr < 0.05:
                phase = 'downtrend_bottom'
            else:
                phase = 'downtrend_range'
        else:
            phase = 'downtrend_gradual'

        result[t] = phase

        if return_data:
            feat[t] = {
                's_medium':     s_medium,
                's_large':      s_large,
                'dn_density':   dn_density,  # 60日下跌bar密度（持续性）
                'ret_22':       ret_22,
                'ret_66':       ret_66,
                'amp22':        amp22,
                'atr_ratio_22': atr_ratio_22,
                'atr_ratio_66': atr_ratio_66,
                'pp_60':        pp_60,
                'parm_1yr':     parm_1yr,   # 高于1年底部的比例（>7%=震荡区间密度路径，>5%=极端旁路）
                'rel_vol':      rel_vol,
                's_22d':        s_22d,
                'pp_hist':      pp_hist,
            }

    if return_data:
        return result, feat
    return result


def smooth_macro_phase(
    phases: List[str],
    close:  np.ndarray,
    high:   np.ndarray,
    low:    np.ndarray,
    volume: Optional[np.ndarray] = None,
    confirmed_only: bool = False,
) -> List[str]:
    """
    宏观趋势平滑：基于状态惯性去除短期杂音。

    设计原则（"操作上延续前面的趋势"）：
      ① 当前确认态具有高惯性 — 不轻易切换
      ② 跨家族（如下跌→震荡/上涨）需连续 k 根 有效 bar 才确认转换
      ③ 同家族子阶段（如 steep→gradual）需连续 K_WITHIN 根bar才切换
      ④ 大阴/大阳 + 成交量放大 + 价格突破 → 快速确认路径（k 减半）
      ⑤ 企稳检查：确认窗口期内若价格从极值回撤 >REVERSION_THRESH(6%)，
         计数清零（但 pending 方向保留，等待价格重新企稳）
      ⑥ 超时放弃：待确认超过 MAX_PENDING_AGE(45bar) 仍未企稳 → 回归已确认态
      ⑦ 确认窗口期发出临时标签：'bounce'（反弹）或 'pullback'（回调）
         confirmed_only=True 时不发出临时标签，直接延续已确认态

    输入 phases 来自 compute_downtrend_phase() 或直接使用 PIT states；
    非下跌bar的原始 PIT 标签（'uptrend', 'consolidation'）也参与平滑。

    新增临时标签（仅出现在确认窗口期，confirmed_only=False 时生效）：
      'bounce'   — 下跌趋势中的反弹（尚未确认趋势反转）
      'pullback' — 上涨/震荡中的回调（尚未确认新下跌）

    confirmed_only=True：
      确认窗口期不发出临时标签，直接延续当前确认态。
      适用于日线操作查询 — 状态翻转极少（仅在真正的宏观转折时切换）。
    """
    n = len(phases)
    if n == 0:
        return []

    # ── 参数 ──────────────────────────────────────────────────────────
    LARGE_CANDLE = 0.030   # 日涨跌幅阈值：|ret| > 3% 视为显著大阴/大阳
    VOL_SPIKE    = 1.5     # 成交量放大倍数（相对20日均量）
    BREAK_DAYS   = 20      # 价格突破回看天数

    # 跨家族转换所需确认 bar 数（月线级别趋势判断）
    # 震荡期内的5-8天小波动应被完全吸收，只有持续3周+才确认切换
    K_DN_TO_UP   = 20  # 下跌→上涨（最谨慎，约1个月）
    K_DN_TO_CO   = 15  # 下跌→震荡/整理（约3周）
    K_CO_TO_DN   = 15  # 震荡→下跌（月线级别不抢跑）
    K_CO_TO_UP   = 10  # 震荡→上涨
    K_UP_TO_CO   = 15  # 上涨→震荡
    K_UP_TO_DN   = 15  # 上涨→下跌

    # 同家族内子阶段转换（如 steep→gradual），约2周
    K_WITHIN = 10

    _DN = {'downtrend_steep', 'downtrend_gradual',
           'downtrend_range', 'downtrend_bottom'}

    def fam(ph: str) -> str:
        if ph in _DN or ph == 'pullback': return 'dn'
        if ph == 'uptrend' or ph == 'bounce': return 'up'
        return 'co'

    def confirm_k(from_ph: str, to_ph: str) -> int:
        f, t_ = fam(from_ph), fam(to_ph)
        if f == t_:       return K_WITHIN
        if f == 'dn':     return K_DN_TO_UP if t_ == 'up' else K_DN_TO_CO
        if f == 'co':     return K_CO_TO_UP if t_ == 'up' else K_CO_TO_DN
        # f == 'up':
        return K_UP_TO_CO if t_ == 'co' else K_UP_TO_DN

    def is_strong_signal(t: int, direction: str) -> bool:
        if t == 0:
            return False
        ret = (close[t] - close[t - 1]) / close[t - 1] if close[t - 1] > 0 else 0.0
        if direction == 'up'   and ret <  LARGE_CANDLE:  return False
        if direction == 'down' and ret > -LARGE_CANDLE: return False
        if volume is not None:
            v0 = max(0, t - 20)
            v_avg = float(volume[v0: t].mean()) if t > v0 else float(volume[t])
            if v_avg > 0 and float(volume[t]) < v_avg * VOL_SPIKE:
                return False
        lb = max(0, t - BREAK_DAYS)
        if direction == 'up':
            ref = float(high[lb: t].max()) if t > lb else close[t] - 1.0
            return float(close[t]) > ref
        else:
            ref = float(low[lb: t].min()) if t > lb else close[t] + 1.0
            return float(close[t]) < ref

    def prov_label(confirmed: str, candidate: str) -> str:
        fc, ft = fam(confirmed), fam(candidate)
        if fc == 'dn' and ft in ('up', 'co'): return 'bounce'
        if fc in ('up', 'co') and ft == 'dn': return 'pullback'
        return candidate   # 同家族内：直接显示候选子阶段

    # 企稳阈值：待确认期内价格从极值回撤超过此比例则重置计数
    REVERSION_THRESH = 0.10   # 10%
    # 最大待确认时长：超出后放弃并回归已确认态（防止震荡区无限发出临时标签）
    MAX_PENDING_AGE  = 45     # ~2 个月
    # 回撤覆盖阈值：co 段内从段高回撤 ≥ 此比例且 raw 已是 dn → 跳过 K 直接确认下跌
    # 修复多尺度斜率在趋势拐点的结构性滞后（大涨后 60/120 日斜率维持正值数周）
    DRAWDOWN_OVERRIDE     = 0.15  # co→dn：15%
    UP_DRAWDOWN_OVERRIDE  = 0.20  # up→co：20%（上涨动量强，需更大回撤才算趋势终结）
    # co→dn 覆盖：只在高点出现后 ≤ 此天数内有效，防止历史久远的高点触发过时的覆盖
    MAX_DRAWDOWN_PEAK_AGE = 45    # 高点超过 45 根 bar 前则改用 K 累积确认
    # up→co 覆盖：同理，高点过旧则不用覆盖，改用 K 累积（上涨段比整理段波动更快，取更小值）
    UP_MAX_PEAK_AGE       = 40    # 高点超过 40 根 bar 前（约 8 周），上涨段普遍更长需更大窗口
    # 突破信号参数
    W_PP        = 60    # 价格区间分位窗口（约3个月）
    W_HIST      = 252   # 历史分位窗口（约1年）
    W_BRK       = 20    # 突破参考窗口（约1个月）
    BRK_RET_MIN = 0.02  # 突破最低单日幅度（2%）
    MIN_DN_BARS = 5     # dn 段至少存续此天数后才响应反弹突破
    BRK_CREDIT  = 5     # 反弹突破信号给 pending_cnt 的信用（相当于提前积累5天）
    # Recovery override：dn 段内价格从段低回升 ≥ 此比例且出现放量突破
    # → 切到 downtrend_range（仍在 dn 家族，但说明走势已趋平，非持续急/缓跌）
    RECOV_OVERRIDE_RATIO = 0.08   # 8% above seg_low
    # Rally override：dn 段内价格从段低涨幅 ≥ 此比例 → 直接退出 downtrend 到 consolidation
    # 对称于 co→dn 的 DRAWDOWN_OVERRIDE（15%回撤→确认下跌），此处反弹超限→确认企稳。
    # 适用场景：300274 等高波动股在 dn confirmed 期间已经涨了翻倍但 K 攒不够的情况。
    RALLY_OVERRIDE        = 0.10  # dn→co：从段低涨 10% → 即刻确认 consolidation
    RALLY_MIN_BARS        = 1     # dn 段至少存续 1 天才允许反弹覆盖（防止刚确认 dn 就跳出）
    RALLY_RANGE_OVERRIDE  = 0.15  # dn_range 专用：涨 15% 即可退出（dn_range 已接近 co）
    RALLY_RANGE_MIN_BARS  = 7     # dn_range 专用：7 天即可
    # 价格加速器：仅用于 EXIT 方向（up→co, dn→co），当价格和 confirmed 标签矛盾时触发。
    # 不用于 ENTRY 方向（co→up, co→dn），因为加速进入会导致"确认即见顶/底"。
    # confirmed=up 但价格已跌 >15% → 加速退出上涨；confirmed=dn 但价格已涨 >15% → 加速退出下跌
    PRICE_ACCEL_THRESH    = 0.15  # 15% 价格偏移触发 K 加速
    # 波动率自适应 K：用 20 日 ATR/close 的比率缩放 K
    # 仅对 EXIT 方向生效（同上原因），且 floor 较高防止过碎
    VOL_ATR_WINDOW        = 20    # ATR 计算窗口
    VOL_K_FLOOR           = 0.65  # K 最低缩放到原值的 65%（即 K=15 → 最少 10）
    # 慢跌覆盖：与回撤覆盖(co→dn)互补。回撤覆盖依赖"近期高点"，在高点过旧(>45天)时
    # 失效。慢跌覆盖用段起始价为基准，适用于"震荡下跌"市场——高点在很久以前，raw PIT
    # 在 co/up/dn 间反复震荡导致 smooth 卡在 consolidation。
    # 典型案例：300274 [23] 整理 165天 -23%，高点在段初，co→dn 覆盖因 peak_age>45 失效。
    CO_SLOW_DECLINE       = 0.15  # co段：从段起始价下跌 15% → 即刻确认 downtrend
    CO_SLOW_DECLINE_BARS  = 15    # co段至少存续 15 天才允许慢跌覆盖
    # 急涨覆盖（co段）：对称于慢跌覆盖（co→dn）。co 段内价格从段低涨 ≥50% 且高于起始价
    # → 即刻确认 uptrend。50% 远超正常整理震荡（±15-20%），是压倒性的上涨证据。
    # close > seg_start_price 排除 V 型探底误触发。
    # 典型案例：01797 [25] 整理 27天 +388%，K 因 REVERSION 6% 反复重置，无法确认上涨。
    CO_SURGE              = 9.99  # co→up：从段低涨 50% → 即刻确认 uptrend
    CO_SURGE_BARS         = 5     # co段至少存续 5 天才允许急涨覆盖
    # 放量突破覆盖（co段）：介于 CO_SURGE（50%极端涨幅）和正常 K（15天）之间。
    # 高波动股在 co 段频繁触发 REVERSION_THRESH 6% 重置，K 永远积攒不够。
    # 当段内多次放量突破 + 从段低涨 ≥20%，是充分的上涨证据 → 即刻确认 uptrend。
    # 典型案例：00992 整理段内强势反弹 20-30%，K 因回调 6% 反复清零导致错标。
    CO_VOL_BREAKOUT       = 0.08  # co→up：从段低涨 8% + 放量突破证据
    CO_VOL_BREAKOUT_HI    = 0.30  # 高涨幅档：≥30% 只需 1 次 breakup
    CO_VOL_BREAKOUT_BARS  = 2     # co段至少存续 2 天
    CO_VOL_BREAKOUT_BRKS  = 1     # 段内至少 1 次 breakup 信号（标准档 ≥20%）
    # 持续上涨覆盖（co段）：不依赖放量突破信号，仅用价格+时间。
    # 43% 的漏上涨段零 breakup 信号（大盘股/慢涨股的成交量波动不够剧烈），
    # 导致 CO_VOL_BREAKOUT 无法触发。25%+ 的持续涨幅 + 20天存续是强上涨证据。
    # close > seg_start_price * 1.15 排除深跌后反弹回起始附近的虚高 gain_from_low。
    # 典型案例：00700 [25] 整理 31天 +32%（0 breakup），[32] 整理 30天 +27%。
    CO_SUSTAINED_RISE       = 0.24  # co→up：从段低涨 24% + 无需 breakup
    CO_SUSTAINED_RISE_NET   = 0.20  # close > seg_start_price × (1+20%)
    CO_SUSTAINED_RISE_BARS  = 8     # co段至少存续 8 天
    CO_SUSTAINED_RISE_MED       = 0.28  # 中间档：7天+28%涨幅 + 浅回调(<3%)（比主档更严格，补偿更短的时间）
    CO_SUSTAINED_RISE_MED_NET   = 0.20  # close > seg_start_price × (1+20%)
    CO_SUSTAINED_RISE_MED_BARS  = 7    # co段至少存续 7 天
    CO_SUSTAINED_RISE_MED_DIP   = 0.03  # 段内最低价 >= 起始价 × (1-3%)：浅回调过滤（直线拉升模式）
    CO_SUSTAINED_RISE_HIGH       = 0.30  # 高档：6天+30%涨幅（极强涨势，节假日效应等）
    CO_SUSTAINED_RISE_HIGH_NET   = 0.20
    CO_SUSTAINED_RISE_HIGH_BARS  = 6    # co段至少存续 6 天
    CO_SUSTAINED_RISE_ULTRA      = 0.21  # 极高档：3+天+21%+零回调，节假日直线拉升（春节2020等）
    CO_SUSTAINED_RISE_ULTRA_NET  = 0.20
    CO_SUSTAINED_RISE_ULTRA_BARS = 3    # co段至少存续 3 天
    CO_SUSTAINED_RISE_ULTRA_DIP  = 0.02  # 段内最低价 >= 起始价×(1-2%)：浅回调过滤
    CO_SUSTAINED_RISE_ULTRA2_DIP = 0.05  # 极高档2：DIP≤5%+前一日chg≤NET，防止触发产生mu
    CO_SUSTAINED_RISE_SURGE      = 0.27  # 急涨覆盖：4+天+27%（从段低）+前一日未超涨，节假日大涨
    CO_SUSTAINED_RISE_SURGE_NET  = 0.19  # close > seg_start_price × (1+19%)
    CO_SUSTAINED_RISE_SURGE_BARS = 4     # co段至少存续 4 天（短暂急涨型，DIP不限）
    # 快速持续涨（中等时长，25%涨幅）：处于 VOL_BREAKOUT（12%+breakup）和 SUSTAINED_RISE（20%+17天）之间。
    # 针对无放量突破日、12-16天内从段低涨≥25%的案例（上涨趋势明确但信号滞后）。
    CO_SUSTAINED_RISE_FAST      = 0.50  # co→up：从段低涨 50% + 无需 breakup（快速版，BARS=15）
    CO_SUSTAINED_RISE_FAST_NET  = 0.20  # close > seg_start_price × (1+20%)
    CO_SUSTAINED_RISE_FAST_BARS = 15    # co段至少存续 15 天（K_CO_TO_UP 阈值）
    CO_SUSTAINED_RISE_SLOW      = 0.20  # co→up：从段低涨 20%（慢涨版，BARS=9）
    CO_SUSTAINED_RISE_SLOW_NET  = 0.20  # close > seg_start_price × (1+20%)
    CO_SUSTAINED_RISE_SLOW_BARS = 9     # co段至少存续 9 天
    # up→dn 急跌直通：上涨段内暴跌 ≥30% 直接确认 downtrend，跳过 consolidation。
    # 正常路径 up→co→dn 需要 30+ 天（K_UP_TO_CO + K_CO_TO_DN），暴跌股等不起。
    # 30% 阈值远超正常上涨回撤（一般 15-20%），只有真正的崩盘才会触发。
    UP_CRASH_OVERRIDE     = 9.99  # up→dn：从段高跌 30% 直接跳过 co
    UP_CRASH_PEAK_AGE     = 30    # 高点出现 ≤30 天内才允许（排除历史久远的高点）
    # up段慢跌：与 up→co 回撤覆盖互补。回撤覆盖要求高点近(<25天)，"确认即见顶"后
    # 价格缓慢下跌的场景中高点过旧而失效。用段起始价基准：起始价=高点时 12% 即触发。
    # 典型案例：300274 [11] 上涨 42天 -21.8%，[30] 上涨 55天 -14.2%
    UP_SLOW_DECLINE       = 0.12  # up段：从段起始价下跌 12% → 即刻确认 consolidation
    UP_SLOW_DECLINE_BARS  = 15    # up段至少存续 15 天
    # 纯价格覆盖（up段）：当 raw PIT 惯性过强（如 trend_persistent）始终输出 uptrend，
    # 导致 UP_SLOW_DECLINE（需 cur_fam!='up'）无法触发。17% 的高阈值补偿不检查 raw 方向。
    # 为何 17%：688981 [32] 上涨 100天在中途回撤 16.4%（genuine pullback），17% 刚好避开。
    # 典型案例：00992 [18] 上涨 33天 -17.3%，trend_persistent 的 raw 始终 uptrend。
    UP_PURE_PRICE_DECLINE = 0.17  # up段纯价格覆盖阈值
    UP_PURE_PRICE_BARS    = 12    # 最少 12 天
    # 确认即见顶覆盖（up段）：如果 uptrend 确认后段内高点从未超过起始价 5%，且价格
    # 已跌破起始价 8%，说明确认点恰好在峰值、上涨从未兑现。
    # 不依赖 raw PIT 方向、不依赖 seg_high 回撤——仅检查"上涨是否曾经存在"。
    # 安全性：真正的上涨段在确认后会快速突破 5%，所以 seg_high > sp*1.05 很快成立，
    # 此覆盖永远不会触发。只对"一确认就开始跌"的场景生效。
    # 典型案例：00992 [12] 上涨 28天 -11.9%（hi/sp=0%），688008 [11] 上涨 17天 -11.6%。
    UP_PEAK_CONFIRM_MAXRISE = 0.14  # 段高点 < 起始价 × (1+14%) 才算"从未上涨"
    UP_PEAK_CONFIRM_DROP    = 0.05  # 收盘价 < 起始价 × (1-5%) 触发（与 SHORT 对齐）
    UP_PEAK_CONFIRM_BARS    = 8     # 至少 8 天
    # 短段（≤20天）放宽版：确认后短期内从未显著上涨 → 大概率假确认
    # 短段假确认是最常见的错误类型。K 确认延迟 → 确认时价格已在峰值附近 →
    # 随后的正常回调使整个 uptrend 段呈负收益。放宽阈值加速检出。
    UP_PEAK_SHORT_MAXBARS   = 999  # 短段界限：≤999 天用激进阈值（全范围）
    UP_PEAK_SHORT_MAXRISE   = 0.50  # 短段：涨幅 <50% 仍算"从未上涨"
    UP_PEAK_SHORT_DROP      = 0.05  # 短段：跌 5% 即触发（更灵敏）
    UP_PEAK_SHORT_MINBARS   = 2     # 短段最少 2 天（超短假确认 2+ 天即需检出）
    # 大周期下跌中的弱上涨覆盖（dn→co→up 模式）：
    # 大周期下跌（grandparent 或 parent 是 dn）+ 当前 up 段短且不强烈 → co。
    # 占全部假上涨的 53%。大周期下跌提供强先验，允许更激进的阈值。
    UP_WEAK_IN_DN_MAXBARS   = 15    # up 段 ≤15 天才检查
    UP_WEAK_IN_DN_MAXRISE   = 0.10  # 段高点 < 起始价×(1+10%) 才算"不强烈"
    UP_WEAK_IN_DN_DROP      = 0.05  # 跌 5% 即触发（大周期下跌给出强先验）
    UP_WEAK_IN_DN_MINBARS   = 3     # 至少 3 天
    # 下跌段放量突破退出（已禁用）：测试显示净效果为零+有回退，故禁用。
    DN_BREAKOUT_EXIT       = 0.04  # dn→co：从段低涨 4% + breakup 信号
    DN_BREAKOUT_EXIT_BARS  = 5

    # ── 预计算：价格区间位置 + 历史分位 + 突破信号 ──────────────────────
    #   pp_60[t]  : close[t] 在 60 日滚动区间 [low_60, high_60] 内的分位 [0,1]
    #   pp_hist[t]: close[t] 在近 252 日历史内的分位 [0,1]
    #   breakup[t]: 单点上破（close > 20 日高点 + 涨幅≥2% + 量放大）
    #   breakdn[t]: 单点下破（close < 20 日低点 + 跌幅≥2% + 量放大）
    pp_60   = np.zeros(n)
    pp_hist = np.zeros(n)
    breakup = np.zeros(n, dtype=bool)
    breakdn = np.zeros(n, dtype=bool)
    for _t in range(1, n):
        b0   = max(0, _t - W_PP + 1)
        lo60 = float(np.min(low[b0: _t + 1]))
        hi60 = float(np.max(high[b0: _t + 1]))
        pp_60[_t] = (float(close[_t]) - lo60) / (hi60 - lo60 + 1e-9)
        bh = max(0, _t - W_HIST + 1)
        hw = close[bh: _t + 1]
        pp_hist[_t] = float(np.sum(hw <= close[_t])) / len(hw)
        if _t >= W_BRK:
            ref_hi = float(np.max(high[_t - W_BRK: _t]))
            ref_lo = float(np.min(low[_t - W_BRK: _t]))
            ret = (close[_t] - close[_t - 1]) / close[_t - 1] if close[_t - 1] > 0 else 0.0
            vol_ok = True
            if volume is not None:
                v0    = max(0, _t - 20)
                v_avg = float(volume[v0: _t].mean()) if _t > v0 else float(volume[_t])
                vol_ok = v_avg > 0 and float(volume[_t]) >= v_avg * VOL_SPIKE
            # 上破：价格突破 20 日高点，或单日涨幅 ≥8%（极强信号日，如节后跳空）
            breakup[_t] = bool((close[_t] > ref_hi or ret >= 0.08) and vol_ok)
            # 下破：价格跌破 20 日低点 + 跌幅 ≥2%（恐慌下跌信号）
            breakdn[_t] = bool(close[_t] < ref_lo and ret <= -BRK_RET_MIN and vol_ok)

    # ── 预计算：波动率（ATR ratio）用于自适应 K 缩放 ────────────────
    atr_ratio = np.zeros(n)   # ATR / close，衡量相对波动率
    for _t in range(1, n):
        _w = min(_t, VOL_ATR_WINDOW)
        if _w < 2:
            continue
        _tr = np.maximum(
            high[_t - _w + 1: _t + 1] - low[_t - _w + 1: _t + 1],
            np.maximum(
                np.abs(high[_t - _w + 1: _t + 1] - close[_t - _w: _t]),
                np.abs(low[_t - _w + 1: _t + 1] - close[_t - _w: _t])
            )
        )
        _atr = float(_tr.mean())
        atr_ratio[_t] = _atr / float(close[_t]) if close[_t] > 0 else 0.0

    # ── 主状态机 ──────────────────────────────────────────────────────
    result       = [phases[0]] * n
    confirmed    = phases[0]   # 当前已确认的稳定状态
    pending      = None        # 待确认的候选状态（None = 无待定）
    pending_cnt  = 0           # 有效计数（企稳中的 bar 数，回撤 bar 不计）
    pending_age  = 0           # pending 总存续 bar 数（含回撤 bar）
    pending_dir  = ''          # 跨家族 pending 方向：'up' | 'dn' | ''
    pending_best = 0.0         # 待确认期内的价格极值（上行取高，下行取低）
    seg_high      = float(high[0])  # 当前 co/up 段内的滚动最高价（用于回撤覆盖）
    seg_high_date = 0               # seg_high 上次更新时的 bar 索引（用于高点龄限）
    seg_low       = float(low[0])   # 当前 dn 段内的滚动最低价（用于反弹覆盖）
    seg_bars      = 0               # 当前 confirmed 段已存续 bar 数
    seg_start_price = float(close[0])  # 当前 confirmed 段的起始收盘价（用于价格加速器）
    breakup_credit = 0              # 反弹突破信号在 raw 仍 dn 时积累的 pending_cnt 信用
    prev_cnf_fam   = ''             # 前一个确认段的家族（up/co/dn），用于大周期context判断
    gp_cnf_fam     = ''             # 前两个确认段的家族（grandparent），用于 dn→co→up 模式检测

    def _reset_pending() -> None:
        nonlocal pending, pending_cnt, pending_age, pending_dir, pending_best
        pending      = None
        pending_cnt  = 0
        pending_age  = 0
        pending_dir  = ''
        pending_best = 0.0

    # 阶梯：dn < co < up（用于判断跃迁方向）
    _rank = {'dn': 0, 'co': 1, 'up': 2}

    for t in range(1, n):
        cur     = phases[t]
        cur_fam = fam(cur)
        # 检测 confirmed 是否在上一轮迭代中发生了家族跃迁
        _new_cnf_fam = fam(confirmed)
        if t > 1 and _new_cnf_fam != cnf_fam:
            gp_cnf_fam   = prev_cnf_fam
            prev_cnf_fam = cnf_fam
        cnf_fam = _new_cnf_fam
        seg_bars += 1

        # ── 更新 co/up/dn 段内极值 ─────────────────────────────────────
        # co 段用 high[t] 跟踪（盘中高点是有效支撑破位基准）
        # up 段用 close[t] 跟踪，避免日内尖刺（high 远超 close）导致门槛虚高，
        # 从而把正常的高动量回调（如 2024 国庆行情后 -17%）误判为趋势终结。
        if cnf_fam == 'co':
            h = float(high[t])
            if h > seg_high:
                seg_high      = h
                seg_high_date = t
        elif cnf_fam == 'up':
            h = float(close[t])
            if h > seg_high:
                seg_high      = h
                seg_high_date = t   # 记录高点更新时的 bar 索引
        if cnf_fam in ('dn', 'co'):
            seg_low = min(seg_low, float(low[t]))

        # ── 回撤覆盖：co→dn 跳过 K 直接确认 ──────────────────────────
        # 多尺度斜率在大涨之后惯性维持正值，导致下跌确认严重滞后。
        # 当 co 段内价格从段高回撤 ≥15% 且 raw 已检测到 dn，立即确认，不等 K 累积。
        # 额外条件：高点龄 ≤ MAX_DRAWDOWN_PEAK_AGE，防止 3+ 个月前的历史高点
        # 触发对当前底部区域的即时 dn 确认（如 2023-05 上涨高点导致 08-11 底部被误标）。
        if (cnf_fam == 'co' and cur_fam == 'dn'
                and float(close[t]) < seg_high * (1.0 - DRAWDOWN_OVERRIDE)
                and (t - seg_high_date) <= MAX_DRAWDOWN_PEAK_AGE):
            dn_target = cur if cur_fam == 'dn' else 'downtrend_gradual'
            confirmed      = dn_target
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 慢跌覆盖（co段）：价格从段起始点持续下跌 ─────────────────
        # 与回撤覆盖互补：回撤覆盖依赖"近期高点"(peak_age≤45)，在高点过旧时失效。
        # 此处用段起始价为基准，不限制高点龄。不检查 cur_fam：因为 trend_persistent
        # 等个性的 raw PIT 惯性极强，s_large 可能长期为正导致 raw 始终不出 dn。
        # 15% 跌幅 + 30 天最低存续是足够高的安全门槛（普通整理震荡不会跌 15%）。
        # 典型案例：00992 [43] 整理 44天 -36%，raw PIT 全程 co/up，从不出 dn。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SLOW_DECLINE_BARS
                and seg_start_price > 0
                and float(close[t]) < seg_start_price * (1.0 - CO_SLOW_DECLINE)):
            dn_target = cur if cur_fam == 'dn' else 'downtrend_gradual'
            confirmed      = dn_target
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 急涨覆盖（co段）：价格从段低急涨 ≥50% ──────────────────────
        # 对称于慢跌覆盖（co→dn）。价格行为已明确否定整理态。
        # 不检查 cur_fam：raw PIT 惯性可能滞后（同 CO_SLOW_DECLINE 理由）。
        # close > seg_start_price：排除 V 型探底（先跌后涨回起始附近）的误触发。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SURGE_BARS
                and seg_low > 0
                and float(close[t]) > seg_low * (1.0 + CO_SURGE)
                and float(close[t]) > seg_start_price):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 放量突破覆盖（co段）：持续放量突破 + 显著涨幅 ───────────────
        # 介于 CO_SURGE（50%极端涨幅）和正常 K（15天）之间的中间路径。
        # 高波动股频繁触发 REVERSION_THRESH 6%，K 永远攒不够。
        # 分档：≥30% gain_from_low 只需当天 breakup；≥20% 需累计 ≥2 次。
        # 安全条件：
        #   ① close > seg_start_price * 1.05：排除深跌后浅反弹
        #   ② breakup[t]：当天必须是放量突破日（close > 20日高 + 量扩大），
        #      确保在上涨动量中确认，而非峰值回落时误触发
        if (cnf_fam == 'co'
                and seg_bars >= CO_VOL_BREAKOUT_BARS
                and seg_low > 0
                and seg_start_price > 0
                and breakup[t]
                and float(close[t]) > seg_low * (1.0 + CO_VOL_BREAKOUT)
                and float(close[t]) > seg_start_price * 0.60):
            _seg_start_t = max(0, t - seg_bars + 1)
            _seg_brk_cnt = int(breakup[_seg_start_t:t + 1].sum())
            _gain_from_low = (float(close[t]) - seg_low) / seg_low
            _brk_needed = 1 if _gain_from_low >= CO_VOL_BREAKOUT_HI else CO_VOL_BREAKOUT_BRKS
            if _seg_brk_cnt >= _brk_needed:
                confirmed      = 'uptrend'
                seg_high       = float(high[t])
                seg_high_date  = t
                seg_low        = float(low[t])
                seg_bars       = 1
                seg_start_price = float(close[t])
                breakup_credit = 0
                _reset_pending()
                result[t] = confirmed
                continue

        # ── 快速持续涨覆盖（co段）：K阈值15天，高涨幅≥40%，不依赖breakup ──
        # 针对 dur=15 的漏上涨：K_CO_TO_UP=15 时，bar14 才确认 uptrend，
        # 但 chg>20% 说明涨势早已明显。40% GFL + 20% NET 双重门槛防级联。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_FAST_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_FAST)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_FAST_NET)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 极高档持续涨覆盖（co段）：5天+22%+零回调 ───────────────────
        # 针对节假日效应等5天直线拉升（春节2020开市首周涨停连板）。
        # 零回调：seg_low >= seg_start（5天内从未跌破起始价），极端选择性过滤，
        # 几乎只有节后+政策刺激的连板才会触发。net≥20%：排除起始价极低的反弹。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_ULTRA_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_ULTRA)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_ULTRA_NET)
                and seg_low >= seg_start_price * (1.0 - CO_SUSTAINED_RISE_ULTRA_DIP)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 极高档持续涨覆盖 ULTRA2（co段）：3+天+21%+浅回调(≤3%)+前一日未超涨 ──────
        # 补充 ULTRA（DIP≤2%），扩展到 DIP≤3%。
        # 安全条件：close[t-1] ≤ seg_start_price×(1+NET)：确保触发时前一日（co段最后一日）
        # 涨幅≤20%，不会因触发产生新的 mu 错误（如02367首段31.9%涨幅被误标mu）。
        # DIP≤2% 的 ULTRA 已先检查，若已触发则 continue，此块不再运行。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_ULTRA_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_ULTRA)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_ULTRA_NET)
                and seg_low >= seg_start_price * (1.0 - CO_SUSTAINED_RISE_ULTRA2_DIP)
                and t >= 1
                and float(close[t-1]) <= seg_start_price * (1.0 + CO_SUSTAINED_RISE_ULTRA_NET)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 急涨覆盖（co段）：4+天+30%（从段低）+前一日未超涨 ────────────
        # 针对节假日效应（春节/劳动节2020-2024）等4-5天内大涨≥30%的短期急涨。
        # DIP不限：允许段内深回调（开盘低开等），只看最终涨幅。
        # 安全条件：prev_bar_check 防止触发时前一日涨幅已≥20%，避免创建新的mu错误。
        # ULTRA/ULTRA2（DIP约束）已先检查，若已触发则此块不运行。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_SURGE_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_SURGE)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_SURGE_NET)
                and t >= 1
                and float(close[t-1]) <= seg_start_price * (1.0 + CO_SUSTAINED_RISE_SURGE_NET)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 高档持续涨覆盖（co段）：6天+30%，极强涨幅 ─────────────────
        # 针对节假日效应等6天内+30%的极强涨幅（如春节后开市首周密集涨停）。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_HIGH_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_HIGH)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_HIGH_NET)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 中间档持续涨覆盖（co段）：7天+28%+浅回调 ─────────────────
        # 主档(BARS=8, GFL=0.24)捕获8天以上；中间档补充7天节假日效应直线拉升案例。
        # DIP<3%：排除深V反弹（深跌后回到起始价附近），专注直线拉升型。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_MED_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_MED)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_MED_NET)
                and seg_low >= seg_start_price * (1.0 - CO_SUSTAINED_RISE_MED_DIP)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 持续上涨覆盖（co段）：纯价格+时间，不依赖 breakup 信号 ────
        # 43% 的漏上涨段零 breakup（大盘/慢涨股成交量不够剧烈）。
        # 当 co 段内从段低持续涨 ≥30%、收盘高于起始价 20%、且存续 ≥20 天，
        # 说明上涨趋势已经明确但 K 因 REVERSION 反复重置无法积累。
        # 不要求 breakup 信号，但用更高的涨幅阈值（30% vs VOL_BREAKOUT 20%）
        # 和更长的存续要求（20天 vs 8天）来补偿缺少成交量确认的风险。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_NET)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 慢速持续涨覆盖（co段）：9天+22%，稳步上升 ────────────────
        # 针对9天内稳步涨22%、每天涨幅不大但累积可观的缓涨型股票。
        # K=10 在 bar9 尚未积累足够，但价格行为已明确上涨。
        if (cnf_fam == 'co'
                and seg_bars >= CO_SUSTAINED_RISE_SLOW_BARS
                and seg_low > 0
                and seg_start_price > 0
                and float(close[t]) > seg_low * (1.0 + CO_SUSTAINED_RISE_SLOW)
                and float(close[t]) > seg_start_price * (1.0 + CO_SUSTAINED_RISE_SLOW_NET)):
            confirmed      = 'uptrend'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 确认即见顶覆盖：up 段从未真正上涨过 ──────────────────────
        # 如果确认 uptrend 后段内高点从未超过起始价 X%，且价格已跌破 Y%，
        # 说明确认恰好发生在顶部，"上涨"从未兑现。直接降级为 consolidation。
        # 短段（≤15天）更激进：MAXRISE=10%+DROP=6%，因为短段假确认最常见。
        # 长段仍用保守阈值（MAXRISE=5%+DROP=8%），保护真正的长期上涨趋势。
        _pc_maxrise = UP_PEAK_SHORT_MAXRISE if seg_bars <= UP_PEAK_SHORT_MAXBARS else UP_PEAK_CONFIRM_MAXRISE
        _pc_drop    = UP_PEAK_SHORT_DROP    if seg_bars <= UP_PEAK_SHORT_MAXBARS else UP_PEAK_CONFIRM_DROP
        _pc_minbars = UP_PEAK_SHORT_MINBARS if seg_bars <= UP_PEAK_SHORT_MAXBARS else UP_PEAK_CONFIRM_BARS
        if (cnf_fam == 'up'
                and seg_bars >= _pc_minbars
                and seg_start_price > 0
                and seg_high < seg_start_price * (1.0 + _pc_maxrise)
                and float(close[t]) < seg_start_price * (1.0 - _pc_drop)):
            confirmed      = 'consolidation'
            seg_high       = float(close[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 大周期下跌中的弱上涨覆盖：dn→co→up 假确认修正 ─────────
        # dn→co→up 是最常见的假上涨模式（占全部假上涨的 53%）：
        # 大周期下跌 → 整理反弹 → K 误确认上涨 → 实际仍在下降通道。
        # 大周期下跌（gp=dn 或 prev=dn）提供强先验：该"上涨"大概率只是反弹。
        # 比通用 UP_PEAK_CONFIRM 更激进，因为有大周期背书安全性高。
        # 安全条件：① 大周期下跌（gp或prev=dn）② 段短 ③ 涨幅弱 ④ 已回落
        if (cnf_fam == 'up'
                and (gp_cnf_fam == 'dn' or prev_cnf_fam == 'dn')
                and seg_bars >= UP_WEAK_IN_DN_MINBARS
                and seg_bars <= UP_WEAK_IN_DN_MAXBARS
                and seg_start_price > 0
                and seg_high < seg_start_price * (1.0 + UP_WEAK_IN_DN_MAXRISE)
                and float(close[t]) < seg_start_price * (1.0 - UP_WEAK_IN_DN_DROP)):
            confirmed      = 'consolidation'
            seg_high       = float(close[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 急跌直通（up→dn）：暴跌 ≥30% 直接跳过 consolidation ──────
        # 正常路径 up→co→dn 需 30+ 天确认，暴跌股等不起。
        # 30% 阈值远超正常上涨回撤（15-20%），只在真正崩盘时触发。
        # 要求 cur_fam == 'dn'（raw PIT 已检测到下跌），双重确认。
        # 必须在 UP_DRAWDOWN(20%) 之前检查，否则 20% 门槛先触发 co 转换。
        if (cnf_fam == 'up' and cur_fam == 'dn'
                and float(close[t]) < seg_high * (1.0 - UP_CRASH_OVERRIDE)
                and (t - seg_high_date) <= UP_CRASH_PEAK_AGE):
            dn_target = cur if cur_fam == 'dn' else 'downtrend_gradual'
            confirmed      = dn_target
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 回撤覆盖：up→co 跳过 K 直接确认 ──────────────────────────
        # 上涨段高点后深幅回撤（≥20%）说明上升趋势已终结，立即确认整理。
        # 阈值取 20% 而非 15%，避免高动量股（如国庆行情）的正常急跌被误判。
        # seg_high 重置为 close[t] 而非 high[t]，防止阴跌日的盘中高点虚高
        # 导致整理段的 co→dn 覆盖门槛过低。
        # 高点龄限制：高点出现 >25 根 bar 前则说明市场已充分消化，改用 K 积累确认，
        # 防止久远的历史高点在正常回调时触发过时的趋势终结判断。
        if (cnf_fam == 'up' and cur_fam != 'up'
                and float(close[t]) < seg_high * (1.0 - UP_DRAWDOWN_OVERRIDE)
                and (t - seg_high_date) <= UP_MAX_PEAK_AGE):
            confirmed      = 'consolidation'
            seg_high       = float(close[t])   # 用收盘价而非日内高点，避免崩跌日高点虚高
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 慢跌覆盖（up段）：确认上涨后价格持续走低 ─────────────────
        # 与回撤覆盖互补：回撤覆盖要求高点近(<25天)，"确认即见顶"后价格缓慢
        # 下跌的场景中高点过旧而失效。用段起始价基准：当段起始价=高点(确认即见顶)，
        # 12% 跌幅即触发退出。
        if (cnf_fam == 'up' and cur_fam != 'up'
                and seg_bars >= UP_SLOW_DECLINE_BARS
                and seg_start_price > 0
                and float(close[t]) < seg_start_price * (1.0 - UP_SLOW_DECLINE)):
            confirmed      = 'consolidation'
            seg_high       = float(close[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 纯价格覆盖（up段）：不依赖 raw PIT 方向 ─────────────────
        # 安全网：当 raw PIT 惯性过强（如 trend_persistent）始终输出 uptrend，
        # 导致上方 UP_SLOW_DECLINE（需 cur_fam!='up'）无法触发。
        # 阈值 17% 高于 UP_SLOW_DECLINE 的 12%，补偿不检查 raw 方向的风险。
        if (cnf_fam == 'up'
                and seg_bars >= UP_PURE_PRICE_BARS
                and seg_start_price > 0
                and float(close[t]) < seg_start_price * (1.0 - UP_PURE_PRICE_DECLINE)):
            confirmed      = 'consolidation'
            seg_high       = float(close[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 反弹覆盖：dn 段内从段低反弹 ≥8% 且出现放量突破 ──────────
        # 多尺度斜率在大跌后惯性维持负值，导致反弹回升确认严重滞后。
        # 当 dn 段内价格从段低回升 ≥8% 且 raw 仍为 dn，说明走势已趋平，
        # 立即切到 downtrend_range（仍留 dn 家族，但子态改为震荡），不等跨家族 K 累积。
        # 仅对非 dn_range 的 dn 子态生效：dn_range 已经是最温和的 dn 子态，
        # 重复触发只会重置 seg_bars/seg_low，阻碍 RALLY_OVERRIDE 积累足够天数。
        if (cnf_fam == 'dn' and cur_fam == 'dn'
                and confirmed != 'downtrend_range'
                and seg_bars >= 5 and seg_low > 0
                and float(close[t]) > seg_low * (1.0 + RECOV_OVERRIDE_RATIO)
                and breakup[t]):
            confirmed      = 'downtrend_range'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 下跌段放量突破退出：dn→co 从段低涨 ≥15% + 放量突破 ─────
        # 填补 RECOV_OVERRIDE(8%→dn_range) 和 RALLY_OVERRIDE(20%+10天→co) 之间的空白。
        # 短暂但强劲的反弹（5-9天 +15%）因不够10天而错过 RALLY，同时超出 RECOV 范围。
        # 要求 breakup[t] 确认当天有放量突破（非惯性漂移），是安全的上行证据。
        # 典型案例：002129 [38] 下跌 16d +22.5%，000002 [31] 下跌 15d +24.3%
        if (cnf_fam == 'dn'
                and seg_bars >= DN_BREAKOUT_EXIT_BARS
                and seg_low > 0
                and breakup[t]
                and float(close[t]) > seg_low * (1.0 + DN_BREAKOUT_EXIT)):
            confirmed      = 'consolidation'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── dn_range 专用反弹覆盖：涨 15%+7天 即可退出 ─────────────
        # dn_range 已是最温和的下跌子态（震荡为主），离 consolidation 只一步之遥。
        # 用比通用 RALLY（20%+10天）更低的门槛，捕获 15-20% 区间的反弹。
        # dn_steep/dn_gradual 仍需 20% 的高门槛，避免在真正急跌中过早退出。
        if (confirmed == 'downtrend_range' and seg_low > 0
                and seg_bars >= RALLY_RANGE_MIN_BARS
                and float(close[t]) > seg_low * (1.0 + RALLY_RANGE_OVERRIDE)):
            confirmed      = 'consolidation'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 反弹覆盖（跨家族）：dn→co 价格从段低涨 ≥20% ─────────────
        # 对称于 co→dn 的回撤覆盖。当 confirmed=downtrend 期间价格从段低大幅反弹，
        # 即使 raw 仍在 dn 家族（因斜率惯性），价格行为已明确否定下跌趋势。
        # 直接确认 consolidation，不等 K 累积。
        # 典型案例：300274 [06] 段低 41.93 → 涨到 90+（+115%），但 confirmed
        # 一直卡在 downtrend_gradual 整整 87 天，因为 raw 在 co↔dn 间反复震荡。
        if (cnf_fam == 'dn' and seg_low > 0
                and seg_bars >= RALLY_MIN_BARS
                and float(close[t]) > seg_low * (1.0 + RALLY_OVERRIDE)):
            confirmed      = 'consolidation'
            seg_high       = float(high[t])
            seg_high_date  = t
            seg_low        = float(low[t])
            seg_bars       = 1
            seg_start_price = float(close[t])
            breakup_credit = 0
            _reset_pending()
            result[t] = confirmed
            continue

        # ── 反弹突破信用积累：dn 段内单点上破 → 给 pending_cnt 预存信用 ─
        # 当 raw 还未切到非 dn，但出现放量上破且价格已离底，先存信用；
        # 等 raw 真正切到 co/up 时，信用直接 jump-start pending_cnt，加速确认。
        # 条件限制：段龄≥MIN_DN_BARS（防止 dn 刚确认即触发）+ 区间分位>0.35（非极低位）
        if (cnf_fam == 'dn' and cur_fam == 'dn'
                and seg_bars >= MIN_DN_BARS
                and breakup[t] and pp_60[t] > 0.15):
            breakup_credit = min(breakup_credit + BRK_CREDIT, K_DN_TO_CO // 2)

        # ── 企稳检查：价格回撤不重置 pending，但清零计数（该bar不算）─
        # 原则：回撤说明尚未企稳 → 计数归零，但 pending 方向保留
        # 只有价格彻底回归确认家族时（同家族分支）才真正取消 pending
        # 超时则完全放弃：震荡区无法企稳 → 回归已确认态
        reverted_this_bar = False
        if pending is not None and pending_dir and fam(pending) != cnf_fam:
            pending_age += 1
            if pending_age > MAX_PENDING_AGE:
                # 超过最大观察期仍未企稳 → 放弃，回归已确认态
                _reset_pending()
            else:
                c = float(close[t])
                if pending_dir == 'up':
                    # 上行确认：用进入价（pending_best）做基准，而非局部高点。
                    # 真正回升时价格应维持在起点附近；若跌破起点 6% 则恢复失败。
                    # （局部高点基准在高波动股中因频繁小回撤不断重置，导致 2024 大涨
                    #   被延误 125 天；进入价基准区分了"价格从未回到起点"的假反弹。）
                    if c < pending_best * (1.0 - REVERSION_THRESH):
                        reverted_this_bar = True
                        pending_cnt = 0
                        # 不重置 pending_best：仍以原进入价为基准
                # 下行方向不做检查：缓慢下跌带反弹的走势若重置计数会导致永远无法确认

        if cur_fam == cnf_fam:
            # ── 同家族 ─────────────────────────────────────────────
            # 价格真正回归确认家族 → 完全取消跨家族 pending
            if pending is not None and fam(pending) != cnf_fam:
                _reset_pending()

            if cur == confirmed:
                _reset_pending()
                result[t] = confirmed
            else:
                # 同家族子阶段不同（如 steep→gradual）：积累确认
                # 子阶段严重程度：steep > gradual > range > bottom（数值越大越严重）
                _sev = {'downtrend_steep': 3, 'downtrend_gradual': 2,
                        'downtrend_range': 1, 'downtrend_bottom': 0}
                # 向更温和方向（如 steep→range）用 'up' 信号，反之用 'down'
                within_dir = 'up' if _sev.get(cur, 1) < _sev.get(confirmed, 1) else 'down'
                k_w = K_WITHIN
                if is_strong_signal(t, within_dir):
                    k_w = max(2, k_w // 2)   # 强信号：确认门槛减半

                if pending == cur:
                    pending_cnt += 1
                    if pending_cnt >= k_w:
                        confirmed      = cur
                        seg_high       = float(high[t])
                        seg_high_date  = t
                        seg_low        = float(low[t])
                        seg_bars       = 1
                        seg_start_price = float(close[t])
                        breakup_credit = 0
                        _reset_pending()
                else:
                    pending     = cur
                    pending_cnt = 1
                # 子阶段过渡期：confirmed_only 等待积累完成再切换，否则直接显示
                result[t] = confirmed if confirmed_only else cur

        else:
            # ── 跨家族（不同方向）─────────────────────────────────
            if pending is not None and fam(cur) == fam(pending):
                # 同向继续积累（即使子阶段不同，如 gradual→steep 都是 dn 家族）
                pending = cur   # 更新为最新子阶段
                if not reverted_this_bar:
                    # 企稳中的有效 bar 才计数；8%+ 大阳/大阴信号额外加权
                    bonus = BRK_CREDIT if (breakup[t] and pending_dir == 'up') else 0
                    # breakdn 加速确认 dn：仅在 pending 已存续 ≥5 天后才加权，
                    # 防止刚出现的急跌次日即加速确认（导致在真正底部立即锁定 dn）
                    # 上涨段退出（cnf_fam=='up'）不施加此加速，给上涨充分时间确认退出
                    bonus += BRK_CREDIT if (breakdn[t] and pending_dir == 'dn'
                                            and pending_age >= 5
                                            and cnf_fam != 'up') else 0
                    pending_cnt += 1 + bonus

                k = confirm_k(confirmed, cur)

                # ── EXIT 方向的自适应加速 ─────────────────────────
                # 仅对 EXIT 方向（up→非up, dn→非dn）生效，不对 ENTRY 方向（co→up,
                # co→dn）生效。ENTRY 方向加速会导致"确认即见顶/底"——确认时价格
                # 已大幅偏离入场点。EXIT 方向加速则修复"标签和价格矛盾"。
                _is_exit = (cnf_fam == 'up' and cur_fam != 'up') or \
                           (cnf_fam == 'dn' and cur_fam != 'dn')

                # 波动率自适应 K（仅 EXIT）
                if _is_exit:
                    _ar = atr_ratio[t]
                    if _ar > 0.015:
                        _vol_scale = max(VOL_K_FLOOR, 1.0 - (_ar - 0.015) * 10.0)
                        k = max(4, int(k * _vol_scale))

                # 价格加速器（仅 EXIT + 价格方向和标签矛盾）
                # confirmed=up 但价格跌了 → 加速；confirmed=dn 但价格涨了 → 加速
                if _is_exit and seg_start_price > 0:
                    _drift = (float(close[t]) - seg_start_price) / seg_start_price
                    _contradicts = (cnf_fam == 'up' and _drift < -PRICE_ACCEL_THRESH) or \
                                   (cnf_fam == 'dn' and _drift > PRICE_ACCEL_THRESH)
                    if _contradicts:
                        k = max(4, k // 2)

                # 强信号快速通道（k 减半）
                # 上涨段退出不施加此加速：给上涨足够保护，单根强信号不足以快速确认趋势终结
                sig_dir = 'up' if cur_fam == 'up' else 'down'
                if is_strong_signal(t, sig_dir) and cnf_fam != 'up':
                    k = max(2, k // 2)

                # 上行确认额外检查：当前价需高于 20 日前
                # 防止高点已过、raw 斜率因惯性仍为正时的假确认
                momentum_ok = True
                if pending_cnt >= k and pending_dir == 'up' and t >= 20:
                    momentum_ok = float(close[t]) > float(close[t - 20])

                if pending_cnt >= k and momentum_ok:
                    confirmed      = cur
                    seg_high       = float(high[t])
                    seg_high_date  = t
                    seg_low        = float(low[t])
                    seg_bars       = 1
                    seg_start_price = float(close[t])
                    breakup_credit = 0
                    _reset_pending()
                    result[t] = confirmed
                else:
                    result[t] = confirmed if confirmed_only else prov_label(confirmed, cur)
            else:
                # 新方向偏离（与当前 pending 方向不同，或无 pending）
                pending     = cur
                new_dir     = 'up' if _rank.get(cur_fam, 1) > _rank.get(cnf_fam, 1) else 'dn'
                pending_dir = new_dir
                # 若有反弹突破信用（dn→co 方向）直接 jump-start pending_cnt
                if new_dir == 'up' and breakup_credit > 0:
                    pending_cnt    = breakup_credit
                    breakup_credit = 0
                else:
                    pending_cnt = 0   # 从 0 开始，下一轮开始积累
                pending_best = float(close[t])
                result[t]    = confirmed if confirmed_only else prov_label(confirmed, cur)

    return result
