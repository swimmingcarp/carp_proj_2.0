"""
股性分析可视化：价格分段图 + 多尺度slope/R²热力图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from typing import List, Dict, Optional
from .segmenter import Segment, SCALES
import pandas as pd


# regime颜色映射
REGIME_COLORS = {
    'strong_up': '#00a651',
    'moderate_up': '#7bc67e',
    'weak_up': '#c6e6c7',
    'sideways': '#d0d0d0',
    'weak_down': '#f5c6c6',
    'moderate_down': '#e57373',
    'strong_down': '#c62828',
    'breakout_up': '#2196f3',
    'reversal_up': '#64b5f6',
    'reversal_down': '#ff9800',
    'unknown': '#ffffff',
}


def plot_stock_personality(close: np.ndarray,
                           dates: np.ndarray,
                           features: Dict[int, Dict[str, np.ndarray]],
                           segments_by_window: Dict[int, List[Segment]],
                           stock_code: str = '',
                           stock_name: str = '',
                           personality: str = '',
                           stage: str = '',
                           window: int = 400,
                           save_path: str = None):
    """
    生成3行子图:
    - 上图: 价格线 + 分段着色
    - 中图: 多尺度slope热力图
    - 下图: 多尺度R²热力图

    参数:
        close: 收盘价（完整序列）
        dates: 日期序列
        features: multi_scale_features()的全量输出
        segments_by_window: analyze_stock()的输出
        window: 显示哪个窗口（默认400天）
    """
    N = len(close)
    display_len = min(window, N)
    start = N - display_len

    # 截取显示区间
    disp_close = close[start:]
    disp_dates = dates[start:]

    # 转换日期为datetime用于绘图
    try:
        date_objs = pd.to_datetime(disp_dates)
    except Exception:
        date_objs = np.arange(len(disp_close))

    segments = segments_by_window.get(window, [])
    scales = sorted(features.keys())

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={'height_ratios': [3, 2, 2]},
                             sharex=True)

    title = f"{stock_code}"
    if stock_name:
        title += f" ({stock_name})"
    if personality:
        title += f"  |  {personality}"
    if stage:
        title += f"  |  {stage}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- 上图: 价格 + 分段着色 ----
    ax1 = axes[0]
    ax1.plot(date_objs, disp_close, color='black', linewidth=0.8, zorder=5)

    for seg in segments:
        color = REGIME_COLORS.get(seg.regime, '#ffffff')
        # Convert global indices to display-relative indices
        s = seg.start_idx - start
        e = seg.end_idx - start
        s = max(0, s)
        e = min(display_len, e)
        if s < e and s < len(date_objs) and e - 1 < len(date_objs):
            ax1.axvspan(date_objs[s], date_objs[e - 1],
                        alpha=0.3, color=color, zorder=1)
            # 标注regime名称
            mid = (s + e) // 2
            if mid < len(date_objs) and seg.duration >= 15:
                ax1.text(date_objs[mid], disp_close[s:e].max() * 1.02,
                         f"{seg.regime}\n{seg.duration}d",
                         ha='center', va='bottom', fontsize=7, alpha=0.8)

    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price with Regime Segments', fontsize=10)

    # ---- 中图: slope热力图 ----
    ax2 = axes[1]
    slope_matrix = np.full((len(scales), display_len), np.nan)
    for si, scale in enumerate(scales):
        if scale in features:
            s_data = features[scale]['slope'][start:]
            slope_matrix[si, :len(s_data)] = s_data

    # 用双色标准化：红负绿正
    vmax = min(np.nanpercentile(np.abs(slope_matrix[~np.isnan(slope_matrix)]), 95), 2.0) if np.any(~np.isnan(slope_matrix)) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im2 = ax2.imshow(slope_matrix, aspect='auto', cmap='RdYlGn',
                      norm=norm, origin='lower',
                      extent=[mdates.date2num(date_objs[0]) if hasattr(date_objs[0], 'year') else 0,
                              mdates.date2num(date_objs[-1]) if hasattr(date_objs[-1], 'year') else len(disp_close),
                              0, len(scales)])
    ax2.set_yticks(np.arange(len(scales)) + 0.5)
    ax2.set_yticklabels([f'{s}d' for s in scales])
    ax2.set_title('Slope (annualized) by Scale', fontsize=10)
    if hasattr(date_objs[0], 'year'):
        ax2.xaxis_date()
    plt.colorbar(im2, ax=ax2, label='Annual Slope', pad=0.02, fraction=0.03)

    # ---- 下图: R²热力图 ----
    ax3 = axes[2]
    r2_matrix = np.full((len(scales), display_len), np.nan)
    for si, scale in enumerate(scales):
        if scale in features:
            r2_data = features[scale]['r2'][start:]
            r2_matrix[si, :len(r2_data)] = r2_data

    im3 = ax3.imshow(r2_matrix, aspect='auto', cmap='YlOrRd',
                      vmin=0, vmax=1, origin='lower',
                      extent=[mdates.date2num(date_objs[0]) if hasattr(date_objs[0], 'year') else 0,
                              mdates.date2num(date_objs[-1]) if hasattr(date_objs[-1], 'year') else len(disp_close),
                              0, len(scales)])
    ax3.set_yticks(np.arange(len(scales)) + 0.5)
    ax3.set_yticklabels([f'{s}d' for s in scales])
    ax3.set_title('R-squared by Scale', fontsize=10)
    if hasattr(date_objs[0], 'year'):
        ax3.xaxis_date()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.colorbar(im3, ax=ax3, label='R²', pad=0.02, fraction=0.03)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        plt.close(fig)
        return None
