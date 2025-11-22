import os
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib import font_manager

# 为 Matplotlib 配置支持中文的字体，避免中文标注无法显示
_preferred_cn_fonts = [
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Microsoft YaHei',
    'SimHei',
    'PingFang HK',
    'PingFang SC',
    'Arial Unicode MS',
]
_cn_font_prop = None
_selected_cn_font = None
_selected_cn_font_path = None
_LABEL_FONT_SIZE = 4.5
_OSC_LABEL_FONT_SIZE = 4.0
_LABEL_BOX_PAD = 0.04
_MIN_LABEL_WIDTH = 16.0
_MIN_LABEL_HEIGHT = 7.0


def _register_cn_font():
    """在系统字体中寻找可用的中文字体，并注册给 Matplotlib 使用。"""
    global _cn_font_prop, _selected_cn_font, _selected_cn_font_path
    if _cn_font_prop is not None:
        return

    font_records = []
    seen_paths = set()
    search_exts = ('ttf', 'otf')
    for ext in search_exts:
        try:
            for font_path in font_manager.findSystemFonts(fontext=ext):
                if not font_path or font_path in seen_paths:
                    continue
                seen_paths.add(font_path)
                font_records.append(font_path)
        except Exception:
            continue
    # 手动补充常见中文字体路径（例如 wqy-microhei 属于 ttc 文件）
    manual_candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    ]
    for candidate in manual_candidates:
        if os.path.exists(candidate) and candidate not in seen_paths:
            seen_paths.add(candidate)
            font_records.append(candidate)

    parsed_fonts = []
    for font_path in font_records:
        try:
            prop = font_manager.FontProperties(fname=font_path)
            font_name = prop.get_name()
        except Exception:
            continue
        parsed_fonts.append((font_name, font_path, prop))

    def _match_preferred():
        for preferred in _preferred_cn_fonts:
            preferred_lower = preferred.lower()
            for font_name, font_path, prop in parsed_fonts:
                if font_name.lower() == preferred_lower:
                    return font_name, font_path, prop
        return None

    def _match_by_keyword():
        keywords = ['wqy', 'micro', 'noto', 'hei', 'fang', 'yahei', 'unicode', 'song', 'kai']
        for font_name, font_path, prop in parsed_fonts:
            name_lower = font_name.lower()
            base_lower = os.path.basename(font_path).lower()
            if any(keyword in name_lower for keyword in keywords) or any(
                keyword in base_lower for keyword in keywords
            ):
                return font_name, font_path, prop
        return None

    matched = _match_preferred() or _match_by_keyword()
    if matched is None:
        return

    matched_name, matched_path, matched_prop = matched
    try:
        font_manager.fontManager.addfont(matched_path)
    except Exception:
        # addfont 在多次调用时可能抛异常，忽略即可
        pass
    _selected_cn_font = matched_prop.get_name()
    _selected_cn_font_path = matched_path
    _cn_font_prop = font_manager.FontProperties(fname=matched_path)


_register_cn_font()
_existing_fonts = plt.rcParams.get('font.sans-serif', [])
if isinstance(_existing_fonts, str):
    _existing_fonts = [_existing_fonts]
if _selected_cn_font:
    filtered_existing = [font for font in _existing_fonts if font and font != _selected_cn_font]
    plt.rcParams['font.sans-serif'] = [_selected_cn_font] + filtered_existing
    plt.rcParams['font.family'] = [_selected_cn_font, 'sans-serif']
else:
    merged = _preferred_cn_fonts + list(_existing_fonts)
    # 去重但保留顺序
    deduped = []
    for name in merged:
        if name and name not in deduped:
            deduped.append(name)
    plt.rcParams['font.sans-serif'] = deduped if deduped else ['sans-serif']
    if not plt.rcParams.get('font.family'):
        plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_kline_with_signals(df, buy_signals, sell_signals, code, out_dir='reports', oscillation_periods=None):
    import pandas as pd
    import numpy as np
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_plot = df.copy()
    # 强制转换索引为DatetimeIndex，mplfinance要求
    import pandas as pd
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        try:
            df_plot.index = pd.to_datetime(df_plot['date'])
        except Exception:
            return None
    df_plot.index.name = 'Date'

    date_series = df_plot['date'] if 'date' in df_plot.columns else None
    date_map = None
    if date_series is not None:
        date_map = {str(pd.to_datetime(d).date()).strip(): idx for idx, d in enumerate(date_series)}

    def resolve_position(ts, fallback_idx):
        """将时间戳或索引转换为 df_plot 的位置索引"""
        import numpy as np

        def _locate(value):
            try:
                loc = df_plot.index.get_loc(value)
                if isinstance(loc, slice):
                    return loc.start
                if isinstance(loc, (list, np.ndarray)):
                    return int(loc[0]) if len(loc) > 0 else None
                if isinstance(loc, (int, np.integer)):
                    return int(loc)
            except Exception:
                return None
            return None

        if fallback_idx is not None:
            pos = _locate(fallback_idx)
            if pos is not None:
                return pos
            if isinstance(fallback_idx, (int, np.integer)) and 0 <= fallback_idx < len(df_plot):
                return int(fallback_idx)

        if ts is not None:
            pos = _locate(ts)
            if pos is not None:
                return pos
            if date_map is not None:
                try:
                    dt_key = str(pd.to_datetime(ts).date()).strip()
                    mapped = date_map.get(dt_key)
                    if mapped is not None:
                        return int(mapped)
                except Exception:
                    return None
        return None

    apds = []
    parsed_oscillation_periods = []
    if oscillation_periods:
        for period in oscillation_periods:
            start = end = score = None
            raw_start_idx = raw_end_idx = None
            if isinstance(period, dict):
                start = period.get('start')
                end = period.get('end')
                score = period.get('score')
                raw_start_idx = period.get('start_idx')
                raw_end_idx = period.get('end_idx')
            elif isinstance(period, (list, tuple)):
                if len(period) >= 5:
                    raw_start_idx, raw_end_idx, start, end = period[:4]
                    score = period[4]
                elif len(period) >= 2:
                    start = period[0]
                    end = period[1]
                    score = period[2] if len(period) > 2 else None
            if start is None or end is None:
                continue
            try:
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end)
            except Exception:
                continue
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            if end_ts <= start_ts:
                continue
            start_pos = resolve_position(start_ts, raw_start_idx)
            end_pos = resolve_position(end_ts, raw_end_idx)
            # 保留原始period中的trend信息
            parsed_period = {
                'start_ts': start_ts,
                'end_ts': end_ts,
                'score': score,
                'start_pos': start_pos,
                'end_pos': end_pos
            }
            # 复制trend/category字段（如果存在）
            if isinstance(period, dict):
                if 'trend' in period:
                    parsed_period['trend'] = period['trend']
                if 'category' in period:
                    parsed_period['category'] = period['category']
            parsed_oscillation_periods.append(parsed_period)

    def resolve_indices(indices, df_index):
        import pandas as pd
        if isinstance(indices, dict):
            return []
        if hasattr(indices, 'values'):
            indices = indices.values
        result = []
        # 始终用date列和信号做字符串比对映射，无论index类型
        if date_map is not None:
            dt_indices = []
            for i in indices:
                try:
                    dt_str = str(pd.to_datetime(i).date()).strip()
                    if dt_str in date_map:
                        idx = date_map[dt_str]
                        dt_indices.append(df_plot.index[idx])
                except Exception:
                    continue
            return dt_indices
        return []
    
    buy_idx = resolve_indices(buy_signals, df_plot.index)
    sell_idx = resolve_indices(sell_signals, df_plot.index)

    def _to_plot_x(value):
        """将索引或时间戳转换为 mpf.plot 所使用的整数坐标"""
        import numpy as np
        if value is None:
            return None
        try:
            loc = df_plot.index.get_loc(value)
            if isinstance(loc, slice):
                loc = loc.start
            elif isinstance(loc, (list, np.ndarray)):
                loc = loc[0] if len(loc) > 0 else None
            if isinstance(loc, (int, np.integer)):
                return float(loc)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    def _estimate_label_size(text, font_size=_LABEL_FONT_SIZE):
        lines = [line for line in str(text).split('\n') if line]
        if not lines:
            lines = ['']
        max_chars = max(len(line) for line in lines)
        approx_char_w = font_size * 0.58
        approx_line_h = font_size * 1.25
        width = max(_MIN_LABEL_WIDTH, max_chars * approx_char_w + 14.0)
        height = max(_MIN_LABEL_HEIGHT, len(lines) * approx_line_h + 8.0)
        return width, height

    def _box_bounds(box):
        x_min = box['x_offset'] - box['width'] / 2.0
        x_max = box['x_offset'] + box['width'] / 2.0
        if box['direction'] > 0:
            y_min = box['y_offset']
            y_max = box['y_offset'] + box['height']
        else:
            y_max = box['y_offset']
            y_min = box['y_offset'] - box['height']
        return x_min, x_max, y_min, y_max

    def _boxes_overlap(box_a, box_b):
        ax1, ax2, ay1, ay2 = _box_bounds(box_a)
        bx1, bx2, by1, by2 = _box_bounds(box_b)
        return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)

    def _create_box(x_pos, x_offset, y_offset, width, height, direction):
        return {
            'x_pos': x_pos,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'width': width,
            'height': height,
            'direction': direction,
        }

    def _allocate_label_position(x_pos, text, base_offset, direction,
                                 active_boxes, row_gap=12.0, cleanup_threshold=30.0):
        width, height = _estimate_label_size(text)
        active_boxes[:] = [box for box in active_boxes if x_pos - box['x_pos'] <= cleanup_threshold]
        max_rows = 6
        for row in range(max_rows):
            step = row * (height + row_gap)
            y_offset = base_offset + (step if direction > 0 else -step)
            candidate = _create_box(x_pos, 0.0, y_offset, width, height, direction)
            collision = False
            for box in active_boxes:
                if _boxes_overlap(candidate, box):
                    collision = True
                    break
            if not collision:
                active_boxes.append(candidate)
                return 0.0, y_offset
        fallback = _create_box(x_pos, 0.0, base_offset, width, height, direction)
        active_boxes.append(fallback)
        return 0.0, base_offset

    def _assign_interval_levels(periods):
        """根据区间长度与重叠情况分配垂直层级，避免互相遮挡。"""
        sorted_periods = sorted(
            [p for p in periods if p.get('_plot_start') is not None and p.get('_plot_end') is not None],
            key=lambda item: item.get('_duration', 0),
            reverse=True
        )
        level_slots = []
        for period in sorted_periods:
            start_pos = period.get('_plot_start')
            end_pos = period.get('_plot_end')
            if start_pos is None or end_pos is None:
                continue
            level_idx = 0
            while True:
                if level_idx >= len(level_slots):
                    level_slots.append([])
                conflict = False
                for other in level_slots[level_idx]:
                    if not (end_pos < other['_plot_start'] or start_pos > other['_plot_end']):
                        conflict = True
                        break
                if conflict:
                    level_idx += 1
                    continue
                level_slots[level_idx].append(period)
                period['_level_idx'] = level_idx
                break
        return len(level_slots)


    buy_idx = sorted(
        buy_idx,
        key=lambda val: (_to_plot_x(val) if _to_plot_x(val) is not None else float('inf'))
    )
    sell_idx = sorted(
        sell_idx,
        key=lambda val: (_to_plot_x(val) if _to_plot_x(val) is not None else float('inf'))
    )
    def _to_plot_x(value):
        """将索引或时间戳转换为绘图坐标（mplfinance 使用的整数序号）"""
        import numpy as np
        if value is None:
            return None
        try:
            loc = df_plot.index.get_loc(value)
            if isinstance(loc, slice):
                loc = loc.start
            elif isinstance(loc, (list, np.ndarray)):
                loc = loc[0] if len(loc) > 0 else None
            if isinstance(loc, (int, np.integer)):
                return float(loc)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    buy_entries = []
    for dt in buy_idx:
        x_pos = _to_plot_x(dt)
        if x_pos is None:
            continue
        price = df_plot.loc[dt, 'low']
        if 'date' in df_plot.columns:
            date_str = str(df_plot.loc[dt, 'date'])
        else:
            date_str = str(df_plot.index[dt]) if hasattr(df_plot.index, '__getitem__') else str(dt)
        buy_entries.append({
            'dt': dt,
            'x_pos': x_pos,
            'price': price,
            'text': f'{date_str}\n{price:.2f}'
        })
    sell_entries = []
    for dt in sell_idx:
        x_pos = _to_plot_x(dt)
        if x_pos is None:
            continue
        price = df_plot.loc[dt, 'high']
        if 'date' in df_plot.columns:
            date_str = str(df_plot.loc[dt, 'date'])
        else:
            date_str = str(df_plot.index[dt]) if hasattr(df_plot.index, '__getitem__') else str(dt)
        sell_entries.append({
            'dt': dt,
            'x_pos': x_pos,
            'price': price,
            'text': f'{date_str}\n{price:.2f}'
        })

    osc_annotations = []
    osc_text_offset = None
    if parsed_oscillation_periods:
        low_vals = df_plot['low'].to_numpy()
        high_vals = df_plot['high'].to_numpy()
        y_min = np.nanmin(low_vals) if len(low_vals) else 0.0
        y_max = np.nanmax(high_vals) if len(high_vals) else 1.0
        if not np.isfinite(y_min):
            y_min = 0.0
        if not np.isfinite(y_max):
            y_max = max(1.0, y_min + 1.0)
        price_span = max(y_max - y_min, 1e-3)
        base_level = y_min + price_span * 0.02
        default_line_step = price_span * 0.015
        line_step = default_line_step
        osc_text_offset = price_span * 0.01

        valid_periods = []
        total_points = len(df_plot)
        close_values = None
        if 'close' in df_plot.columns:
            try:
                close_values = df_plot['close'].to_numpy()
            except Exception:
                close_values = None
        for period in parsed_oscillation_periods:
            start_pos = period.get('start_pos')
            end_pos = period.get('end_pos')
            if start_pos is None or end_pos is None:
                continue
            start_pos = max(0, min(total_points - 1, int(start_pos)))
            end_pos = max(0, min(total_points - 1, int(end_pos)))
            if end_pos <= start_pos:
                continue
            period['_plot_start'] = start_pos
            period['_plot_end'] = end_pos
            period['_duration'] = end_pos - start_pos
            trend = period.get('trend') or period.get('category')
            if trend not in ('decline', 'range', 'down', 'neutral'):
                trend = None
            if trend is None and close_values is not None:
                try:
                    start_close = float(close_values[start_pos])
                    end_close = float(close_values[end_pos])
                except Exception:
                    start_close = end_close = None
                if start_close is not None and end_close is not None and np.isfinite(start_close) and abs(start_close) > 1e-6:
                    pct_change = (end_close - start_close) / start_close
                    if pct_change <= -0.02:
                        trend = 'decline'
                    else:
                        trend = 'range'
            if trend in ('down',):
                trend = 'decline'
            elif trend == 'neutral':
                trend = 'range'
            period['_trend'] = trend if trend else 'range'
            valid_periods.append(period)

        if valid_periods:
            _assign_interval_levels(valid_periods)
            max_level_idx = max((p.get('_level_idx', 0) for p in valid_periods), default=0)
            if max_level_idx > 0:
                available_span = price_span * 0.25
                line_step = min(default_line_step, available_span / (max_level_idx + 1))
            if line_step > 0:
                osc_text_offset = min(osc_text_offset, line_step * 0.45)
        range_colors = ['#ff9800', '#ff7043', '#ffb300', '#ff5722', '#ffa726']
        decline_colors = ['#80d8ff', '#4fc3f7', '#29b6f6']
        range_color_idx = 0
        decline_color_idx = 0
        for idx, period in enumerate(valid_periods):
            start_pos = period.get('_plot_start')
            end_pos = period.get('_plot_end')
            level_idx = period.get('_level_idx', 0)
            level = base_level + level_idx * line_step
            if period.get('_trend') == 'decline':
                color = decline_colors[decline_color_idx % len(decline_colors)]
                decline_color_idx += 1
            else:
                color = range_colors[range_color_idx % len(range_colors)]
                range_color_idx += 1
            segment = np.full(len(df_plot), np.nan)
            segment[start_pos:end_pos + 1] = level
            apds.append(
                mpf.make_addplot(
                    segment,
                    color=color,
                    panel=0,
                    width=4,
                    secondary_y=False,
                    alpha=0.85
                )
            )
            osc_annotations.append((
                start_pos,
                end_pos,
                level,
                period.get('score'),
                period.get('start_ts'),
                period.get('end_ts'),
                color,
                period.get('_trend', 'range')
            ))
    # 绘图
    # 自定义style，细化上下影线（wick），通过 rc dict 设置宽度，兼容 mplfinance 0.12.10b0
    # 使用更细的上下影线和蜡烛线宽度，SVG放大依然清晰
    # 只用lines.linewidth控制上下影线，mplfinance不支持candle_linewidth
    font_family = _selected_cn_font if _selected_cn_font else 'sans-serif'
    custom_style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        rc={
            'lines.linewidth': 0.05,    # 极细上下影线和蜡烛线，SVG放大也不会变粗
            'font.family': font_family
        }
    )
    total_signals = len(buy_idx) + len(sell_idx)
    fig_width = 30.0
    if total_signals > 40:
        fig_width += min(20.0, (total_signals - 40) * 0.4)
    fig, axlist = mpf.plot(
        df_plot,
        type='candle',
        style=custom_style,
        addplot=apds,
        volume=True,
        returnfig=True,
        figsize=(fig_width, 16),
        title=f'{code} K线及买卖点'
    )
    ax = axlist[0]
    # 让成交量等辅助子图背景透明，仅保留柱状图
    for extra_ax in axlist[1:]:
        try:
            extra_ax.set_facecolor((1, 1, 1, 0))
            if extra_ax.patch is not None:
                extra_ax.patch.set_alpha(0)
        except Exception:
            continue
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    price_range = max(y_max - y_min, 1e-6)
    lower_safe = y_min + price_range * 0.12
    upper_safe = y_max - price_range * 0.12
    # 标注买入点价格和时间，并绘制带尾巴的小箭头
    buy_label_boxes = []
    sell_label_boxes = []

    for entry in buy_entries:
        x_pos = entry['x_pos']
        price = entry['price']
        text_label = entry['text']
        target_y = price - max(60.0, price_range * 0.05)
        target_y = min(target_y, price - price_range * 0.02)
        if target_y < lower_safe:
            target_y = lower_safe
        base_offset = target_y - price
        if base_offset > -10.0:
            base_offset = -10.0
        direction = -1
        valign = 'top'
        x_offset, y_offset = _allocate_label_position(
            x_pos, text_label, base_offset=base_offset, direction=direction,
            active_boxes=buy_label_boxes
        )
        arrow_style = dict(arrowstyle='-', color='lime', lw=0.8, shrinkA=0, shrinkB=0)
        price_pix = ax.transData.transform((x_pos, price))[1]
        bottom_pix = ax.transData.transform((x_pos, y_min))[1]
        available_pix = max(0.0, price_pix - bottom_pix - 8.0)
        available_pt = max(4.0, available_pix * 72.0 / fig.dpi)
        if y_offset < -available_pt:
            y_offset = -available_pt
        if y_offset > -4.0:
            y_offset = -4.0
        ax.scatter(
            [x_pos],
            [price],
            marker='^',
            color='lime',
            s=36,
            zorder=15,
            clip_on=False
        )
        ax.annotate(
                    text_label,
                    xy=(x_pos, price),
                    xytext=(x_offset, y_offset),
                    textcoords='offset points',
                    ha='center', va=valign,
                    fontsize=_LABEL_FONT_SIZE, color='lime',
                    bbox=dict(boxstyle='round,pad={}'.format(_LABEL_BOX_PAD), fc='white', ec='lime', alpha=0.85),
                    arrowprops=arrow_style,
                    zorder=20,
                    annotation_clip=False,
                    clip_on=False)
    # 标注卖出点价格和时间，并绘制带尾巴的小箭头
    for entry in sell_entries:
        x_pos = entry['x_pos']
        price = entry['price']
        text_label = entry['text']
        target_y = price + max(60.0, price_range * 0.05)
        target_y = max(target_y, price + price_range * 0.02)
        if target_y > upper_safe:
            target_y = upper_safe
        base_offset = target_y - price
        if base_offset < 10.0:
            base_offset = 10.0
        direction = 1
        valign = 'bottom'
        x_offset, y_offset = _allocate_label_position(
            x_pos, text_label, base_offset=base_offset, direction=direction,
            active_boxes=sell_label_boxes
        )
        arrow_style = dict(arrowstyle='-', color='red', lw=0.8, shrinkA=0, shrinkB=0)
        price_pix = ax.transData.transform((x_pos, price))[1]
        top_pix = ax.transData.transform((x_pos, y_max))[1]
        available_pix = max(0.0, top_pix - price_pix - 8.0)
        available_pt = max(4.0, available_pix * 72.0 / fig.dpi)
        if y_offset > available_pt:
            y_offset = available_pt
        if y_offset < 4.0:
            y_offset = 4.0
        ax.scatter(
            [x_pos],
            [price],
            marker='v',
            color='red',
            s=36,
            zorder=15,
            clip_on=False
        )
        ax.annotate(
                    text_label,
                    xy=(x_pos, price),
                    xytext=(x_offset, y_offset),
                    textcoords='offset points',
                    ha='center', va=valign,
                    fontsize=_LABEL_FONT_SIZE, color='red',
                    bbox=dict(boxstyle='round,pad={}'.format(_LABEL_BOX_PAD), fc='white', ec='red', alpha=0.85),
                    arrowprops=arrow_style,
                    zorder=20,
                    annotation_clip=False,
                    clip_on=False)

    if osc_annotations and osc_text_offset is not None:
        def _format_ts(ts):
            try:
                return pd.to_datetime(ts).strftime('%Y-%m-%d')
            except Exception:
                return str(ts) if ts is not None else ''

        for start_pos, end_pos, level, score, start_ts, end_ts, color, trend in osc_annotations:
            midpoint_pos = start_pos + (end_pos - start_pos) // 2
            midpoint_pos = max(0, min(len(df_plot) - 1, midpoint_pos))
            midpoint_x = float(midpoint_pos)
            label = "震荡下跌" if trend == 'decline' else "震荡区间"
            if score is not None:
                try:
                    label = f"{label}({float(score):.1f})"
                except Exception:
                    pass
            ax.text(
                midpoint_x,
                level + osc_text_offset,
                label,
                color=color,
                ha='center',
                va='bottom',
                fontsize=_OSC_LABEL_FONT_SIZE,
                bbox=dict(boxstyle='round,pad={}'.format(_LABEL_BOX_PAD), fc='white', ec=color, alpha=0.75),
                fontproperties=_cn_font_prop
            )
            start_label = _format_ts(start_ts)
            end_label = _format_ts(end_ts)
            y_text = level + osc_text_offset * 0.4
            # 同时展示"确认时间/终止时间"文字和具体日期
            # 注意：这里的start_ts是震荡确认时间（逐日判断确认的时间点），不是回溯的起始时间
            if start_label:
                start_x = float(start_pos)
                start_text = f"震荡确认\n{start_label}"
                ax.text(
                    start_x,
                    y_text,
                    start_text,
                    color=color,
                    ha='center',
                    va='bottom',
                    fontsize=_OSC_LABEL_FONT_SIZE,
                    bbox=dict(boxstyle='round,pad={}'.format(_LABEL_BOX_PAD), fc='white', ec=color, alpha=0.65),
                    fontproperties=_cn_font_prop
                )
            if end_label:
                end_x = float(end_pos)
                end_text = f"震荡结束\n{end_label}"
                ax.text(
                    end_x,
                    y_text,
                    end_text,
                    color=color,
                    ha='center',
                    va='bottom',
                    fontsize=_OSC_LABEL_FONT_SIZE,
                    bbox=dict(boxstyle='round,pad={}'.format(_LABEL_BOX_PAD), fc='white', ec=color, alpha=0.65),
                    fontproperties=_cn_font_prop
                )

    svg_path = os.path.join(out_dir, f'kline_{code}.svg')
    fig.savefig(svg_path, format='svg')
    plt.close(fig)
    return svg_path
