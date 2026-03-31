"""
因子武器库 (Factor Arsenal)
============================
350+ 因子的完整计算实现，用于区分"趋势追踪入场" vs "回调买入"场景。

使用方式:
    from factor_library import FactorEngine
    engine = FactorEngine(close, high, low, volume)
    factors = engine.compute_all(bar_idx)  # 返回 dict[str, float]

所有因子仅使用 bar_idx 及之前的数据，无未来函数。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def kama_indicator(
    close: pd.Series,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average (KAMA).

    仅使用当期及历史数据递推计算，不引入未来信息。
    """
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)
    c = pd.Series(close, copy=False).astype(float)
    values = c.to_numpy(dtype=float, copy=False)
    n = len(c)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=c.index, dtype=float)

    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)

    first_idx = max(1, int(er_period))
    if first_idx >= n:
        out[-1] = values[-1]
        return pd.Series(out, index=c.index, dtype=float).ffill()

    out[first_idx] = values[first_idx]
    abs_diff = np.zeros(n, dtype=float)
    abs_diff[1:] = np.abs(np.diff(values))
    abs_csum = np.concatenate(([0.0], np.cumsum(abs_diff)))
    for i in range(first_idx + 1, n):
        signal = abs(values[i] - values[i - er_period])
        noise = abs_csum[i + 1] - abs_csum[i - er_period + 1]
        er = signal / noise if noise > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        prev = out[i - 1] if not np.isnan(out[i - 1]) else values[i - 1]
        out[i] = prev + sc * (values[i] - prev)

    return pd.Series(out, index=c.index, dtype=float).ffill()


def choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Choppiness Index (CHOP).

    CHOP 越高表示越震荡，越低表示越趋势化。
    """
    if high is None or low is None or close is None:
        return pd.Series(dtype=float)
    h = pd.Series(high, copy=False).astype(float)
    l = pd.Series(low, copy=False).astype(float)
    c = pd.Series(close, copy=False).astype(float)
    idx = c.index
    if len(c) == 0:
        return pd.Series(dtype=float, index=idx)

    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)

    tr_sum = tr.rolling(period, min_periods=period).sum()
    hh = h.rolling(period, min_periods=period).max()
    ll = l.rolling(period, min_periods=period).min()
    span = (hh - ll).replace(0, np.nan)
    base = np.log10(float(period))

    chop = 100.0 * (np.log10(tr_sum / span) / base)
    return chop.replace([np.inf, -np.inf], np.nan)


def efficiency_ratio_indicator(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Kaufman Efficiency Ratio (ER).

    ER 越高表示近段价格运动更“直线化”（趋势更清晰），
    越低表示来回震荡更明显。
    """
    if close is None:
        return pd.Series(dtype=float)
    c = pd.Series(close, copy=False).astype(float)
    if len(c) == 0:
        return pd.Series(dtype=float, index=c.index)

    direction = (c - c.shift(period)).abs()
    volatility = c.diff().abs().rolling(period, min_periods=period).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.replace([np.inf, -np.inf], np.nan)


class FactorEngine:
    """因子计算引擎：预计算所有中间指标，按需提取因子值"""

    def __init__(self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 volume: np.ndarray, dates: Optional[np.ndarray] = None):
        self.close = close.astype(float)
        self.high = high.astype(float)
        self.low = low.astype(float)
        self.volume = volume.astype(float)
        self.dates = dates
        self.n = len(close)

        # 预计算常用指标
        self._precompute()

    # ================================================================
    # 预计算
    # ================================================================
    def _precompute(self):
        c, h, l, v = self.close, self.high, self.low, self.volume
        n = self.n

        # MA系列
        self.ma5 = self._sma(c, 5)
        self.ma10 = self._sma(c, 10)
        self.ma15 = self._sma(c, 15)
        self.ma20 = self._sma(c, 20)
        self.ma60 = self._sma(c, 60)
        self.ma120 = self._sma(c, 120)
        self.ma200 = self._sma(c, 200) if n >= 200 else np.full(n, np.nan)
        self.ma250 = self._sma(c, 250) if n >= 250 else np.full(n, np.nan)

        # EMA系列
        self.ema5 = self._ema(c, 5)
        self.ema10 = self._ema(c, 10)
        self.ema12 = self._ema(c, 12)
        self.ema20 = self._ema(c, 20)
        self.ema26 = self._ema(c, 26)
        self.ema50 = self._ema(c, 50)

        # MACD
        self.macd_line = self.ema12 - self.ema26
        self.macd_signal = self._ema(self.macd_line, 9)
        self.macd_hist = self.macd_line - self.macd_signal

        # ATR
        self.tr = self._true_range(h, l, c)
        self.atr14 = self._ema_atr(self.tr, 14)
        self.atr20 = self._ema_atr(self.tr, 20)

        # RSI
        self.rsi14 = self._rsi(c, 14)
        self.rsi29 = self._rsi(c, 29)

        # Bollinger Bands (20, 2)
        self.bb_mid = self.ma20
        self.bb_std = self._rolling_std(c, 20)
        self.bb_upper = self.bb_mid + 2 * self.bb_std
        self.bb_lower = self.bb_mid - 2 * self.bb_std

        # Volume MA
        self.vol_ma5 = self._sma(v, 5)
        self.vol_ma10 = self._sma(v, 10)
        self.vol_ma20 = self._sma(v, 20)

        # Daily returns (log)
        self.log_ret = np.zeros(n)
        self.log_ret[1:] = np.log(c[1:] / c[:-1])

        # Daily returns (simple)
        self.pct_ret = np.zeros(n)
        self.pct_ret[1:] = (c[1:] - c[:-1]) / c[:-1]

        # OBV
        self.obv = self._compute_obv(c, v)

        # Stochastic K (14)
        self.stoch_k = self._stochastic_k(h, l, c, 14)

    # ================================================================
    # 核心计算工具
    # ================================================================
    @staticmethod
    def _sma(arr, period):
        out = np.full(len(arr), np.nan)
        if len(arr) >= period:
            cs = np.cumsum(arr)
            out[period-1:] = (cs[period-1:] - np.concatenate([[0], cs[:-period]])) / period
        return out

    @staticmethod
    def _ema(arr, period):
        out = np.full(len(arr), np.nan)
        if len(arr) < period:
            return out
        alpha = 2.0 / (period + 1)
        out[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out

    @staticmethod
    def _ema_atr(tr, period):
        out = np.full(len(tr), np.nan)
        if len(tr) < period:
            return out
        out[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            out[i] = (out[i-1] * (period - 1) + tr[i]) / period
        return out

    @staticmethod
    def _true_range(h, l, c):
        tr = np.zeros(len(c))
        tr[0] = h[0] - l[0]
        for i in range(1, len(c)):
            tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        return tr

    @staticmethod
    def _rsi(close, period):
        n = len(close)
        rsi = np.full(n, np.nan)
        gains = np.zeros(n)
        losses = np.zeros(n)
        for i in range(1, n):
            d = close[i] - close[i-1]
            if d > 0: gains[i] = d
            else: losses[i] = -d
        if n <= period:
            return rsi
        ag = np.mean(gains[1:period+1])
        al = np.mean(losses[1:period+1])
        if al > 0:
            rsi[period] = 100 - 100 / (1 + ag/al)
        for i in range(period+1, n):
            ag = (ag*(period-1) + gains[i]) / period
            al = (al*(period-1) + losses[i]) / period
            rsi[i] = 100 - 100/(1+ag/al) if al > 0 else 100
        return rsi

    @staticmethod
    def _rolling_std(arr, period):
        out = np.full(len(arr), np.nan)
        for i in range(period-1, len(arr)):
            out[i] = np.std(arr[i-period+1:i+1], ddof=1)
        return out

    @staticmethod
    def _compute_obv(close, volume):
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    @staticmethod
    def _stochastic_k(high, low, close, period):
        n = len(close)
        k = np.full(n, np.nan)
        for i in range(period-1, n):
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
            k[i] = (close[i] - ll) / (hh - ll) * 100 if hh > ll else 50
        return k

    def _safe(self, arr, idx, default=np.nan):
        if idx < 0 or idx >= len(arr):
            return default
        v = arr[idx]
        return default if np.isnan(v) or not np.isfinite(v) else v

    # ================================================================
    # 主入口：计算bar_idx时刻的全部因子
    # ================================================================
    def compute_all(self, idx: int) -> Dict[str, float]:
        """计算bar_idx时刻的全部因子。仅用idx及之前的数据。"""
        if idx < 120:
            return {}

        f = {}
        c, h, l, v = self.close, self.high, self.low, self.volume
        price = c[idx]

        # === 一、波动率类 (30个) ===
        f.update(self._volatility_factors(idx))

        # === 二、趋势/MA类 (30个) ===
        f.update(self._trend_ma_factors(idx))

        # === 三、动量/收益类 (25个) ===
        f.update(self._momentum_factors(idx))

        # === 四、成交量类 (20个) ===
        f.update(self._volume_factors(idx))

        # === 五、价格结构类 (25个) ===
        f.update(self._price_structure_factors(idx))

        # === 六、统计类 (30个) ===
        f.update(self._statistical_factors(idx))

        # === 七、MACD/振荡器类 (15个) ===
        f.update(self._oscillator_factors(idx))

        # === 八、支撑阻力类 (10个) ===
        f.update(self._support_resistance_factors(idx))

        # === 九、频域/周期性类 (25个) ===
        f.update(self._frequency_factors(idx))

        # === 十、高级统计/检验类 (25个) ===
        f.update(self._advanced_stat_factors(idx))

        # === 十一、价格行为微观类 (25个) ===
        f.update(self._micro_price_action_factors(idx))

        # === 十二、多周期/跨时间类 (20个) ===
        f.update(self._multi_timeframe_factors(idx))

        # === 十三、成交量微结构类 (15个) ===
        f.update(self._volume_micro_factors(idx))

        # === 十四、趋势质量/成熟度类 (20个) ===
        f.update(self._trend_quality_factors(idx))

        # === 十五、扩展振荡器类 (15个) ===
        f.update(self._extended_oscillator_factors(idx))

        # === 十六、扩展多周期类 (20个) ===
        f.update(self._extended_multi_tf_factors(idx))

        # === 十七、扩展支撑阻力/形态类 (15个) ===
        f.update(self._extended_sr_pattern_factors(idx))

        # === 十八、波动率衍生/高级类 (15个) ===
        f.update(self._extended_volatility_factors(idx))

        # === 十九、资金流/换手率/A股特有类 (15个) ===
        f.update(self._astock_specific_factors(idx))

        # === 二十、长周期因子 (50个) ===
        f.update(self._long_term_factors(idx))

        return f

    # ================================================================
    # 一、波动率类 (30个)
    # ================================================================
    def _volatility_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        price = c[idx]
        f = {}

        # ATR系列
        f['atr14_pct'] = self._safe(self.atr14, idx, 0) / price * 100
        f['atr20_pct'] = self._safe(self.atr20, idx, 0) / price * 100
        f['atr20_change_5d'] = self._pct_change(self.atr20, idx, 5)
        f['atr20_change_10d'] = self._pct_change(self.atr20, idx, 10)
        f['atr20_change_20d'] = self._pct_change(self.atr20, idx, 20)

        # ATR百分位(200日)
        if idx >= 200:
            atr_hist = self.atr20[idx-199:idx+1] / c[idx-199:idx+1] * 100
            valid = atr_hist[~np.isnan(atr_hist)]
            f['atr_percentile_200d'] = np.sum(valid < f['atr20_pct']) / len(valid) if len(valid) > 10 else 0.5
        else:
            f['atr_percentile_200d'] = 0.5

        # 年化波动率
        f['ann_vol_5d'] = np.std(self.log_ret[max(0,idx-4):idx+1]) * np.sqrt(252) * 100
        f['ann_vol_10d'] = np.std(self.log_ret[max(0,idx-9):idx+1]) * np.sqrt(252) * 100
        f['ann_vol_20d'] = np.std(self.log_ret[max(0,idx-19):idx+1]) * np.sqrt(252) * 100
        f['ann_vol_60d'] = np.std(self.log_ret[max(0,idx-59):idx+1]) * np.sqrt(252) * 100
        f['vol_ratio_5_20'] = f['ann_vol_5d'] / f['ann_vol_20d'] if f['ann_vol_20d'] > 0 else 1
        f['vol_ratio_10_60'] = f['ann_vol_10d'] / f['ann_vol_60d'] if f['ann_vol_60d'] > 0 else 1

        # 日内振幅
        ranges = (h[idx-19:idx+1] - l[idx-19:idx+1]) / c[idx-19:idx+1] * 100
        f['intraday_range_pct'] = (h[idx]-l[idx]) / price * 100
        f['intraday_range_med_10d'] = np.median(ranges[-10:])
        f['intraday_range_med_20d'] = np.median(ranges)
        f['intraday_range_std_20d'] = np.std(ranges)

        # Bollinger Band Width & Squeeze
        bw = self._safe(self.bb_upper, idx, 0) - self._safe(self.bb_lower, idx, 0)
        f['bb_width_pct'] = bw / self._safe(self.bb_mid, idx, 1) * 100
        f['bb_pct'] = (price - self._safe(self.bb_lower, idx, price)) / bw if bw > 0 else 0.5
        # BB宽度百分位
        if idx >= 120:
            bw_hist = []
            for j in range(idx-119, idx+1):
                bw_j = self._safe(self.bb_upper, j, 0) - self._safe(self.bb_lower, j, 0)
                mid_j = self._safe(self.bb_mid, j, 1)
                if mid_j > 0:
                    bw_hist.append(bw_j / mid_j * 100)
            f['bb_width_percentile'] = np.sum(np.array(bw_hist) < f['bb_width_pct']) / len(bw_hist) if bw_hist else 0.5
        else:
            f['bb_width_percentile'] = 0.5

        # Parkinson波动率
        hl_log = np.log(h[idx-19:idx+1] / l[idx-19:idx+1])
        f['parkinson_vol'] = np.sqrt(np.sum(hl_log**2) / (4*20*np.log(2))) * 100

        # 方差比(Variance Ratio)
        ret1 = self.log_ret[max(0,idx-39):idx+1]
        if len(ret1) >= 20:
            v1 = np.var(ret1)
            # 2日收益
            ret2 = ret1[::2] if len(ret1) >= 4 else ret1
            v2 = np.var(ret2) if len(ret2) >= 2 else v1
            f['variance_ratio_2d'] = v2 / (2*v1) if v1 > 0 else 1
        else:
            f['variance_ratio_2d'] = 1

        # 连续高/低波动天数
        med_rng = np.median(ranges)
        low_v = 0
        for j in range(idx, max(idx-20, 0), -1):
            if (h[j]-l[j])/c[j]*100 < med_rng * 0.7:
                low_v += 1
            else:
                break
        f['consec_low_vol_days'] = low_v

        high_v = 0
        for j in range(idx, max(idx-20, 0), -1):
            if (h[j]-l[j])/c[j]*100 > med_rng * 1.3:
                high_v += 1
            else:
                break
        f['consec_high_vol_days'] = high_v

        # Garman-Klass波动率
        n_gk = min(20, idx)
        if n_gk >= 5:
            gk = 0
            for j in range(idx-n_gk+1, idx+1):
                u = np.log(h[j]/c[j-1]) if c[j-1] > 0 else 0
                d = np.log(l[j]/c[j-1]) if c[j-1] > 0 else 0
                cc = np.log(c[j]/c[j-1]) if c[j-1] > 0 else 0
                gk += 0.5*(u-d)**2 - (2*np.log(2)-1)*cc**2
            f['garman_klass_vol'] = np.sqrt(gk/n_gk) * 100
        else:
            f['garman_klass_vol'] = 0

        return f

    # ================================================================
    # 二、趋势/MA类 (30个)
    # ================================================================
    def _trend_ma_factors(self, idx: int) -> Dict[str, float]:
        c = self.close
        price = c[idx]
        f = {}

        # 价格位置 (Price Position)
        for name, period in [('pp10', 10), ('pp20', 20), ('pp60', 60), ('pp120', 120)]:
            start = max(0, idx - period + 1)
            hh = np.max(self.high[start:idx+1])
            ll = np.min(self.low[start:idx+1])
            f[name] = (price - ll) / (hh - ll) if hh > ll else 0.5

        # MA乖离率
        for name, ma in [('dist_ma5', self.ma5), ('dist_ma10', self.ma10),
                          ('dist_ma15', self.ma15), ('dist_ma20', self.ma20),
                          ('dist_ma60', self.ma60), ('dist_ma120', self.ma120)]:
            v = self._safe(ma, idx, 0)
            f[name] = (price / v - 1) * 100 if v > 0 else 0

        # MA斜率
        for name, ma, lookback in [('ma5_slope_3d', self.ma5, 3), ('ma10_slope_5d', self.ma10, 5),
                                     ('ma15_slope_5d', self.ma15, 5), ('ma20_slope_10d', self.ma20, 10),
                                     ('ma60_slope_20d', self.ma60, 20), ('ma120_slope_20d', self.ma120, 20)]:
            f[name] = self._pct_change(ma, idx, lookback)

        # EMA排列一致性
        emas = [self._safe(self.ema5, idx), self._safe(self.ema10, idx),
                self._safe(self.ema20, idx), self._safe(self.ema50, idx)]
        valid_emas = [v for v in emas if not np.isnan(v)]
        if len(valid_emas) >= 2:
            aligned = sum(1 for i in range(len(valid_emas)-1) if valid_emas[i] > valid_emas[i+1])
            f['ema_alignment'] = aligned / (len(valid_emas)-1)
        else:
            f['ema_alignment'] = 0.5

        # 价格在MA上/下
        f['above_ma5'] = 1 if price > self._safe(self.ma5, idx, price+1) else 0
        f['above_ma20'] = 1 if price > self._safe(self.ma20, idx, price+1) else 0
        f['above_ma60'] = 1 if price > self._safe(self.ma60, idx, price+1) else 0
        f['above_ma120'] = 1 if price > self._safe(self.ma120, idx, price+1) else 0

        # MA交叉频率
        for name, ma, period in [('cross_ma5_freq_10d', self.ma5, 10),
                                   ('cross_ma20_freq_20d', self.ma20, 20),
                                   ('cross_ma60_freq_60d', self.ma60, 60)]:
            cnt = 0
            for j in range(max(0, idx-period+1), idx+1):
                if not np.isnan(ma[j]) and not np.isnan(ma[j-1]):
                    if (c[j] > ma[j]) != (c[j-1] > ma[j-1]):
                        cnt += 1
            f[name] = cnt / period

        # 线性回归
        for name, period in [('lr_r2_10d', 10), ('lr_r2_20d', 20), ('lr_r2_60d', 60)]:
            x = np.arange(period)
            y = c[idx-period+1:idx+1]
            if len(y) == period:
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                f[name] = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                f[name.replace('r2', 'slope')] = slope / np.mean(y) * 100
            else:
                f[name] = 0
                f[name.replace('r2', 'slope')] = 0

        return f

    # ================================================================
    # 三、动量/收益类 (25个)
    # ================================================================
    def _momentum_factors(self, idx: int) -> Dict[str, float]:
        c = self.close
        f = {}

        # 收益率
        for name, period in [('ret_1d', 1), ('ret_2d', 2), ('ret_3d', 3), ('ret_5d', 5),
                               ('ret_10d', 10), ('ret_20d', 20), ('ret_60d', 60), ('ret_120d', 120)]:
            f[name] = (c[idx]/c[idx-period]-1)*100 if idx >= period else 0

        # RSI系列
        f['rsi14'] = self._safe(self.rsi14, idx, 50)
        f['rsi29'] = self._safe(self.rsi29, idx, 50)
        f['rsi29_med_5d'] = np.nanmedian(self.rsi29[max(0,idx-4):idx+1])
        f['rsi29_med_10d'] = np.nanmedian(self.rsi29[max(0,idx-9):idx+1])
        f['rsi29_med_20d'] = np.nanmedian(self.rsi29[max(0,idx-19):idx+1])
        f['rsi29_slope_5d'] = (self._safe(self.rsi29, idx, 50) - self._safe(self.rsi29, idx-5, 50)) / 5
        f['rsi29_range_20d'] = np.nanmax(self.rsi29[max(0,idx-19):idx+1]) - np.nanmin(self.rsi29[max(0,idx-19):idx+1])
        f['rsi_cross50_count_20d'] = sum(1 for j in range(max(1,idx-19), idx+1)
                                          if not np.isnan(self.rsi29[j]) and not np.isnan(self.rsi29[j-1])
                                          and (self.rsi29[j] > 50) != (self.rsi29[j-1] > 50))

        # Kaufman Efficiency Ratio
        for name, period in [('er_10', 10), ('er_15', 15), ('er_20', 20)]:
            if idx >= period:
                direction = abs(c[idx] - c[idx-period])
                noise = sum(abs(c[j]-c[j-1]) for j in range(idx-period+1, idx+1))
                f[name] = direction / noise if noise > 0 else 0
            else:
                f[name] = 0

        # 连续涨跌
        up = 0
        for j in range(idx, max(0, idx-30), -1):
            if c[j] > c[j-1]: up += 1
            else: break
        f['consec_up'] = up

        down = 0
        for j in range(idx, max(0, idx-30), -1):
            if c[j] < c[j-1]: down += 1
            else: break
        f['consec_down'] = down

        # 涨跌比
        f['up_ratio_10d'] = sum(1 for j in range(idx-9, idx+1) if c[j] > c[j-1]) / 10
        f['up_ratio_20d'] = sum(1 for j in range(idx-19, idx+1) if c[j] > c[j-1]) / 20

        return f

    # ================================================================
    # 四、成交量类 (20个)
    # ================================================================
    def _volume_factors(self, idx: int) -> Dict[str, float]:
        v = self.volume
        c = self.close
        f = {}

        vm20 = self._safe(self.vol_ma20, idx, 1)
        vm5 = self._safe(self.vol_ma5, idx, 1)

        f['vol_ratio'] = v[idx] / vm20 if vm20 > 0 else 1
        f['vol_5d_ratio'] = vm5 / vm20 if vm20 > 0 else 1
        f['vol_change_5d'] = (vm5 / self._safe(self.vol_ma5, idx-5, vm5) - 1) * 100 if idx >= 5 else 0

        # 量价相关
        if idx >= 20:
            vols = v[idx-19:idx+1]
            price_chg = np.abs(np.diff(c[idx-20:idx+1]))
            if len(vols) == len(price_chg) and np.std(vols) > 0 and np.std(price_chg) > 0:
                f['vol_price_corr_20d'] = np.corrcoef(vols, price_chg)[0,1]
            else:
                f['vol_price_corr_20d'] = 0
        else:
            f['vol_price_corr_20d'] = 0

        # OBV斜率
        if idx >= 20:
            obv_seg = self.obv[idx-19:idx+1]
            x = np.arange(20)
            if np.std(obv_seg) > 0:
                slope = np.polyfit(x, obv_seg, 1)[0]
                f['obv_slope_20d'] = slope / (np.mean(np.abs(obv_seg)) + 1)
            else:
                f['obv_slope_20d'] = 0
        else:
            f['obv_slope_20d'] = 0

        # 成交量方向性
        up_vol = sum(v[j] for j in range(max(0,idx-19), idx+1) if c[j] > c[j-1])
        dn_vol = sum(v[j] for j in range(max(0,idx-19), idx+1) if c[j] < c[j-1])
        f['vol_direction_20d'] = up_vol / (up_vol + dn_vol) if (up_vol + dn_vol) > 0 else 0.5

        # 放量/缩量天数
        f['vol_spike_ratio_20d'] = sum(1 for j in range(idx-19, idx+1) if vm20 > 0 and v[j] > vm20 * 1.5) / 20
        f['vol_dry_ratio_20d'] = sum(1 for j in range(idx-19, idx+1) if vm20 > 0 and v[j] < vm20 * 0.5) / 20

        # 成交量波动
        if idx >= 20:
            vol_seg = v[idx-19:idx+1]
            f['vol_cv'] = np.std(vol_seg) / np.mean(vol_seg) if np.mean(vol_seg) > 0 else 0
        else:
            f['vol_cv'] = 0

        # 成交量趋势
        if idx >= 10:
            v_first = np.mean(v[idx-9:idx-4])
            v_last = np.mean(v[idx-4:idx+1])
            f['vol_trend_10d'] = (v_last / v_first - 1) * 100 if v_first > 0 else 0
        else:
            f['vol_trend_10d'] = 0

        # Accumulation/Distribution简化
        if idx >= 20:
            ad = 0
            for j in range(idx-19, idx+1):
                clv = ((c[j]-self.low[j]) - (self.high[j]-c[j])) / (self.high[j]-self.low[j]) if self.high[j] > self.low[j] else 0
                ad += clv * v[j]
            f['ad_line_20d'] = ad / (np.mean(v[idx-19:idx+1]) * 20) if np.mean(v[idx-19:idx+1]) > 0 else 0
        else:
            f['ad_line_20d'] = 0

        # MFI (14)
        if idx >= 14:
            pos_flow = 0
            neg_flow = 0
            for j in range(idx-13, idx+1):
                tp = (self.high[j] + self.low[j] + c[j]) / 3
                tp_prev = (self.high[j-1] + self.low[j-1] + c[j-1]) / 3
                mf = tp * v[j]
                if tp > tp_prev:
                    pos_flow += mf
                else:
                    neg_flow += mf
            f['mfi_14'] = 100 - 100 / (1 + pos_flow/neg_flow) if neg_flow > 0 else 100
        else:
            f['mfi_14'] = 50

        # CMF (20)
        if idx >= 20:
            cmf_num = 0
            cmf_den = 0
            for j in range(idx-19, idx+1):
                clv = ((c[j]-self.low[j]) - (self.high[j]-c[j])) / (self.high[j]-self.low[j]) if self.high[j] > self.low[j] else 0
                cmf_num += clv * v[j]
                cmf_den += v[j]
            f['cmf_20'] = cmf_num / cmf_den if cmf_den > 0 else 0
        else:
            f['cmf_20'] = 0

        return f

    # ================================================================
    # 五、价格结构类 (25个)
    # ================================================================
    def _price_structure_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # K线形态
        f['close_position'] = (c[idx]-l[idx]) / (h[idx]-l[idx]) if h[idx] > l[idx] else 0.5
        body = abs(c[idx] - c[idx-1])
        rng = h[idx] - l[idx]
        f['body_ratio'] = body / rng if rng > 0 else 0
        f['upper_wick'] = (h[idx] - max(c[idx], c[idx-1])) / rng if rng > 0 else 0
        f['lower_wick'] = (min(c[idx], c[idx-1]) - l[idx]) / rng if rng > 0 else 0

        # Higher High / Higher Low 比率
        f['hh_ratio_10d'] = sum(1 for j in range(idx-9, idx+1) if h[j] > h[j-1]) / 10
        f['hl_ratio_10d'] = sum(1 for j in range(idx-9, idx+1) if l[j] > l[j-1]) / 10
        f['hh_ratio_20d'] = sum(1 for j in range(idx-19, idx+1) if h[j] > h[j-1]) / 20
        f['hl_ratio_20d'] = sum(1 for j in range(idx-19, idx+1) if l[j] > l[j-1]) / 20

        # 方向变化频率
        f['dir_change_freq_10d'] = sum(1 for j in range(idx-9, idx) if (c[j+1]-c[j])*(c[j]-c[j-1]) < 0) / 9
        f['dir_change_freq_20d'] = sum(1 for j in range(idx-19, idx) if (c[j+1]-c[j])*(c[j]-c[j-1]) < 0) / 19

        # 跳空
        f['gap_count_10d'] = sum(1 for j in range(idx-9, idx+1) if abs(c[j]-c[j-1])/c[j-1]*100 > 1.5) / 10
        f['avg_gap_size_10d'] = np.mean([abs(c[j]-c[j-1])/c[j-1]*100 for j in range(idx-9, idx+1)])

        # 最近新高/新低距离
        hh_60 = np.max(h[idx-59:idx+1])
        ll_60 = np.min(l[idx-59:idx+1])
        f['days_since_60d_high'] = 0
        for j in range(idx, idx-60, -1):
            if h[j] >= hh_60 * 0.99:
                f['days_since_60d_high'] = idx - j
                break
        f['days_since_60d_low'] = 0
        for j in range(idx, idx-60, -1):
            if l[j] <= ll_60 * 1.01:
                f['days_since_60d_low'] = idx - j
                break

        # 最大回撤
        peak = c[idx-59]
        max_dd = 0
        for j in range(idx-59, idx+1):
            if c[j] > peak: peak = c[j]
            dd = (c[j]/peak - 1) * 100
            if dd < max_dd: max_dd = dd
        f['max_drawdown_60d'] = max_dd

        # 最大涨幅
        trough = c[idx-59]
        max_up = 0
        for j in range(idx-59, idx+1):
            if c[j] < trough: trough = c[j]
            up = (c[j]/trough - 1) * 100
            if up > max_up: max_up = up
        f['max_rally_60d'] = max_up

        # 价格加速度
        if idx >= 10:
            ret_recent = (c[idx]/c[idx-5]-1)*100
            ret_prev = (c[idx-5]/c[idx-10]-1)*100
            f['price_acceleration'] = ret_recent - ret_prev
        else:
            f['price_acceleration'] = 0

        # 最大连续涨跌天数(30日)
        max_up = 0; streak = 0
        for j in range(max(0,idx-29), idx+1):
            if c[j] > c[j-1]: streak += 1; max_up = max(max_up, streak)
            else: streak = 0
        f['max_up_streak_30d'] = max_up

        max_dn = 0; streak = 0
        for j in range(max(0,idx-29), idx+1):
            if c[j] < c[j-1]: streak += 1; max_dn = max(max_dn, streak)
            else: streak = 0
        f['max_down_streak_30d'] = max_dn

        # 近期振幅范围(range)
        f['range_20d_pct'] = (np.max(h[idx-19:idx+1]) - np.min(l[idx-19:idx+1])) / c[idx] * 100
        f['range_60d_pct'] = (np.max(h[idx-59:idx+1]) - np.min(l[idx-59:idx+1])) / c[idx] * 100

        # Fibonacci回撤深度
        high_60 = np.max(h[idx-59:idx+1])
        low_60 = np.min(l[idx-59:idx+1])
        if high_60 > low_60:
            f['fib_retracement'] = (high_60 - c[idx]) / (high_60 - low_60)
        else:
            f['fib_retracement'] = 0.5

        return f

    # ================================================================
    # 六、统计类 (30个)
    # ================================================================
    def _statistical_factors(self, idx: int) -> Dict[str, float]:
        f = {}

        # 不同窗口的收益率序列
        for name, period in [('_20d', 20), ('_60d', 60)]:
            rets = self.log_ret[max(1,idx-period+1):idx+1]
            if len(rets) < 5:
                for k in ['skewness', 'kurtosis', 'autocorr1', 'autocorr5']:
                    f[k + name] = 0
                continue

            f['skewness' + name] = float(np.mean((rets - np.mean(rets))**3) / (np.std(rets)**3 + 1e-10))
            f['kurtosis' + name] = float(np.mean((rets - np.mean(rets))**4) / (np.std(rets)**4 + 1e-10)) - 3

            # 自相关
            if len(rets) > 2:
                f['autocorr1' + name] = np.corrcoef(rets[:-1], rets[1:])[0,1]
            else:
                f['autocorr1' + name] = 0

            if len(rets) > 6:
                f['autocorr5' + name] = np.corrcoef(rets[:-5], rets[5:])[0,1]
            else:
                f['autocorr5' + name] = 0

        # 方差比
        f['variance_ratio'] = self._compute_variance_ratio(idx, 20)

        # 尾部分布
        rets_20 = self.log_ret[max(1,idx-19):idx+1]
        if len(rets_20) >= 10:
            mu, sigma = np.mean(rets_20), np.std(rets_20)
            f['left_tail_pct'] = np.sum(rets_20 < mu - 2*sigma) / len(rets_20)
            f['right_tail_pct'] = np.sum(rets_20 > mu + 2*sigma) / len(rets_20)
            f['downside_vol'] = np.std(rets_20[rets_20 < 0]) * np.sqrt(252) * 100 if np.sum(rets_20 < 0) > 1 else 0
            f['upside_vol'] = np.std(rets_20[rets_20 > 0]) * np.sqrt(252) * 100 if np.sum(rets_20 > 0) > 1 else 0
            f['up_down_vol_ratio'] = f['upside_vol'] / f['downside_vol'] if f['downside_vol'] > 0 else 1
        else:
            f['left_tail_pct'] = 0
            f['right_tail_pct'] = 0
            f['downside_vol'] = 0
            f['upside_vol'] = 0
            f['up_down_vol_ratio'] = 1

        # 排列熵 (Permutation Entropy)
        rets_30 = self.log_ret[max(1,idx-29):idx+1]
        f['perm_entropy_3'] = self._permutation_entropy(rets_30, order=3)
        f['perm_entropy_4'] = self._permutation_entropy(rets_30, order=4)

        # 样本熵 (Sample Entropy, 简化版)
        f['sample_entropy'] = self._sample_entropy(rets_30, m=2, r_mult=0.2)

        # 中位数绝对偏差
        f['mad_20d'] = np.median(np.abs(rets_20 - np.median(rets_20))) if len(rets_20) >= 5 else 0

        # Runs Test (游程检验)
        if len(rets_20) >= 10:
            signs = (rets_20 > 0).astype(int)
            runs = 1 + sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
            n_pos = np.sum(signs)
            n_neg = len(signs) - n_pos
            if n_pos > 0 and n_neg > 0:
                exp_runs = 1 + 2*n_pos*n_neg / (n_pos + n_neg)
                var_runs = (2*n_pos*n_neg*(2*n_pos*n_neg - n_pos - n_neg)) / ((n_pos+n_neg)**2 * (n_pos+n_neg-1))
                f['runs_z'] = (runs - exp_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
            else:
                f['runs_z'] = 0
        else:
            f['runs_z'] = 0

        # ACF衰减率
        if len(rets_30) >= 15:
            acf_vals = [np.corrcoef(rets_30[:-k], rets_30[k:])[0,1] for k in range(1, 6)]
            acf_abs = [abs(v) for v in acf_vals if not np.isnan(v)]
            if len(acf_abs) >= 2 and acf_abs[0] > 0:
                f['acf_decay_rate'] = acf_abs[-1] / acf_abs[0]
            else:
                f['acf_decay_rate'] = 1
        else:
            f['acf_decay_rate'] = 1

        return f

    # ================================================================
    # 七、MACD/振荡器类 (15个)
    # ================================================================
    def _oscillator_factors(self, idx: int) -> Dict[str, float]:
        f = {}

        f['macd_line'] = self._safe(self.macd_line, idx, 0)
        f['macd_signal'] = self._safe(self.macd_signal, idx, 0)
        f['macd_hist'] = self._safe(self.macd_hist, idx, 0)

        # MACD斜率
        if idx >= 5:
            f['macd_hist_slope'] = self._safe(self.macd_hist, idx, 0) - self._safe(self.macd_hist, idx-5, 0)
        else:
            f['macd_hist_slope'] = 0

        # MACD方向持续天数
        macd_dir = 0
        for j in range(idx, max(0, idx-30), -1):
            if not np.isnan(self.macd_hist[j]):
                if self.macd_hist[j] > 0:
                    macd_dir += 1
                else:
                    break
            else:
                break
        f['macd_positive_days'] = macd_dir

        # Stochastic K
        f['stoch_k'] = self._safe(self.stoch_k, idx, 50)

        # Williams %R (14)
        if idx >= 14:
            hh = np.max(self.high[idx-13:idx+1])
            ll = np.min(self.low[idx-13:idx+1])
            f['williams_r'] = (hh - self.close[idx]) / (hh - ll) * -100 if hh > ll else -50
        else:
            f['williams_r'] = -50

        # CCI (20)
        if idx >= 20:
            tp = (self.high[idx-19:idx+1] + self.low[idx-19:idx+1] + self.close[idx-19:idx+1]) / 3
            tp_mean = np.mean(tp)
            tp_mad = np.mean(np.abs(tp - tp_mean))
            f['cci_20'] = (tp[-1] - tp_mean) / (0.015 * tp_mad) if tp_mad > 0 else 0
        else:
            f['cci_20'] = 0

        # ROC (10, 20)
        f['roc_10'] = (self.close[idx]/self.close[idx-10]-1)*100 if idx >= 10 else 0
        f['roc_20'] = (self.close[idx]/self.close[idx-20]-1)*100 if idx >= 20 else 0

        # CMO (Chande Momentum Oscillator, 14)
        if idx >= 14:
            ups = sum(max(0, self.close[j]-self.close[j-1]) for j in range(idx-13, idx+1))
            dns = sum(max(0, self.close[j-1]-self.close[j]) for j in range(idx-13, idx+1))
            f['cmo_14'] = (ups - dns) / (ups + dns) * 100 if (ups + dns) > 0 else 0
        else:
            f['cmo_14'] = 0

        # PPO (Percentage Price Oscillator)
        e12 = self._safe(self.ema12, idx, 0)
        e26 = self._safe(self.ema26, idx, 0)
        f['ppo'] = (e12 - e26) / e26 * 100 if e26 > 0 else 0

        # Awesome Oscillator简化
        if idx >= 34:
            mid5 = np.mean((self.high[idx-4:idx+1] + self.low[idx-4:idx+1]) / 2)
            mid34 = np.mean((self.high[idx-33:idx+1] + self.low[idx-33:idx+1]) / 2)
            f['awesome_osc'] = (mid5 - mid34) / self.close[idx] * 100
        else:
            f['awesome_osc'] = 0

        return f

    # ================================================================
    # 八、支撑阻力类 (10个)
    # ================================================================
    def _support_resistance_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        price = c[idx]
        f = {}

        # 距离近期支撑/阻力
        lookback = min(60, idx)
        recent_highs = []
        recent_lows = []
        for j in range(idx-lookback, idx):
            if j >= 2 and h[j] > h[j-1] and h[j] > h[j+1]:
                recent_highs.append(h[j])
            if j >= 2 and l[j] < l[j-1] and l[j] < l[j+1]:
                recent_lows.append(l[j])

        # 距离最近阻力
        resistances = [r for r in recent_highs if r > price]
        if resistances:
            f['dist_to_resistance_pct'] = (min(resistances) - price) / price * 100
        else:
            f['dist_to_resistance_pct'] = 10  # 无阻力

        # 距离最近支撑
        supports = [s for s in recent_lows if s < price]
        if supports:
            f['dist_to_support_pct'] = (price - max(supports)) / price * 100
        else:
            f['dist_to_support_pct'] = 10  # 无支撑

        # 支撑/阻力密度
        price_range = price * 0.05  # ±5%内
        f['resistance_density'] = sum(1 for r in recent_highs if abs(r - price) < price_range)
        f['support_density'] = sum(1 for s in recent_lows if abs(s - price) < price_range)

        # Pivot点
        f['pivot_r1_dist'] = 0
        f['pivot_s1_dist'] = 0
        if idx >= 1:
            pivot = (h[idx-1] + l[idx-1] + c[idx-1]) / 3
            r1 = 2 * pivot - l[idx-1]
            s1 = 2 * pivot - h[idx-1]
            f['pivot_r1_dist'] = (r1 - price) / price * 100
            f['pivot_s1_dist'] = (price - s1) / price * 100

        # 价格在前N日高低之间的位置
        f['position_in_5d_range'] = (price - np.min(l[idx-4:idx+1])) / (np.max(h[idx-4:idx+1]) - np.min(l[idx-4:idx+1])) if np.max(h[idx-4:idx+1]) > np.min(l[idx-4:idx+1]) else 0.5

        # 突破前高天数
        prev_high = np.max(h[idx-20:idx])
        f['above_prev_20d_high'] = 1 if price > prev_high else 0
        f['pct_above_prev_20d_high'] = (price / prev_high - 1) * 100

        return f

    # ================================================================
    # 九、频域/周期性类 (25个)
    # ================================================================
    def _frequency_factors(self, idx: int) -> Dict[str, float]:
        f = {}
        rets = self.log_ret[max(1, idx-59):idx+1]
        if len(rets) < 20:
            for k in ['fft_dominant_period', 'fft_dominant_power', 'fft_power_ratio_low',
                       'fft_power_ratio_mid', 'fft_power_ratio_high', 'spectral_entropy',
                       'spectral_centroid', 'fft_peak_count', 'fft_concentration',
                       'price_dom_period', 'price_spectral_entropy']:
                f[k] = 0
            return f

        # FFT on returns
        n_fft = len(rets)
        fft_vals = np.fft.rfft(rets - np.mean(rets))
        power = np.abs(fft_vals)**2
        freqs = np.fft.rfftfreq(n_fft)

        if len(power) > 1 and np.sum(power[1:]) > 0:
            # 主频
            dom_idx = np.argmax(power[1:]) + 1
            f['fft_dominant_period'] = 1.0 / freqs[dom_idx] if freqs[dom_idx] > 0 else n_fft
            f['fft_dominant_power'] = power[dom_idx] / np.sum(power[1:])

            # 低/中/高频能量比
            n_bins = len(power) - 1
            third = max(1, n_bins // 3)
            f['fft_power_ratio_low'] = np.sum(power[1:1+third]) / np.sum(power[1:])
            f['fft_power_ratio_mid'] = np.sum(power[1+third:1+2*third]) / np.sum(power[1:])
            f['fft_power_ratio_high'] = np.sum(power[1+2*third:]) / np.sum(power[1:])

            # 谱熵
            p_norm = power[1:] / np.sum(power[1:])
            p_norm = p_norm[p_norm > 0]
            f['spectral_entropy'] = -np.sum(p_norm * np.log2(p_norm)) / np.log2(len(p_norm)) if len(p_norm) > 1 else 1

            # 谱质心
            f['spectral_centroid'] = np.sum(freqs[1:len(power)] * power[1:]) / np.sum(power[1:])

            # 谱峰数
            threshold = np.mean(power[1:]) + np.std(power[1:])
            f['fft_peak_count'] = np.sum(power[1:] > threshold)

            # 谱集中度(top 3频率占总能量比)
            sorted_power = np.sort(power[1:])[::-1]
            f['fft_concentration'] = np.sum(sorted_power[:3]) / np.sum(power[1:])
        else:
            for k in ['fft_dominant_period', 'fft_dominant_power', 'fft_power_ratio_low',
                       'fft_power_ratio_mid', 'fft_power_ratio_high', 'spectral_entropy',
                       'spectral_centroid', 'fft_peak_count', 'fft_concentration']:
                f[k] = 0

        # 对价格序列做FFT（检测价格周期性）
        prices = self.close[max(0, idx-59):idx+1]
        prices_detrend = prices - np.linspace(prices[0], prices[-1], len(prices))
        if len(prices_detrend) >= 20:
            fft_p = np.fft.rfft(prices_detrend)
            power_p = np.abs(fft_p)**2
            freqs_p = np.fft.rfftfreq(len(prices_detrend))
            if len(power_p) > 1 and np.sum(power_p[1:]) > 0:
                dom_p = np.argmax(power_p[1:]) + 1
                f['price_dom_period'] = 1.0 / freqs_p[dom_p] if freqs_p[dom_p] > 0 else len(prices_detrend)
                p_norm_p = power_p[1:] / np.sum(power_p[1:])
                p_norm_p = p_norm_p[p_norm_p > 0]
                f['price_spectral_entropy'] = -np.sum(p_norm_p * np.log2(p_norm_p)) / np.log2(len(p_norm_p)) if len(p_norm_p) > 1 else 1
            else:
                f['price_dom_period'] = 0
                f['price_spectral_entropy'] = 1
        else:
            f['price_dom_period'] = 0
            f['price_spectral_entropy'] = 1

        # Hilbert变换相关（瞬时幅度/频率的稳定性）
        try:
            from scipy.signal import hilbert as _hilbert
            analytic = _hilbert(rets)
            inst_amp = np.abs(analytic)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) / (2 * np.pi)
            f['hilbert_amp_mean'] = np.mean(inst_amp)
            f['hilbert_amp_std'] = np.std(inst_amp)
            f['hilbert_freq_mean'] = np.mean(inst_freq) if len(inst_freq) > 0 else 0
            f['hilbert_freq_std'] = np.std(inst_freq) if len(inst_freq) > 0 else 0
            f['hilbert_amp_cv'] = f['hilbert_amp_std'] / f['hilbert_amp_mean'] if f['hilbert_amp_mean'] > 0 else 0
        except ImportError:
            f['hilbert_amp_mean'] = 0
            f['hilbert_amp_std'] = 0
            f['hilbert_freq_mean'] = 0
            f['hilbert_freq_std'] = 0
            f['hilbert_amp_cv'] = 0

        # 自相关函数衰减特征
        if len(rets) >= 15:
            acf_vals = []
            for lag in range(1, 11):
                if len(rets) > lag + 1:
                    corr = np.corrcoef(rets[:-lag], rets[lag:])[0, 1]
                    acf_vals.append(corr if not np.isnan(corr) else 0)
                else:
                    acf_vals.append(0)
            f['acf_sum_abs'] = np.sum(np.abs(acf_vals))
            f['acf_first_negative'] = next((i+1 for i, v in enumerate(acf_vals) if v < 0), 10)
            abs_acf = [abs(v) for v in acf_vals]
            f['acf_decay'] = abs_acf[-1] / abs_acf[0] if abs_acf[0] > 0.01 else 1
        else:
            f['acf_sum_abs'] = 0
            f['acf_first_negative'] = 10
            f['acf_decay'] = 1

        return f

    # ================================================================
    # 十、高级统计/检验类 (25个)
    # ================================================================
    def _advanced_stat_factors(self, idx: int) -> Dict[str, float]:
        f = {}
        c = self.close

        # ADF检验（单位根）- 价格是否平稳
        prices_60 = c[max(0, idx-59):idx+1]
        if len(prices_60) >= 30:
            try:
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(prices_60, maxlag=5, autolag=None)
                f['adf_stat'] = result[0]
                f['adf_pvalue'] = result[1]
            except (ImportError, Exception):
                f['adf_stat'] = 0
                f['adf_pvalue'] = 0.5
        else:
            f['adf_stat'] = 0
            f['adf_pvalue'] = 0.5

        # Hurst指数 (R/S法, 60日窗口)
        f['hurst_60d'] = self._compute_hurst(c, idx, 60)
        f['hurst_120d'] = self._compute_hurst(c, idx, 120)

        # DFA (Detrended Fluctuation Analysis) 简化版
        rets_60 = self.log_ret[max(1, idx-59):idx+1]
        f['dfa_alpha'] = self._compute_dfa(rets_60)

        # 多分形谱宽度简化
        f['mf_width'] = abs(self._compute_hurst(c, idx, 60) - self._compute_hurst(c, idx, 30)) if idx >= 60 else 0

        # 收益率分布检验
        rets_30 = self.log_ret[max(1, idx-29):idx+1]
        if len(rets_30) >= 10:
            # Jarque-Bera
            s = float(np.mean((rets_30 - np.mean(rets_30))**3) / (np.std(rets_30)**3 + 1e-10))
            k = float(np.mean((rets_30 - np.mean(rets_30))**4) / (np.std(rets_30)**4 + 1e-10)) - 3
            n = len(rets_30)
            f['jb_stat'] = n/6 * (s**2 + k**2/4)

            # 下行/上行半方差
            neg_rets = rets_30[rets_30 < 0]
            pos_rets = rets_30[rets_30 > 0]
            f['semivar_down'] = np.var(neg_rets) if len(neg_rets) > 1 else 0
            f['semivar_up'] = np.var(pos_rets) if len(pos_rets) > 1 else 0
            f['semivar_ratio'] = f['semivar_down'] / f['semivar_up'] if f['semivar_up'] > 0 else 1

            # Gain/Loss不对称
            avg_gain = np.mean(pos_rets) if len(pos_rets) > 0 else 0
            avg_loss = np.mean(np.abs(neg_rets)) if len(neg_rets) > 0 else 0
            f['gain_loss_ratio'] = avg_gain / avg_loss if avg_loss > 0 else 1

            # 涨跌天数的二项分布检验
            n_up = np.sum(rets_30 > 0)
            p_up = n_up / len(rets_30)
            f['up_prob'] = p_up
            # 偏离50%的显著性
            f['up_prob_z'] = (p_up - 0.5) / np.sqrt(0.25 / len(rets_30))
        else:
            for k_name in ['jb_stat', 'semivar_down', 'semivar_up', 'semivar_ratio',
                           'gain_loss_ratio', 'up_prob', 'up_prob_z']:
                f[k_name] = 0

        # LZ复杂度 (Lempel-Ziv)
        if len(rets_30) >= 10:
            binary = ''.join(['1' if r > 0 else '0' for r in rets_30])
            f['lz_complexity'] = self._lempel_ziv_complexity(binary) / (len(binary) / np.log2(len(binary)) + 1e-10)
        else:
            f['lz_complexity'] = 0

        # 条件波动率变化 (ARCH效应)
        if len(rets_30) >= 10:
            sq_rets = rets_30**2
            if np.std(sq_rets) > 0:
                f['arch_corr'] = np.corrcoef(sq_rets[:-1], sq_rets[1:])[0, 1]
            else:
                f['arch_corr'] = 0
        else:
            f['arch_corr'] = 0

        # 最大回撤恢复速度
        peak_idx = max(0, idx - 59)
        peak_val = c[peak_idx]
        max_dd_idx = peak_idx
        max_dd = 0
        for j in range(peak_idx, idx + 1):
            if c[j] > peak_val:
                peak_val = c[j]
                peak_idx = j
            dd = (c[j] / peak_val - 1) * 100
            if dd < max_dd:
                max_dd = dd
                max_dd_idx = j
        # 从最大回撤点到当前的恢复比例
        if max_dd < -1 and max_dd_idx < idx:
            recovery = (c[idx] / c[max_dd_idx] - 1) * 100
            f['dd_recovery_pct'] = recovery
            f['dd_recovery_speed'] = recovery / (idx - max_dd_idx) if idx > max_dd_idx else 0
        else:
            f['dd_recovery_pct'] = 0
            f['dd_recovery_speed'] = 0

        # 价格与均值的Z-score
        if idx >= 60:
            mean_60 = np.mean(c[idx-59:idx+1])
            std_60 = np.std(c[idx-59:idx+1])
            f['price_zscore_60d'] = (c[idx] - mean_60) / std_60 if std_60 > 0 else 0
        else:
            f['price_zscore_60d'] = 0

        if idx >= 20:
            mean_20 = np.mean(c[idx-19:idx+1])
            std_20 = np.std(c[idx-19:idx+1])
            f['price_zscore_20d'] = (c[idx] - mean_20) / std_20 if std_20 > 0 else 0
        else:
            f['price_zscore_20d'] = 0

        return f

    # ================================================================
    # 十一、价格行为微观类 (25个)
    # ================================================================
    def _micro_price_action_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # 连续同色K线
        bullish_streak = 0
        for j in range(idx, max(0, idx-20), -1):
            if c[j] > c[j-1]:
                bullish_streak += 1
            else:
                break
        f['bullish_streak'] = bullish_streak

        bearish_streak = 0
        for j in range(idx, max(0, idx-20), -1):
            if c[j] < c[j-1]:
                bearish_streak += 1
            else:
                break
        f['bearish_streak'] = bearish_streak

        # Doji频率（实体<振幅20%）
        doji_count = 0
        for j in range(idx-9, idx+1):
            rng = h[j] - l[j]
            body = abs(c[j] - c[j-1])
            if rng > 0 and body / rng < 0.2:
                doji_count += 1
        f['doji_freq_10d'] = doji_count / 10

        # 大阳/大阴线频率（body > 振幅60%）
        big_bull = 0
        big_bear = 0
        for j in range(idx-9, idx+1):
            rng = h[j] - l[j]
            if rng > 0:
                body = c[j] - c[j-1]
                if body / rng > 0.6:
                    big_bull += 1
                elif body / rng < -0.6:
                    big_bear += 1
        f['big_bull_freq_10d'] = big_bull / 10
        f['big_bear_freq_10d'] = big_bear / 10

        # 吞没形态频率
        engulf_count = 0
        for j in range(idx-9, idx+1):
            if j < 1:
                continue
            body_prev = c[j-1] - c[j-2] if j >= 2 else 0
            body_curr = c[j] - c[j-1]
            if body_prev < 0 and body_curr > 0 and abs(body_curr) > abs(body_prev):
                engulf_count += 1
            elif body_prev > 0 and body_curr < 0 and abs(body_curr) > abs(body_prev):
                engulf_count += 1
        f['engulf_freq_10d'] = engulf_count / 10

        # 锤子线/射击星频率
        hammer_count = 0
        shooting_count = 0
        for j in range(idx-9, idx+1):
            rng = h[j] - l[j]
            if rng <= 0:
                continue
            body = abs(c[j] - c[j-1])
            upper = h[j] - max(c[j], c[j-1])
            lower = min(c[j], c[j-1]) - l[j]
            if lower > body * 2 and upper < body * 0.5:
                hammer_count += 1
            if upper > body * 2 and lower < body * 0.5:
                shooting_count += 1
        f['hammer_freq_10d'] = hammer_count / 10
        f['shooting_star_freq_10d'] = shooting_count / 10

        # 价格缺口（gap）统计
        up_gaps = 0
        down_gaps = 0
        gap_sizes = []
        for j in range(idx-19, idx+1):
            gap = (c[j] - c[j-1]) / c[j-1] * 100 if c[j-1] > 0 else 0
            if gap > 1.0:
                up_gaps += 1
                gap_sizes.append(gap)
            elif gap < -1.0:
                down_gaps += 1
                gap_sizes.append(gap)
        f['up_gap_freq_20d'] = up_gaps / 20
        f['down_gap_freq_20d'] = down_gaps / 20
        f['avg_gap_size'] = np.mean(np.abs(gap_sizes)) if gap_sizes else 0

        # 日内反转频率（开盘涨/收盘跌或反之）
        # 用 close vs prev_close 模拟
        intraday_reversal = 0
        for j in range(idx-9, idx+1):
            morning = (h[j] + l[j]) / 2 - c[j-1]  # 日中相对昨收
            evening = c[j] - c[j-1]  # 收盘相对昨收
            if morning * evening < 0:  # 日中和收盘方向相反
                intraday_reversal += 1
        f['intraday_reversal_freq'] = intraday_reversal / 10

        # 尾盘强度（收盘在日内位置的稳定性）
        close_positions = [(c[j] - l[j]) / (h[j] - l[j]) if h[j] > l[j] else 0.5 for j in range(idx-9, idx+1)]
        f['close_position_mean_10d'] = np.mean(close_positions)
        f['close_position_std_10d'] = np.std(close_positions)

        # 真实波幅稳定性
        tr_vals = self.tr[max(0, idx-19):idx+1]
        f['tr_cv_20d'] = np.std(tr_vals) / np.mean(tr_vals) if np.mean(tr_vals) > 0 else 0

        # 价格离前高/前低的距离
        hh_20 = np.max(h[idx-19:idx+1])
        ll_20 = np.min(l[idx-19:idx+1])
        f['pct_from_20d_high'] = (c[idx] / hh_20 - 1) * 100
        f['pct_from_20d_low'] = (c[idx] / ll_20 - 1) * 100

        # 平均阳线/阴线长度比
        bull_bodies = []
        bear_bodies = []
        for j in range(idx-19, idx+1):
            body = c[j] - c[j-1]
            if body > 0:
                bull_bodies.append(body / c[j-1] * 100)
            elif body < 0:
                bear_bodies.append(abs(body) / c[j-1] * 100)
        avg_bull = np.mean(bull_bodies) if bull_bodies else 0
        avg_bear = np.mean(bear_bodies) if bear_bodies else 0
        f['bull_bear_body_ratio'] = avg_bull / avg_bear if avg_bear > 0 else 1

        # 连续创新高/低天数
        new_high_count = 0
        for j in range(idx, max(0, idx-20), -1):
            if h[j] >= np.max(h[max(0,j-5):j]) and j > 5:
                new_high_count += 1
            else:
                break
        f['consec_new_high'] = new_high_count

        return f

    # ================================================================
    # 十二、多周期/跨时间类 (20个)
    # ================================================================
    def _multi_timeframe_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # 不同周期的趋势方向一致性
        trends = []
        for period in [5, 10, 20, 60]:
            if idx >= period:
                ret = c[idx] / c[idx-period] - 1
                trends.append(1 if ret > 0 else -1)
        if len(trends) >= 2:
            f['trend_alignment'] = sum(1 for t in trends if t == trends[0]) / len(trends)
            f['trend_agreement_sign'] = np.mean(trends)
        else:
            f['trend_alignment'] = 0.5
            f['trend_agreement_sign'] = 0

        # 不同周期MA排列度
        ma_vals = []
        for ma in [self.ma5, self.ma10, self.ma20, self.ma60, self.ma120]:
            v = self._safe(ma, idx)
            if not np.isnan(v):
                ma_vals.append(v)
        if len(ma_vals) >= 3:
            # 完美多头排列得分
            bullish_order = sum(1 for i in range(len(ma_vals)-1) if ma_vals[i] > ma_vals[i+1])
            f['ma_bullish_alignment'] = bullish_order / (len(ma_vals) - 1)
            # MA间距标准化
            ma_spreads = [(ma_vals[i] - ma_vals[i+1]) / c[idx] * 100 for i in range(len(ma_vals)-1)]
            f['ma_spread_mean'] = np.mean(ma_spreads)
            f['ma_spread_std'] = np.std(ma_spreads)
        else:
            f['ma_bullish_alignment'] = 0.5
            f['ma_spread_mean'] = 0
            f['ma_spread_std'] = 0

        # 短期vs长期收益率对比
        f['ret_ratio_5_20'] = 0
        if idx >= 20:
            r5 = c[idx] / c[idx-5] - 1
            r20 = c[idx] / c[idx-20] - 1
            f['ret_ratio_5_20'] = r5 / r20 if abs(r20) > 0.001 else 0

        f['ret_ratio_10_60'] = 0
        if idx >= 60:
            r10 = c[idx] / c[idx-10] - 1
            r60 = c[idx] / c[idx-60] - 1
            f['ret_ratio_10_60'] = r10 / r60 if abs(r60) > 0.001 else 0

        # 短期波动率 vs 长期波动率
        vol_5 = np.std(self.log_ret[max(1,idx-4):idx+1])
        vol_20 = np.std(self.log_ret[max(1,idx-19):idx+1])
        vol_60 = np.std(self.log_ret[max(1,idx-59):idx+1])
        f['vol_regime_5_60'] = vol_5 / vol_60 if vol_60 > 0 else 1

        # 周收益率（模拟）
        if idx >= 5:
            weekly_rets = [c[idx-j]/c[idx-j-5]-1 for j in range(0, min(20, idx-5), 5)]
            f['weekly_ret_mean'] = np.mean(weekly_rets) * 100
            f['weekly_ret_std'] = np.std(weekly_rets) * 100
            f['weekly_win_rate'] = sum(1 for r in weekly_rets if r > 0) / len(weekly_rets)
        else:
            f['weekly_ret_mean'] = 0
            f['weekly_ret_std'] = 0
            f['weekly_win_rate'] = 0.5

        # 隔夜跳空特征
        gaps = []
        for j in range(max(1, idx-19), idx+1):
            gap = (c[j] - c[j-1]) / c[j-1] * 100  # 简化：用收盘价差近似
            gaps.append(gap)
        f['gap_mean_20d'] = np.mean(gaps)
        f['gap_std_20d'] = np.std(gaps)
        f['gap_positive_ratio'] = sum(1 for g in gaps if g > 0) / len(gaps)

        # 趋势加速度（不同周期的斜率变化）
        if idx >= 20:
            slope_recent = (c[idx]/c[idx-5]-1) * 100
            slope_prev = (c[idx-5]/c[idx-10]-1) * 100 if idx >= 10 else 0
            f['trend_acceleration_5d'] = slope_recent - slope_prev
        else:
            f['trend_acceleration_5d'] = 0

        if idx >= 40:
            slope_recent = (c[idx]/c[idx-10]-1) * 100
            slope_prev = (c[idx-10]/c[idx-20]-1) * 100
            f['trend_acceleration_10d'] = slope_recent - slope_prev
        else:
            f['trend_acceleration_10d'] = 0

        return f

    # ================================================================
    # 十三、成交量微结构类 (15个)
    # ================================================================
    def _volume_micro_factors(self, idx: int) -> Dict[str, float]:
        c, v = self.close, self.volume
        f = {}

        # 量价配合度（价涨量增 vs 价涨量缩）
        vp_agree = 0
        vp_disagree = 0
        for j in range(max(1, idx-19), idx+1):
            price_up = c[j] > c[j-1]
            vol_up = v[j] > v[j-1]
            if (price_up and vol_up) or (not price_up and not vol_up):
                vp_agree += 1
            else:
                vp_disagree += 1
        total = vp_agree + vp_disagree
        f['vol_price_agreement'] = vp_agree / total if total > 0 else 0.5

        # 放量上涨 vs 放量下跌
        vm20 = self._safe(self.vol_ma20, idx, 1)
        surge_up = 0
        surge_down = 0
        for j in range(max(1, idx-19), idx+1):
            if v[j] > vm20 * 1.5:
                if c[j] > c[j-1]:
                    surge_up += 1
                else:
                    surge_down += 1
        f['vol_surge_up_ratio'] = surge_up / 20
        f['vol_surge_down_ratio'] = surge_down / 20

        # 缩量上涨 vs 缩量下跌
        thin_up = 0
        thin_down = 0
        for j in range(max(1, idx-19), idx+1):
            if v[j] < vm20 * 0.6:
                if c[j] > c[j-1]:
                    thin_up += 1
                else:
                    thin_down += 1
        f['vol_thin_up_ratio'] = thin_up / 20
        f['vol_thin_down_ratio'] = thin_down / 20

        # 成交量衰减速度
        if idx >= 10:
            v_first5 = np.mean(v[idx-9:idx-4])
            v_last5 = np.mean(v[idx-4:idx+1])
            f['vol_decay_rate'] = (v_last5 / v_first5 - 1) * 100 if v_first5 > 0 else 0
        else:
            f['vol_decay_rate'] = 0

        # 成交量集中在涨日还是跌日
        up_vol_total = sum(v[j] for j in range(max(1, idx-19), idx+1) if c[j] > c[j-1])
        dn_vol_total = sum(v[j] for j in range(max(1, idx-19), idx+1) if c[j] < c[j-1])
        f['vol_up_concentration'] = up_vol_total / (up_vol_total + dn_vol_total) if (up_vol_total + dn_vol_total) > 0 else 0.5

        # 成交量的自相关性
        v_20 = v[max(0, idx-19):idx+1].astype(float)
        if len(v_20) >= 5 and np.std(v_20) > 0:
            f['vol_autocorr'] = np.corrcoef(v_20[:-1], v_20[1:])[0, 1]
        else:
            f['vol_autocorr'] = 0

        # VWAP偏离
        if idx >= 20:
            vwap = np.sum(c[idx-19:idx+1] * v[idx-19:idx+1]) / np.sum(v[idx-19:idx+1]) if np.sum(v[idx-19:idx+1]) > 0 else c[idx]
            f['vwap_deviation'] = (c[idx] / vwap - 1) * 100
        else:
            f['vwap_deviation'] = 0

        # 高量日价格影响力
        if idx >= 20:
            impacts = []
            for j in range(idx-19, idx+1):
                if vm20 > 0 and v[j] > vm20 * 1.3:
                    impact = abs(c[j] - c[j-1]) / c[j-1] * 100
                    impacts.append(impact)
            f['high_vol_price_impact'] = np.mean(impacts) if impacts else 0
        else:
            f['high_vol_price_impact'] = 0

        # 换手率近似（volume / 近期平均volume的标准化）
        f['turnover_ratio'] = v[idx] / np.mean(v[max(0, idx-59):idx+1]) if np.mean(v[max(0, idx-59):idx+1]) > 0 else 1
        f['turnover_trend'] = 0
        if idx >= 20:
            tr_recent = np.mean(v[idx-4:idx+1]) / np.mean(v[max(0, idx-59):idx+1])
            tr_prev = np.mean(v[idx-14:idx-9]) / np.mean(v[max(0, idx-59):idx+1]) if idx >= 14 else tr_recent
            f['turnover_trend'] = (tr_recent / tr_prev - 1) * 100 if tr_prev > 0 else 0

        return f

    # ================================================================
    # 十四、趋势质量/成熟度类 (20个)
    # ================================================================
    def _trend_quality_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # 趋势持续天数（价格持续在MA20上方/下方）
        above_ma20_days = 0
        for j in range(idx, max(0, idx-120), -1):
            if not np.isnan(self.ma20[j]) and c[j] > self.ma20[j]:
                above_ma20_days += 1
            else:
                break
        f['days_above_ma20'] = above_ma20_days

        below_ma20_days = 0
        for j in range(idx, max(0, idx-120), -1):
            if not np.isnan(self.ma20[j]) and c[j] < self.ma20[j]:
                below_ma20_days += 1
            else:
                break
        f['days_below_ma20'] = below_ma20_days

        # 趋势持续天数（MA60上方/下方）
        above_ma60_days = 0
        for j in range(idx, max(0, idx-200), -1):
            if not np.isnan(self.ma60[j]) and c[j] > self.ma60[j]:
                above_ma60_days += 1
            else:
                break
        f['days_above_ma60'] = above_ma60_days

        # 趋势年龄（连续正收益的天数加权）
        trend_age = 0
        trend_strength = 0
        for j in range(idx, max(0, idx-60), -1):
            ret = (c[j] / c[j-1] - 1) if c[j-1] > 0 else 0
            if ret > 0:
                trend_age += 1
                trend_strength += ret
            elif ret < -0.02:  # 2%以上的回调中断
                break
        f['trend_age_strict'] = trend_age
        f['trend_cumret_strict'] = trend_strength * 100

        # 趋势强度指数（方向一致性 × 斜率）
        if idx >= 20:
            x = np.arange(20)
            y = c[idx-19:idx+1]
            slope, _ = np.polyfit(x, y, 1)
            direction_consistency = sum(1 for j in range(idx-18, idx+1) if (c[j]-c[j-1]) * slope > 0) / 19
            f['trend_strength_index'] = direction_consistency * abs(slope) / np.mean(y) * 1000
        else:
            f['trend_strength_index'] = 0

        # 价格在通道内的位置（回归通道）
        if idx >= 20:
            x = np.arange(20)
            y = c[idx-19:idx+1]
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            residual = y[-1] - y_pred[-1]
            std_resid = np.std(y - y_pred)
            f['channel_position'] = residual / std_resid if std_resid > 0 else 0
        else:
            f['channel_position'] = 0

        # MA120斜率角度（趋势成熟度）
        if idx >= 120:
            ma120_start = self._safe(self.ma120, idx-60)
            ma120_end = self._safe(self.ma120, idx)
            if ma120_start > 0 and not np.isnan(ma120_start):
                f['ma120_momentum'] = (ma120_end / ma120_start - 1) * 100
            else:
                f['ma120_momentum'] = 0
        else:
            f['ma120_momentum'] = 0

        # 新高/新低的频率和间隔
        new_high_days = []
        new_low_days = []
        for j in range(max(0, idx-59), idx+1):
            if j >= 20:
                if h[j] >= np.max(h[j-20:j]):
                    new_high_days.append(j)
                if l[j] <= np.min(l[j-20:j]):
                    new_low_days.append(j)
        f['new_high_count_60d'] = len(new_high_days)
        f['new_low_count_60d'] = len(new_low_days)
        f['new_high_recency'] = idx - new_high_days[-1] if new_high_days else 60
        f['new_low_recency'] = idx - new_low_days[-1] if new_low_days else 60

        # 趋势方向的连续性（Efficiency Ratio的变体）
        if idx >= 20:
            net_move = abs(c[idx] - c[idx-20])
            total_path = sum(abs(c[j] - c[j-1]) for j in range(idx-19, idx+1))
            f['path_efficiency_20d'] = net_move / total_path if total_path > 0 else 0
        else:
            f['path_efficiency_20d'] = 0

        if idx >= 60:
            net_move = abs(c[idx] - c[idx-60])
            total_path = sum(abs(c[j] - c[j-1]) for j in range(idx-59, idx+1))
            f['path_efficiency_60d'] = net_move / total_path if total_path > 0 else 0
        else:
            f['path_efficiency_60d'] = 0

        # 趋势的"噪音度"（收益率标准差 / 平均绝对收益率）
        rets_20 = self.log_ret[max(1, idx-19):idx+1]
        if len(rets_20) >= 5:
            f['trend_noise_20d'] = np.std(rets_20) / (np.mean(np.abs(rets_20)) + 1e-10)
        else:
            f['trend_noise_20d'] = 1

        # 最近大幅波动距今天数
        last_big_move = 60
        for j in range(idx, max(0, idx-60), -1):
            if abs(c[j] / c[j-1] - 1) > 0.03:  # 3%以上日波动
                last_big_move = idx - j
                break
        f['days_since_big_move'] = last_big_move

        return f

    # ================================================================
    # 十五、扩展振荡器类 (15个)
    # ================================================================
    def _extended_oscillator_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # Stochastic D (K的3日SMA)
        if idx >= 16:
            k_vals = self.stoch_k[idx-2:idx+1]
            f['stoch_d'] = np.nanmean(k_vals)
            f['stoch_k_d_diff'] = self._safe(self.stoch_k, idx, 50) - f['stoch_d']
        else:
            f['stoch_d'] = 50; f['stoch_k_d_diff'] = 0

        # RSI(14) vs RSI(29) 差异
        f['rsi_14_29_diff'] = self._safe(self.rsi14, idx, 50) - self._safe(self.rsi29, idx, 50)

        # MACD柱状图方向变化
        macd_hist_changes = 0
        for j in range(max(1, idx-9), idx+1):
            h1 = self._safe(self.macd_hist, j, 0)
            h2 = self._safe(self.macd_hist, j-1, 0)
            if h1 * h2 < 0:
                macd_hist_changes += 1
        f['macd_hist_cross_freq'] = macd_hist_changes / 10

        # MACD柱状图连续正/负天数
        macd_pos_streak = 0
        for j in range(idx, max(0, idx-30), -1):
            if self._safe(self.macd_hist, j, 0) > 0:
                macd_pos_streak += 1
            else:
                break
        f['macd_hist_pos_streak'] = macd_pos_streak

        macd_neg_streak = 0
        for j in range(idx, max(0, idx-30), -1):
            if self._safe(self.macd_hist, j, 0) < 0:
                macd_neg_streak += 1
            else:
                break
        f['macd_hist_neg_streak'] = macd_neg_streak

        # RSI背离简化检测 (价格创新高但RSI没有)
        if idx >= 20:
            price_high_20 = np.argmax(h[idx-19:idx+1]) + idx - 19
            rsi_at_price_high = self._safe(self.rsi29, price_high_20, 50)
            f['rsi_divergence'] = self._safe(self.rsi29, idx, 50) - rsi_at_price_high
        else:
            f['rsi_divergence'] = 0

        # Aroon指标
        if idx >= 25:
            aroon_up = ((25 - (idx - (idx-24 + np.argmax(h[idx-24:idx+1])))) / 25) * 100
            aroon_down = ((25 - (idx - (idx-24 + np.argmin(l[idx-24:idx+1])))) / 25) * 100
            f['aroon_up'] = aroon_up
            f['aroon_down'] = aroon_down
            f['aroon_osc'] = aroon_up - aroon_down
        else:
            f['aroon_up'] = 50; f['aroon_down'] = 50; f['aroon_osc'] = 0

        # Ultimate Oscillator 简化
        if idx >= 28:
            bp_sum7 = sum(c[j] - min(l[j], c[j-1]) for j in range(idx-6, idx+1))
            tr_sum7 = sum(max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(idx-6, idx+1))
            bp_sum14 = sum(c[j] - min(l[j], c[j-1]) for j in range(idx-13, idx+1))
            tr_sum14 = sum(max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(idx-13, idx+1))
            bp_sum28 = sum(c[j] - min(l[j], c[j-1]) for j in range(idx-27, idx+1))
            tr_sum28 = sum(max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(idx-27, idx+1))
            avg7 = bp_sum7 / tr_sum7 if tr_sum7 > 0 else 0.5
            avg14 = bp_sum14 / tr_sum14 if tr_sum14 > 0 else 0.5
            avg28 = bp_sum28 / tr_sum28 if tr_sum28 > 0 else 0.5
            f['ultimate_osc'] = (4*avg7 + 2*avg14 + avg28) / 7 * 100
        else:
            f['ultimate_osc'] = 50

        # TSI (True Strength Index) 简化
        if idx >= 25:
            mom = c[idx] - c[idx-1]
            f['tsi'] = mom / (abs(mom) + self._safe(self.atr14, idx, 1)) * 100
        else:
            f['tsi'] = 0

        return f

    # ================================================================
    # 十六、扩展多周期类 (20个)
    # ================================================================
    def _extended_multi_tf_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # 不同周期RSI一致性
        rsi14_dir = 1 if self._safe(self.rsi14, idx, 50) > 50 else -1
        rsi29_dir = 1 if self._safe(self.rsi29, idx, 50) > 50 else -1
        f['rsi_multi_agree'] = 1 if rsi14_dir == rsi29_dir else 0

        # 不同周期MACD一致性
        macd_dir = 1 if self._safe(self.macd_hist, idx, 0) > 0 else -1
        # 模拟周线MACD（用长周期EMA差）
        ema50_v = self._safe(self.ema50, idx, 0)
        ema20_v = self._safe(self.ema20, idx, 0)
        weekly_macd_dir = 1 if ema20_v > ema50_v else -1
        f['macd_multi_agree'] = 1 if macd_dir == weekly_macd_dir else 0

        # 价格在不同MA的上下分布
        above_count = 0
        for ma in [self.ma5, self.ma10, self.ma20, self.ma60, self.ma120]:
            v = self._safe(ma, idx)
            if not np.isnan(v) and c[idx] > v:
                above_count += 1
        f['above_ma_count'] = above_count
        f['above_ma_ratio'] = above_count / 5

        # 多周期动量一致性
        mom_signs = []
        for period in [3, 5, 10, 20, 60]:
            if idx >= period:
                mom_signs.append(1 if c[idx] > c[idx-period] else -1)
        if mom_signs:
            f['momentum_consistency'] = sum(1 for s in mom_signs if s == mom_signs[0]) / len(mom_signs)
            f['momentum_sum'] = sum(mom_signs)
        else:
            f['momentum_consistency'] = 0.5; f['momentum_sum'] = 0

        # 短期强于长期的程度
        if idx >= 60:
            r5 = (c[idx] / c[idx-5] - 1) * 252 / 5  # 年化
            r20 = (c[idx] / c[idx-20] - 1) * 252 / 20
            r60 = (c[idx] / c[idx-60] - 1) * 252 / 60
            f['short_vs_long_mom'] = r5 - r60
            f['mid_vs_long_mom'] = r20 - r60
        else:
            f['short_vs_long_mom'] = 0; f['mid_vs_long_mom'] = 0

        # MA扇形展开/收缩
        mas = []
        for ma in [self.ma5, self.ma10, self.ma20, self.ma60]:
            v = self._safe(ma, idx)
            if not np.isnan(v):
                mas.append(v)
        if len(mas) >= 3:
            dists = [abs(mas[i] - mas[i+1]) / c[idx] * 100 for i in range(len(mas)-1)]
            f['ma_fan_width'] = sum(dists)
            # 5天前的扇形宽度
            mas_prev = []
            for ma in [self.ma5, self.ma10, self.ma20, self.ma60]:
                v = self._safe(ma, idx-5)
                if not np.isnan(v):
                    mas_prev.append(v)
            if len(mas_prev) == len(mas):
                dists_prev = [abs(mas_prev[i] - mas_prev[i+1]) / c[idx-5] * 100 for i in range(len(mas_prev)-1)]
                f['ma_fan_expanding'] = sum(dists) - sum(dists_prev)
            else:
                f['ma_fan_expanding'] = 0
        else:
            f['ma_fan_width'] = 0; f['ma_fan_expanding'] = 0

        # 周线模拟（5日周期）
        if idx >= 25:
            # 最近5周的收益
            week_rets = [(c[idx-i*5] / c[idx-(i+1)*5] - 1) * 100 for i in range(5) if idx-(i+1)*5 >= 0]
            f['weekly_win_streak'] = 0
            for r in week_rets:
                if r > 0:
                    f['weekly_win_streak'] += 1
                else:
                    break
            f['weekly_avg_ret'] = np.mean(week_rets) if week_rets else 0
            f['weekly_ret_consistency'] = sum(1 for r in week_rets if r > 0) / len(week_rets) if week_rets else 0.5
        else:
            f['weekly_win_streak'] = 0; f['weekly_avg_ret'] = 0; f['weekly_ret_consistency'] = 0.5

        # 月度模拟（20日周期）
        if idx >= 60:
            month_rets = [(c[idx-i*20] / c[idx-(i+1)*20] - 1) * 100 for i in range(3) if idx-(i+1)*20 >= 0]
            f['monthly_win_count'] = sum(1 for r in month_rets if r > 0)
            f['monthly_avg_ret'] = np.mean(month_rets) if month_rets else 0
        else:
            f['monthly_win_count'] = 0; f['monthly_avg_ret'] = 0

        return f

    # ================================================================
    # 十七、扩展支撑阻力/形态类 (15个)
    # ================================================================
    def _extended_sr_pattern_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # Pivot Point系列（经典/Fibonacci/Woodie）
        if idx >= 1:
            pivot = (h[idx-1] + l[idx-1] + c[idx-1]) / 3
            r1 = 2*pivot - l[idx-1]
            s1 = 2*pivot - h[idx-1]
            r2 = pivot + (h[idx-1] - l[idx-1])
            s2 = pivot - (h[idx-1] - l[idx-1])
            f['pivot_position'] = (c[idx] - s2) / (r2 - s2) if r2 > s2 else 0.5
            f['above_pivot'] = 1 if c[idx] > pivot else 0
            f['pivot_range_pct'] = (r2 - s2) / c[idx] * 100
        else:
            f['pivot_position'] = 0.5; f['above_pivot'] = 0; f['pivot_range_pct'] = 0

        # Donchian Channel位置
        if idx >= 20:
            don_high = np.max(h[idx-19:idx+1])
            don_low = np.min(l[idx-19:idx+1])
            f['donchian_position_20d'] = (c[idx] - don_low) / (don_high - don_low) if don_high > don_low else 0.5
        else:
            f['donchian_position_20d'] = 0.5

        if idx >= 55:
            don_high = np.max(h[idx-54:idx+1])
            don_low = np.min(l[idx-54:idx+1])
            f['donchian_position_55d'] = (c[idx] - don_low) / (don_high - don_low) if don_high > don_low else 0.5
        else:
            f['donchian_position_55d'] = 0.5

        # Keltner Channel位置
        if idx >= 20:
            kc_mid = self._safe(self.ema20, idx, c[idx])
            kc_atr = self._safe(self.atr20, idx, 0)
            kc_upper = kc_mid + 2 * kc_atr
            kc_lower = kc_mid - 2 * kc_atr
            f['keltner_position'] = (c[idx] - kc_lower) / (kc_upper - kc_lower) if kc_upper > kc_lower else 0.5
            # BB与Keltner挤压
            bb_inside = 1 if self._safe(self.bb_upper, idx, 999) < kc_upper and self._safe(self.bb_lower, idx, -999) > kc_lower else 0
            f['bb_keltner_squeeze'] = bb_inside
        else:
            f['keltner_position'] = 0.5; f['bb_keltner_squeeze'] = 0

        # 前N日高低点突破
        for period_name, period in [('breakout_10d', 10), ('breakout_20d', 20), ('breakout_60d', 60)]:
            if idx >= period:
                prev_high = np.max(h[idx-period:idx])
                prev_low = np.min(l[idx-period:idx])
                f[f'{period_name}_up'] = 1 if c[idx] > prev_high else 0
                f[f'{period_name}_down'] = 1 if c[idx] < prev_low else 0
            else:
                f[f'{period_name}_up'] = 0; f[f'{period_name}_down'] = 0

        # 价格密集区
        if idx >= 60:
            # 最近60日收盘价分布的众数区间
            hist, bins = np.histogram(c[idx-59:idx+1], bins=10)
            mode_bin = np.argmax(hist)
            mode_center = (bins[mode_bin] + bins[mode_bin+1]) / 2
            f['price_mode_dist_pct'] = (c[idx] - mode_center) / c[idx] * 100
        else:
            f['price_mode_dist_pct'] = 0

        return f

    # ================================================================
    # 十八、波动率衍生/高级类 (15个)
    # ================================================================
    def _extended_volatility_factors(self, idx: int) -> Dict[str, float]:
        c, h, l = self.close, self.high, self.low
        f = {}

        # 波动率锥（当前波动率在历史中的分位数）
        if idx >= 200:
            # 不同窗口的波动率百分位
            for name, window in [('vol_cone_10d', 10), ('vol_cone_20d', 20), ('vol_cone_60d', 60)]:
                current_vol = np.std(self.log_ret[max(1,idx-window+1):idx+1]) * np.sqrt(252)
                hist_vols = []
                for j in range(idx-199, idx-window+1, 5):
                    v = np.std(self.log_ret[max(1,j):j+window]) * np.sqrt(252)
                    hist_vols.append(v)
                if hist_vols:
                    f[name] = np.sum(np.array(hist_vols) < current_vol) / len(hist_vols)
                else:
                    f[name] = 0.5
        else:
            f['vol_cone_10d'] = 0.5; f['vol_cone_20d'] = 0.5; f['vol_cone_60d'] = 0.5

        # 波动率均值回归速度
        if idx >= 60:
            vols = [np.std(self.log_ret[max(1,idx-j-9):idx-j+1])*np.sqrt(252) for j in range(0, 50, 5)]
            if len(vols) >= 5:
                mean_vol = np.mean(vols)
                # 最近波动率偏离均值后回归的速度
                dev = vols[0] - mean_vol
                prev_dev = vols[1] - mean_vol if len(vols) > 1 else dev
                f['vol_mean_revert_speed'] = (prev_dev - dev) / (abs(prev_dev) + 0.001) if abs(prev_dev) > 0.001 else 0
            else:
                f['vol_mean_revert_speed'] = 0
        else:
            f['vol_mean_revert_speed'] = 0

        # 实现波动率 vs 隐含波动率的代理（用ATR近似）
        # 短期ATR / 长期ATR作为波动率期限结构
        atr_short = self._safe(self.atr14, idx, 1)
        atr_long = self._safe(self.atr20, idx, 1)
        f['vol_term_structure'] = atr_short / atr_long if atr_long > 0 else 1

        # 上行波动率 vs 下行波动率（20日）
        rets_20 = self.log_ret[max(1,idx-19):idx+1]
        if len(rets_20) >= 5:
            up_rets = rets_20[rets_20 > 0]
            dn_rets = rets_20[rets_20 < 0]
            f['vol_asymmetry'] = np.std(up_rets) / np.std(dn_rets) if len(dn_rets) > 1 and np.std(dn_rets) > 0 else 1
            f['vol_skew_proxy'] = (np.mean(up_rets)**2 - np.mean(dn_rets)**2) / (np.std(rets_20)**2 + 1e-10) if len(up_rets) > 0 and len(dn_rets) > 0 else 0
        else:
            f['vol_asymmetry'] = 1; f['vol_skew_proxy'] = 0

        # 波动率集聚度（连续高波动天数占比）
        if idx >= 20:
            daily_vols = np.abs(self.log_ret[max(1,idx-19):idx+1])
            med_vol = np.median(daily_vols)
            high_vol_days = daily_vols > med_vol * 1.5
            # 连续高波动的最长段
            max_cluster = 0; cluster = 0
            for hv in high_vol_days:
                if hv:
                    cluster += 1; max_cluster = max(max_cluster, cluster)
                else:
                    cluster = 0
            f['vol_cluster_max'] = max_cluster
            f['vol_cluster_ratio'] = np.sum(high_vol_days) / len(high_vol_days)
        else:
            f['vol_cluster_max'] = 0; f['vol_cluster_ratio'] = 0

        # 价格跳跃强度（大于2倍ATR的日波动）
        if idx >= 20:
            atr_val = self._safe(self.atr20, idx, 1)
            daily_moves = np.abs(c[idx-19:idx+1] - np.concatenate([[c[idx-20]], c[idx-19:idx]]))
            f['jump_count_20d'] = np.sum(daily_moves > 2 * atr_val)
            f['jump_intensity'] = np.mean(daily_moves[daily_moves > 2*atr_val]) / atr_val if np.sum(daily_moves > 2*atr_val) > 0 else 0
        else:
            f['jump_count_20d'] = 0; f['jump_intensity'] = 0

        # 波动率趋势（ATR在上升还是下降）
        if idx >= 20:
            x = np.arange(20)
            atr_vals = self.atr20[idx-19:idx+1]
            valid_mask = ~np.isnan(atr_vals)
            if np.sum(valid_mask) >= 10:
                slope = np.polyfit(x[valid_mask], atr_vals[valid_mask], 1)[0]
                f['atr_trend_slope'] = slope / np.nanmean(atr_vals) * 100
            else:
                f['atr_trend_slope'] = 0
        else:
            f['atr_trend_slope'] = 0

        return f

    # ================================================================
    # 十九、资金流/换手率/A股特有类 (15个)
    # ================================================================
    def _astock_specific_factors(self, idx: int) -> Dict[str, float]:
        c, h, l, v = self.close, self.high, self.low, self.volume
        f = {}

        # 换手率代理（用volume/近期均量标准化）
        if idx >= 60:
            avg_vol_60 = np.mean(v[idx-59:idx+1])
            f['turnover_proxy'] = v[idx] / avg_vol_60 if avg_vol_60 > 0 else 1
            f['turnover_5d'] = np.mean(v[idx-4:idx+1]) / avg_vol_60 if avg_vol_60 > 0 else 1
            f['turnover_percentile'] = np.sum(v[idx-59:idx+1] < v[idx]) / 60
        else:
            f['turnover_proxy'] = 1; f['turnover_5d'] = 1; f['turnover_percentile'] = 0.5

        # 换手率变化趋势
        if idx >= 20:
            tr_recent = np.mean(v[idx-4:idx+1])
            tr_prev = np.mean(v[idx-14:idx-9])
            f['turnover_change'] = (tr_recent / tr_prev - 1) * 100 if tr_prev > 0 else 0
        else:
            f['turnover_change'] = 0

        # 涨停/跌停近似检测（日涨幅>9.5% 或 <-9.5%）
        limit_up_count = 0
        limit_down_count = 0
        for j in range(max(1, idx-59), idx+1):
            ret_j = (c[j] / c[j-1] - 1) * 100
            if ret_j > 9.5:
                limit_up_count += 1
            elif ret_j < -9.5:
                limit_down_count += 1
        f['limit_up_count_60d'] = limit_up_count
        f['limit_down_count_60d'] = limit_down_count

        # 涨幅>5%的大阳线频率
        big_up_count = sum(1 for j in range(max(1, idx-19), idx+1) if (c[j]/c[j-1]-1)*100 > 5)
        big_down_count = sum(1 for j in range(max(1, idx-19), idx+1) if (c[j]/c[j-1]-1)*100 < -5)
        f['big_up_freq_20d'] = big_up_count / 20
        f['big_down_freq_20d'] = big_down_count / 20

        # T+1影响：隔夜收益vs日内收益
        if idx >= 10:
            overnight_rets = []  # open vs prev_close 的近似
            intraday_rets = []   # close vs open 的近似
            for j in range(idx-9, idx+1):
                # 用 (high+low)/2 近似开盘价
                approx_open = (h[j] + l[j] + c[j-1]) / 3
                overnight_rets.append((approx_open / c[j-1] - 1) * 100)
                intraday_rets.append((c[j] / approx_open - 1) * 100)
            f['overnight_ret_mean'] = np.mean(overnight_rets)
            f['intraday_ret_mean'] = np.mean(intraday_rets)
            f['overnight_vs_intraday'] = f['overnight_ret_mean'] - f['intraday_ret_mean']
        else:
            f['overnight_ret_mean'] = 0; f['intraday_ret_mean'] = 0; f['overnight_vs_intraday'] = 0

        # 缩量后放量（底部信号）
        if idx >= 10:
            vol_min_5d = np.min(v[idx-4:idx+1])
            vol_max_prev5d = np.max(v[idx-9:idx-4])
            f['vol_expansion_ratio'] = v[idx] / vol_min_5d if vol_min_5d > 0 else 1
        else:
            f['vol_expansion_ratio'] = 1

        return f

    # ================================================================
    # Hurst/DFA/LZ辅助计算
    # ================================================================
    def _compute_hurst(self, close, idx, window):
        if idx < window:
            return 0.5
        ts = close[idx-window+1:idx+1]
        if len(ts) < 20:
            return 0.5
        lags = range(2, min(20, len(ts)//5))
        tau = []
        rs_values = []
        for lag in lags:
            chunks = [ts[j:j+lag] for j in range(0, len(ts)-lag+1, lag)]
            rs_list = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_c = np.mean(chunk)
                dev = chunk - mean_c
                cum_dev = np.cumsum(dev)
                R = np.max(cum_dev) - np.min(cum_dev)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_list.append(R / S)
            if rs_list:
                tau.append(lag)
                rs_values.append(np.mean(rs_list))
        if len(tau) >= 3:
            log_tau = np.log(tau)
            log_rs = np.log(rs_values)
            slope = np.polyfit(log_tau, log_rs, 1)[0]
            return slope
        return 0.5

    @staticmethod
    def _compute_dfa(series):
        if len(series) < 20:
            return 0.5
        y = np.cumsum(series - np.mean(series))
        scales = [4, 8, 16, 32]
        scales = [s for s in scales if s < len(y) // 2]
        if len(scales) < 2:
            return 0.5
        flucts = []
        for s in scales:
            n_segs = len(y) // s
            if n_segs < 1:
                continue
            rms = 0
            for seg in range(n_segs):
                segment = y[seg*s:(seg+1)*s]
                x = np.arange(s)
                p = np.polyfit(x, segment, 1)
                trend = np.polyval(p, x)
                rms += np.mean((segment - trend)**2)
            flucts.append(np.sqrt(rms / n_segs))
        if len(flucts) >= 2 and all(f > 0 for f in flucts):
            log_s = np.log(scales[:len(flucts)])
            log_f = np.log(flucts)
            alpha = np.polyfit(log_s, log_f, 1)[0]
            return alpha
        return 0.5

    @staticmethod
    def _lempel_ziv_complexity(binary_string):
        s = binary_string
        n = len(s)
        if n == 0:
            return 0
        complexity = 1
        l = 1
        k = 1
        k_max = 1
        while l + k <= n:
            if s[l+k-1] == s[k-1] if k <= l else False:
                k += 1
            else:
                k_max = max(k_max, k)
                l += 1
                if l + 1 > n:
                    break
                k = 1
                complexity += 1
        return complexity

    # ================================================================
    # 辅助函数
    # ================================================================
    def _pct_change(self, arr, idx, lookback):
        if idx < lookback:
            return 0
        prev = self._safe(arr, idx - lookback, 0)
        curr = self._safe(arr, idx, 0)
        return (curr / prev - 1) * 100 if prev > 0 and not np.isnan(prev) else 0

    def _compute_variance_ratio(self, idx, period=20):
        if idx < period * 2:
            return 1
        rets = self.log_ret[idx-period*2+1:idx+1]
        if len(rets) < period:
            return 1
        v1 = np.var(rets)
        rets_2d = rets[::2]
        v2 = np.var(rets_2d) if len(rets_2d) >= 2 else v1
        return v2 / (2 * v1) if v1 > 0 else 1

    @staticmethod
    def _permutation_entropy(series, order=3):
        if len(series) < order + 1:
            return 1.0
        from math import factorial
        patterns = {}
        for i in range(len(series) - order + 1):
            pat = tuple(np.argsort(series[i:i+order]))
            patterns[pat] = patterns.get(pat, 0) + 1
        total = sum(patterns.values())
        max_entropy = np.log2(factorial(order))
        if max_entropy == 0:
            return 1.0
        entropy = -sum((v/total) * np.log2(v/total) for v in patterns.values())
        return entropy / max_entropy

    @staticmethod
    def _sample_entropy(series, m=2, r_mult=0.2):
        if len(series) < m + 2:
            return 0
        r = r_mult * np.std(series)
        if r == 0:
            return 0
        n = len(series)

        def count_matches(length):
            count = 0
            for i in range(n - length):
                for j in range(i + 1, n - length):
                    if all(abs(series[i+k] - series[j+k]) < r for k in range(length)):
                        count += 1
            return count

        a = count_matches(m + 1)
        b = count_matches(m)
        if b == 0:
            return 0
        return -np.log(a / b) if a > 0 else 0

    # ================================================================
    # 二十、长周期因子 (50+个) — 120~500天窗口
    # ================================================================
    def _long_term_factors(self, idx: int) -> Dict[str, float]:
        """长周期因子: 捕获长期趋势结构、趋势质量、趋势成熟度"""
        f = {}
        c, h, l, v = self.close, self.high, self.low, self.volume
        if idx < 250:
            return f

        # ---- A. 长期收益率/动量 (8个) ----
        f['lt_ret_120d'] = (c[idx] / c[idx - 120] - 1) * 100
        f['lt_ret_250d'] = (c[idx] / c[idx - 250] - 1) * 100
        if idx >= 500:
            f['lt_ret_500d'] = (c[idx] / c[idx - 500] - 1) * 100
        else:
            f['lt_ret_500d'] = np.nan
        # 长期ROC
        f['lt_roc_120'] = f['lt_ret_120d']
        f['lt_roc_250'] = f['lt_ret_250d']
        # 6个月动量 (≈120天)
        f['lt_mom_6m'] = f['lt_ret_120d']
        # 12个月动量 (≈250天)
        f['lt_mom_12m'] = f['lt_ret_250d']
        # 短长期动量比 (20天动量/250天动量)
        ret20 = (c[idx] / c[idx - 20] - 1) * 100
        f['lt_mom_ratio_20_250'] = ret20 / (f['lt_ret_250d'] + 0.01)

        # ---- B. 长期均线斜率/位置 (12个) ----
        ma120 = self.ma120[idx] if not np.isnan(self.ma120[idx]) else c[idx]
        ma250 = self.ma250[idx] if not np.isnan(self.ma250[idx]) else c[idx]
        ma120_prev20 = self.ma120[idx - 20] if idx >= 20 and not np.isnan(self.ma120[idx - 20]) else ma120
        ma250_prev20 = self.ma250[idx - 20] if idx >= 20 and not np.isnan(self.ma250[idx - 20]) else ma250
        ma120_prev60 = self.ma120[idx - 60] if idx >= 60 and not np.isnan(self.ma120[idx - 60]) else ma120
        ma250_prev60 = self.ma250[idx - 60] if idx >= 60 and not np.isnan(self.ma250[idx - 60]) else ma250

        f['lt_ma120_slope_20d'] = (ma120 / ma120_prev20 - 1) * 100
        f['lt_ma120_slope_60d'] = (ma120 / ma120_prev60 - 1) * 100
        f['lt_ma250_slope_20d'] = (ma250 / ma250_prev20 - 1) * 100
        f['lt_ma250_slope_60d'] = (ma250 / ma250_prev60 - 1) * 100
        f['lt_dist_ma120'] = (c[idx] / ma120 - 1) * 100
        f['lt_dist_ma250'] = (c[idx] / ma250 - 1) * 100
        f['lt_above_ma120'] = 1.0 if c[idx] > ma120 else 0.0
        f['lt_above_ma250'] = 1.0 if c[idx] > ma250 else 0.0
        # MA120和MA250的相对位置
        f['lt_ma120_above_ma250'] = 1.0 if ma120 > ma250 else 0.0
        f['lt_ma_spread_120_250'] = (ma120 / ma250 - 1) * 100
        # 长期均线多头排列度
        ma60 = self.ma60[idx] if not np.isnan(self.ma60[idx]) else c[idx]
        ma20 = self.ma20[idx] if not np.isnan(self.ma20[idx]) else c[idx]
        f['lt_ma_alignment_full'] = sum([
            1 if ma20 > ma60 else 0,
            1 if ma60 > ma120 else 0,
            1 if ma120 > ma250 else 0,
        ]) / 3.0
        f['lt_price_ma_stack'] = sum([
            1 if c[idx] > ma20 else 0,
            1 if c[idx] > ma60 else 0,
            1 if c[idx] > ma120 else 0,
            1 if c[idx] > ma250 else 0,
        ]) / 4.0

        # ---- C. 长期线性回归 (8个) ----
        for window in [120, 250]:
            x = np.arange(window)
            y = c[idx - window + 1:idx + 1]
            if len(y) == window:
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                f[f'lt_lr_r2_{window}d'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                f[f'lt_lr_slope_{window}d'] = slope / np.mean(y) * 100
                # 当前价格偏离回归线
                expected = slope * (window - 1) + intercept
                f[f'lt_lr_deviation_{window}d'] = (c[idx] / expected - 1) * 100
                # 回归残差的标准差 (趋势噪音)
                residuals = y - y_pred
                f[f'lt_lr_noise_{window}d'] = np.std(residuals) / np.mean(y) * 100

        # ---- D. 长期效率比 (4个) ----
        for window in [120, 250]:
            net = abs(c[idx] - c[idx - window])
            path = sum(abs(c[idx - window + j + 1] - c[idx - window + j]) for j in range(window))
            f[f'lt_er_{window}d'] = net / (path + 1e-10)

        # 效率比变化(近60天的ER vs 前60天的ER)
        if idx >= 180:
            net_recent = abs(c[idx] - c[idx - 60])
            path_recent = sum(abs(c[idx - 60 + j + 1] - c[idx - 60 + j]) for j in range(60))
            er_recent = net_recent / (path_recent + 1e-10)
            net_older = abs(c[idx - 60] - c[idx - 120])
            path_older = sum(abs(c[idx - 120 + j + 1] - c[idx - 120 + j]) for j in range(60))
            er_older = net_older / (path_older + 1e-10)
            f['lt_er_trend'] = er_recent - er_older
        else:
            f['lt_er_trend'] = 0

        f['lt_er_change_120d'] = f.get(f'lt_er_{120}d', 0)

        # ---- E. 长期Hurst指数 (3个) ----
        for window in [120, 250]:
            ret = np.diff(np.log(c[idx - window + 1:idx + 1]))
            f[f'lt_hurst_{window}d'] = self._rs_hurst(ret)

        # Hurst变化
        if idx >= 310:
            ret_recent = np.diff(np.log(c[idx - 119:idx + 1]))
            ret_older = np.diff(np.log(c[idx - 249:idx - 119]))
            f['lt_hurst_trend'] = self._rs_hurst(ret_recent) - self._rs_hurst(ret_older)
        else:
            f['lt_hurst_trend'] = 0

        # ---- F. 长期方差比 (4个) ----
        ret250 = np.diff(np.log(c[idx - 249:idx + 1]))
        var1 = np.var(ret250)
        if var1 > 0:
            for q in [20, 60]:
                ret_q = np.array([np.sum(ret250[i:i + q]) for i in range(len(ret250) - q + 1)])
                f[f'lt_vr_{q}_250d'] = np.var(ret_q) / (q * var1)
        else:
            f['lt_vr_20_250d'] = 1.0
            f['lt_vr_60_250d'] = 1.0

        # 120天方差比
        ret120 = np.diff(np.log(c[idx - 119:idx + 1]))
        var1_120 = np.var(ret120)
        if var1_120 > 0:
            for q in [10, 20]:
                ret_q = np.array([np.sum(ret120[i:i + q]) for i in range(len(ret120) - q + 1)])
                f[f'lt_vr_{q}_120d'] = np.var(ret_q) / (q * var1_120)
        else:
            f['lt_vr_10_120d'] = 1.0
            f['lt_vr_20_120d'] = 1.0

        # ---- G. 250日高低点相关 (6个) ----
        high250 = np.max(h[idx - 249:idx + 1])
        low250 = np.min(l[idx - 249:idx + 1])
        f['lt_pct_from_250d_high'] = (c[idx] / high250 - 1) * 100
        f['lt_pct_from_250d_low'] = (c[idx] / low250 - 1) * 100
        f['lt_pp250'] = (c[idx] - low250) / (high250 - low250) if high250 > low250 else 0.5

        # 250日高点在几天前？
        high_idx = idx - 249 + np.argmax(h[idx - 249:idx + 1])
        f['lt_days_since_250d_high'] = idx - high_idx
        low_idx = idx - 249 + np.argmin(l[idx - 249:idx + 1])
        f['lt_days_since_250d_low'] = idx - low_idx

        # 高点在低点之后(上升趋势) vs 高点在低点之前(下降趋势)
        f['lt_high_after_low'] = 1.0 if high_idx > low_idx else 0.0

        # ---- H. Coppock Curve (2个) ----
        if idx >= 280:
            roc14m = (c[idx] / c[idx - 280] - 1) * 100
            roc11m = (c[idx] / c[idx - 220] - 1) * 100
            f['lt_coppock'] = (roc14m + roc11m) / 2
            # Coppock方向
            if idx >= 300:
                roc14m_prev = (c[idx - 20] / c[idx - 300] - 1) * 100
                roc11m_prev = (c[idx - 20] / c[idx - 240] - 1) * 100
                coppock_prev = (roc14m_prev + roc11m_prev) / 2
                f['lt_coppock_slope'] = f['lt_coppock'] - coppock_prev
            else:
                f['lt_coppock_slope'] = 0
        else:
            f['lt_coppock'] = 0
            f['lt_coppock_slope'] = 0

        # ---- I. 趋势成熟度/阶段 (6个) ----
        # 连续在MA120上方的天数
        above120_days = 0
        for j in range(idx, max(idx - 250, 0), -1):
            if not np.isnan(self.ma120[j]) and c[j] > self.ma120[j]:
                above120_days += 1
            else:
                break
        f['lt_days_above_ma120'] = above120_days

        # 连续在MA250上方的天数
        above250_days = 0
        for j in range(idx, max(idx - 500, 0), -1):
            if not np.isnan(self.ma250[j]) and c[j] > self.ma250[j]:
                above250_days += 1
            else:
                break
        f['lt_days_above_ma250'] = above250_days

        # MA120穿越频率(长期震荡 vs 单边趋势)
        cross120 = 0
        for j in range(max(1, idx - 249), idx + 1):
            if (not np.isnan(self.ma120[j]) and not np.isnan(self.ma120[j - 1])
                    and ((c[j] > self.ma120[j]) != (c[j - 1] > self.ma120[j - 1]))):
                cross120 += 1
        f['lt_ma120_cross_freq'] = cross120 / 250

        # MA250穿越频率
        cross250 = 0
        for j in range(max(1, idx - 249), idx + 1):
            if (not np.isnan(self.ma250[j]) and not np.isnan(self.ma250[j - 1])
                    and ((c[j] > self.ma250[j]) != (c[j - 1] > self.ma250[j - 1]))):
                cross250 += 1
        f['lt_ma250_cross_freq'] = cross250 / 250

        # 长期方向运动比
        dm_plus = 0
        dm_minus = 0
        for j in range(idx - 119, idx + 1):
            if j < 1:
                continue
            up = h[j] - h[j - 1]
            dn = l[j - 1] - l[j]
            if up > dn and up > 0:
                dm_plus += up
            elif dn > up and dn > 0:
                dm_minus += dn
        f['lt_dm_ratio_120d'] = dm_plus / (dm_plus + dm_minus + 1e-10)

        # Weinstein阶段近似 (MA120斜率+价格位置)
        if f['lt_ma120_slope_20d'] > 0.5 and c[idx] > ma120:
            f['lt_weinstein_stage'] = 2.0  # Stage 2: 上升
        elif f['lt_ma120_slope_20d'] < -0.5 and c[idx] < ma120:
            f['lt_weinstein_stage'] = 4.0  # Stage 4: 下降
        elif f['lt_ma120_slope_20d'] > -0.3 and abs(c[idx] / ma120 - 1) < 0.05:
            f['lt_weinstein_stage'] = 1.0  # Stage 1: 筑底
        else:
            f['lt_weinstein_stage'] = 3.0  # Stage 3: 见顶

        # ---- J. 长期成交量趋势 (4个) ----
        vol_60d = np.mean(v[idx - 59:idx + 1])
        vol_120d = np.mean(v[idx - 119:idx + 1])
        vol_250d = np.mean(v[idx - 249:idx + 1])
        f['lt_vol_trend_60_250'] = vol_60d / (vol_250d + 1e-10)
        f['lt_vol_trend_120_250'] = vol_120d / (vol_250d + 1e-10)
        # 长期量价配合度
        price_slope = f.get('lt_lr_slope_120d', 0)
        vol_slope_arr = v[idx - 119:idx + 1]
        x120 = np.arange(120)
        vol_lr_slope = np.polyfit(x120, vol_slope_arr, 1)[0] / (np.mean(vol_slope_arr) + 1e-10) * 100
        f['lt_vol_price_agreement_120d'] = 1.0 if (price_slope > 0 and vol_lr_slope > 0) or (price_slope < 0 and vol_lr_slope < 0) else 0.0
        f['lt_vol_lr_slope_120d'] = vol_lr_slope

        # ---- K. 长短期交互因子 (5个) ----
        # RSI(短期) × 长期趋势质量
        rsi = self.rsi29[idx] if not np.isnan(self.rsi29[idx]) else 50
        f['lt_rsi_x_lr_r2_250'] = rsi * f.get('lt_lr_r2_250d', 0)
        f['lt_rsi_x_ma_alignment'] = rsi * f['lt_ma_alignment_full']
        f['lt_rsi_x_er_250'] = rsi * f.get('lt_er_250d', 0)
        # 短期动量 × 长期趋势方向一致性
        f['lt_mom_alignment_3'] = 1.0 if (ret20 > 0 and f['lt_ret_120d'] > 0 and f['lt_ret_250d'] > 0) else 0.0
        # 长期趋势支撑分数 (综合)
        f['lt_trend_support_score'] = (
            f['lt_above_ma120'] * 0.2 +
            f['lt_above_ma250'] * 0.2 +
            f['lt_ma120_above_ma250'] * 0.2 +
            min(1, max(0, f.get('lt_lr_r2_250d', 0))) * 0.2 +
            min(1, max(0, f.get('lt_er_250d', 0) * 5)) * 0.2
        )

        # ---- L. 趋势一致性 (Ehsani-Linnainmaa inspired, 6个) ----
        # 把250天分成4/5个子区间，看几个子区间收益为正
        for n_sub in [4, 5]:
            sub_len = 250 // n_sub
            pos_count = 0
            for q in range(n_sub):
                start_idx = idx - 250 + q * sub_len
                end_idx = start_idx + sub_len
                if start_idx >= 0 and end_idx <= idx:
                    if c[end_idx] > c[start_idx]:
                        pos_count += 1
            f[f'lt_trend_consistency_{n_sub}q'] = pos_count / n_sub

        # 120天分3个子区间
        sub40 = 120 // 3
        pos3 = sum(1 for q in range(3) if c[idx-120+q*sub40+sub40] > c[idx-120+q*sub40])
        f['lt_trend_consistency_3q_120d'] = pos3 / 3

        # 连续正季度数（从最近往回数）
        consec_pos = 0
        for q in range(4, 0, -1):
            s = idx - q * 63
            e = s + 63
            if s >= 0 and e <= idx and c[e] > c[s]:
                consec_pos += 1
            else:
                break
        f['lt_consec_pos_quarters'] = consec_pos

        # 多周期动量一致性 (1m/3m/6m/12m)
        horizons_pos = 0
        for h_days in [21, 63, 126, 250]:
            if idx >= h_days and c[idx] > c[idx - h_days]:
                horizons_pos += 1
        f['lt_multi_horizon_score'] = horizons_pos / 4
        f['lt_multi_horizon_all_pos'] = 1.0 if horizons_pos == 4 else 0.0

        # ---- M. TQI 趋势/噪音比 (4个) ----
        for window in [120, 250]:
            y = c[idx - window + 1:idx + 1]
            x_arr = np.arange(window, dtype=float)
            slope_t, intercept_t = np.polyfit(x_arr, y, 1)
            trend_line = slope_t * x_arr + intercept_t
            noise = np.mean(np.abs(y - trend_line))
            if noise > 0:
                f[f'lt_tqi_{window}d'] = (slope_t * window) / noise
            else:
                f[f'lt_tqi_{window}d'] = 0

        # ---- N. 52周高点接近度 (George-Hwang因子, 3个) ----
        f['lt_nearness_252d_high'] = c[idx] / high250  # 0-1, 接近1=近高点
        # 52周高点出现在多少天前的比率
        f['lt_high_recency_ratio'] = 1 - f['lt_days_since_250d_high'] / 250
        # 52周低点接近度
        f['lt_nearness_252d_low'] = low250 / c[idx]  # 0-1, 接近1=近低点

        # ---- O. 绝对动量/双动量 (Antonacci, 4个) ----
        # 250天绝对动量(正=上涨趋势)
        f['lt_abs_momentum_250'] = 1.0 if f['lt_ret_250d'] > 0 else 0.0
        f['lt_abs_momentum_120'] = 1.0 if f['lt_ret_120d'] > 0 else 0.0
        # 风险调整动量
        ret_std_250 = np.std(np.diff(np.log(c[idx-249:idx+1]))) * np.sqrt(250) * 100
        f['lt_risk_adj_momentum'] = f['lt_ret_250d'] / (ret_std_250 + 1e-10)
        # 动量加速度（6个月动量 vs 12个月动量的一半）
        f['lt_momentum_accel'] = f['lt_ret_120d'] - f['lt_ret_250d'] / 2

        # ---- P. Elder三重滤网长期趋势 (3个) ----
        # 65日EMA斜率（≈周线EMA13）
        ema65 = self._ema(c, 65) if hasattr(self, '_ema') else np.full(len(c), np.nan)
        if idx >= 70 and not np.isnan(ema65[idx]) and not np.isnan(ema65[idx-5]):
            f['lt_elder_screen1_slope'] = (ema65[idx] / ema65[idx-5] - 1) * 100
            f['lt_elder_screen1_up'] = 1.0 if f['lt_elder_screen1_slope'] > 0.1 else 0.0
        else:
            f['lt_elder_screen1_slope'] = 0
            f['lt_elder_screen1_up'] = 0.5
        # 130日EMA斜率（≈周线EMA26）
        ema130 = self._ema(c, 130) if hasattr(self, '_ema') else np.full(len(c), np.nan)
        if idx >= 135 and not np.isnan(ema130[idx]) and not np.isnan(ema130[idx-5]):
            f['lt_elder_weekly_macd'] = (ema65[idx] - ema130[idx]) / c[idx] * 100
        else:
            f['lt_elder_weekly_macd'] = 0

        return f

    def _rs_hurst(self, ts):
        """R/S法估计Hurst指数"""
        n = len(ts)
        if n < 20:
            return 0.5
        max_k = n // 4
        log_rs = []
        log_n = []
        for k in range(10, max_k + 1, max(1, (max_k - 10) // 8)):
            rs_vals = []
            for start in range(0, n - k + 1, k):
                seg = ts[start:start + k]
                mean_seg = np.mean(seg)
                cumdev = np.cumsum(seg - mean_seg)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(seg, ddof=1) if np.std(seg) > 0 else 1e-10
                rs_vals.append(R / S)
            if rs_vals:
                log_rs.append(np.log(np.mean(rs_vals)))
                log_n.append(np.log(k))
        if len(log_rs) < 2:
            return 0.5
        return np.polyfit(log_n, log_rs, 1)[0]

    # ================================================================
    # 因子名称列表
    # ================================================================
    def get_factor_names(self) -> list:
        """返回所有因子名称（需要先计算一次）"""
        if self.n > 120:
            factors = self.compute_all(120)
            return list(factors.keys())
        return []


# ================================================================
# 便捷接口
# ================================================================
def compute_factors_at_bar(close, high, low, volume, bar_idx, dates=None):
    """一次性计算指定bar的全部因子"""
    engine = FactorEngine(close, high, low, volume, dates)
    return engine.compute_all(bar_idx)


def compute_factors_batch(close, high, low, volume, bar_indices, dates=None):
    """批量计算多个bar的因子（共享预计算）"""
    engine = FactorEngine(close, high, low, volume, dates)
    results = {}
    for idx in bar_indices:
        results[idx] = engine.compute_all(idx)
    return results


if __name__ == '__main__':
    # 测试：加载000001，计算第500根bar的所有因子
    import glob
    import pandas as pd

    files = glob.glob('stock_trading_advisor/data/cache/000001*')
    df = pd.read_csv(files[0])

    engine = FactorEngine(
        df['close'].values, df['high'].values,
        df['low'].values, df['volume'].values
    )

    factors = engine.compute_all(500)
    print(f'总因子数: {len(factors)}')
    print()
    print('因子列表:')
    for i, (name, val) in enumerate(sorted(factors.items())):
        print(f'  {i+1:3d}. {name:35s} = {val:10.4f}')
