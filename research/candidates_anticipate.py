"""ANTICIPATE instead of CONFIRM: the frozen launch detector + a NON-momentum direction filter.

Measured motivation (this session, both clean pools, 8 cells = pool x market x period):
  - the frozen detector alone is DIRECTIONLESS: fwd60 excess over the same stock's own uncondtional
    mean is -1.6..+8.7pp, positive in only 3/8 cells.
  - detector high AND the stock oversold by its OWN history (RSI-14 in the bottom 30% of its trailing
    250 bars, or 2+ ATR below its MA120, or 30%+ below its 250-bar high) -> fwd60 excess +2.3..+18.5pp,
    8/8 cells, win rate 66% vs a 48.6% base rate.
  - detector high AND extended (the momentum version) -> fwd60 excess NEGATIVE in 7-8/8 cells,
    win rate 33% vs 49.5%. The falsification test therefore agrees with the mechanism: the reversal
    prior supplies the DIRECTION, the detector supplies the MAGNITUDE.

Trading form: an OVERLAY on the formal strategy. The formal strategy is untouched; when it is flat and
the anticipation signal fires, the overlay buys at the close and holds a fixed NHOLD bars under a hard
stop, then hands the position back to the formal strategy (kept if the formal strategy is long by then,
sold otherwise). Everything the signal reads is causal: that stock's own OHLCV, the frozen coefficients
of research/launch_model.json and the frozen training quantiles.

Controls in the same file, deliberately:
  anti_rand_*  exposure-matched RANDOM timing: same mechanics, same per-stock fire rate, no information.
  anti_rot_*   calendar rotation: the real signal, circularly shifted, so the calendar and the signal's
               own run structure survive and only the alignment with price is destroyed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from hooked import HookedStrategy  # noqa: E402
import launch_features as LF  # noqa: E402

MODEL = json.load(open(os.path.join(HERE, "launch_model.json")))


def _score_from_features(f, model):
    """The frozen logistic, evaluated on an already-built feature frame (LF.score without rebuild)."""
    cols = model["cols"]
    mu = pd.Series(model["mu"])[cols]
    sd = pd.Series(model["sd"])[cols]
    co = pd.Series(model["coef"])[cols].to_numpy(float)
    z = ((f[cols] - mu) / sd).clip(-5, 5)
    ok = z.notna().all(axis=1)
    lin = np.full(len(f), np.nan)
    lin[ok.to_numpy()] = z.loc[ok].to_numpy(float) @ co + model["intercept"]
    return pd.Series(1.0 / (1.0 + np.exp(-lin)), index=f.index)


def _rank250(s):
    return s.rolling(250, min_periods=120).rank(pct=True)


def proxy_series(f, close, kind):
    """Causal, NON-momentum direction proxies. 'beaten' = the reversal prior says up."""
    atr_abs = f["atr_pct_price"] * close / 100.0
    ma120 = close.rolling(120).mean()
    if kind == "rsi30":
        return _rank250(f["rsi14"]) <= 0.30
    if kind == "rsi20":
        return _rank250(f["rsi14"]) <= 0.20
    if kind == "rsi40":
        return _rank250(f["rsi14"]) <= 0.40
    if kind == "ma120m2":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) <= -2.0
    if kind == "ma120m3":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) <= -3.0
    if kind == "ma120m4":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) <= -4.0
    if kind == "ma120m5":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) <= -5.0
    if kind == "ma120m7":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) <= -7.0
    if kind == "ma120m1":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) <= -1.0
    if kind == "hi250_30":
        return (-f["pos_250"]) >= 30.0
    if kind == "hi250_20":
        return (-f["pos_250"]) >= 20.0
    # ---- the momentum inverses: the falsification arm ----
    if kind == "x_rsi70":
        return _rank250(f["rsi14"]) >= 0.70
    if kind == "x_ma120p2":
        return ((close - ma120) / atr_abs.replace(0, np.nan)) >= 2.0
    if kind == "x_hi250_3":
        return (-f["pos_250"]) <= 3.0
    if kind == "none":
        return pd.Series(True, index=close.index)
    raise ValueError(kind)


class AnticipateOverlay(HookedStrategy):
    """Formal strategy + an anticipation overlay that only ever acts while the formal strategy is flat.

    ANTICIPATION_ENABLED is False because this class implements its OWN overlay inside
    _build_position_series; leaving the inherited production overlay on would apply it twice.
    """
    ANTICIPATION_ENABLED = False
    Q = "90"            # frozen TRAINING quantile of the detector score
    PROXY = "rsi30"
    NHOLD = 20          # bars held by the overlay before control returns to the formal strategy
    STOP = 12.0         # hard stop on the overlay leg, percent
    COOLDOWN = 0        # bars to wait after an overlay exit before the overlay may fire again
    TAG = "full"
    MODE = "real"       # real | random | rotate
    SEED = 0

    def _signal(self, data):
        close = data["close"].astype(float)
        f = LF.build_features(data)
        score = _score_from_features(f, MODEL[self.TAG])
        if self.Q == "off":   # control: the direction proxy ALONE, detector removed
            hot = pd.Series(True, index=close.index)
        else:
            hot = (score >= MODEL[self.TAG]["q"][self.Q]).fillna(False)
        beat = proxy_series(f, close, self.PROXY).fillna(False)
        sig = (hot & beat).to_numpy(bool)
        warm = np.isfinite(score.to_numpy())
        sig &= warm
        if self.MODE in ("real", "matched"):
            return sig
        rng = np.random.default_rng(self.SEED * 1000003 + (abs(hash(str(self.stock_code))) % 99991))
        if self.MODE == "random":
            # exposure-matched random timing: same mechanics, same per-stock fire rate, no information
            rate = sig.mean() if sig.mean() > 0 else 0.0
            out = rng.random(len(sig)) < rate
            out[~warm] = False
            return out
        if self.MODE == "rotate":
            # calendar rotation: keep the calendar and the signal's own run structure, break alignment
            k = int(rng.integers(60, max(61, len(sig) - 60)))
            return np.roll(sig, k)
        raise ValueError(self.MODE)

    def _overlay(self, pos, close, sig):
        """Run the overlay on top of a baseline position series. Returns (merged, entry_bars)."""
        n = len(pos)
        out = pos.copy()
        entries = []
        ov = False; entry_px = np.nan; held = 0; cool_until = -1
        for i in range(n):
            closed_here = False
            if ov:
                px = close[i]
                if np.isfinite(entry_px) and entry_px > 0 and px <= entry_px * (1.0 - self.STOP / 100.0):
                    ov = False; closed_here = True; cool_until = i + self.COOLDOWN
                else:
                    held += 1
                    if held >= self.NHOLD:
                        ov = False; closed_here = True
                        if pos[i] == 0:
                            cool_until = i + self.COOLDOWN
            if (not ov) and (not closed_here) and pos[i] == 0 and sig[i] and i > cool_until:
                ov = True; entry_px = close[i]; held = 0
                entries.append(i)
            out[i] = 1 if (ov or pos[i] == 1) else 0
        return out, entries

    def _build_position_series(self, entry_condition, exit_condition, data):
        base = HookedStrategy._build_position_series(self, entry_condition, exit_condition, data)
        pos, ent, ext, stp, prf, er, xr = base
        close = data["close"].to_numpy(float)
        sig = self._signal(data)
        if self.MODE == "matched":
            # SAME-EXPOSURE RANDOM-TIMING CONTROL, two-sided. Take the REAL rule's entry count and
            # added-day count, place the same number of blocks at random among the bars where the
            # formal strategy is flat, then TRIM or EXTEND the last block until the added-day count
            # matches exactly. Neither direction can silently return NaN.
            real, entries = self._overlay(pos, close, sig)
            target = int(real.sum() - pos.sum())
            rng = np.random.default_rng(self.SEED * 7919 + (abs(hash(str(self.stock_code))) % 99991))
            flat = np.flatnonzero((pos == 0) & np.isfinite(close))
            flat = flat[flat > 250]
            if len(entries) == 0 or len(flat) == 0:
                out = pos.copy()
            else:
                pick = rng.choice(flat, size=min(len(entries), len(flat)), replace=False)
                rsig = np.zeros(len(pos), bool); rsig[pick] = True
                out, _ = self._overlay(pos, close, rsig)
                added = int(out.sum() - pos.sum())
                idx = np.flatnonzero((out == 1) & (pos == 0))
                while added > target and len(idx):           # trim
                    out[idx[-1]] = 0; idx = idx[:-1]; added -= 1
                j = int(idx[-1]) + 1 if len(idx) else 0
                while added < target and j < len(pos):       # extend
                    if pos[j] == 0 and out[j] == 0:
                        out[j] = 1; added += 1
                    j += 1
            return out, ent, ext, stp, prf, er, xr
        out, entries = self._overlay(pos, close, sig)
        ent = ent.copy(); ext = ext.copy(); stp = stp.copy(); er = list(er); xr = list(xr)
        for i in entries:
            ent[i] = 1; er[i] = "anticipate"
        return out, ent, ext, stp, prf, er, xr


def make(proxy="rsi30", q="90", nhold=20, stop=12.0, cooldown=0, mode="real", seed=0, tag="full"):
    class A(AnticipateOverlay):
        PROXY = proxy; Q = q; NHOLD = nhold; STOP = stop; COOLDOWN = cooldown
        MODE = mode; SEED = seed; TAG = tag
    A.__name__ = "Anti_%s_q%s_h%d_s%g_%s%s" % (proxy, q, nhold, stop, mode,
                                               "" if mode == "real" else str(seed))
    return A


REGISTRY = {}
# --- the anticipation arm: detector + beaten-down, a hold-length plateau and a threshold plateau ---
for _p in ("rsi30", "ma120m2", "hi250_30"):
    for _h in (10, 20, 30, 40, 60):
        REGISTRY["anti_%s_q90_h%d" % (_p, _h)] = make(_p, "90", _h)
for _q in ("85", "90", "95"):
    REGISTRY["anti_rsi30_q%s_h20" % _q] = make("rsi30", _q, 20)
    REGISTRY["anti_ma120m2_q%s_h20" % _q] = make("ma120m2", _q, 20)
for _p in ("rsi20", "rsi40", "ma120m1", "hi250_20"):
    REGISTRY["anti_%s_q90_h20" % _p] = make(_p, "90", 20)
REGISTRY["anti_rsi30_q90_h20_s8"] = make("rsi30", "90", 20, stop=8.0)
REGISTRY["anti_rsi30_q90_h20_s20"] = make("rsi30", "90", 20, stop=20.0)
REGISTRY["anti_rsi30_q90_h20_cd20"] = make("rsi30", "90", 20, cooldown=20)
REGISTRY["anti_none_q90_h20"] = make("none", "90", 20)        # detector alone, no direction filter
# --- the falsification arm: detector + extended (momentum) ---
for _p in ("x_rsi70", "x_ma120p2", "x_hi250_3"):
    REGISTRY["anti_%s_q90_h20" % _p] = make(_p, "90", 20)
# --- controls ---
for _s in range(8):
    REGISTRY["anti_rsi30_q90_h20_rand%d" % _s] = make("rsi30", "90", 20, mode="random", seed=_s)
    REGISTRY["anti_rsi30_q90_h20_rot%d" % _s] = make("rsi30", "90", 20, mode="rotate", seed=_s)
    REGISTRY["anti_ma120m2_q90_h20_rand%d" % _s] = make("ma120m2", "90", 20, mode="random", seed=_s)
    REGISTRY["anti_ma120m2_q90_h20_rot%d" % _s] = make("ma120m2", "90", 20, mode="rotate", seed=_s)

# --- control: the direction proxy WITHOUT the detector (is the detector doing anything?) ---
for _p in ("rsi30", "ma120m2", "hi250_30"):
    REGISTRY["anti_%s_qoff_h20" % _p] = make(_p, "off", 20)

# --- rule 5: EXACT same-exposure random-timing control, trim AND extend ---
for _s in range(6):
    REGISTRY["anti_ma120m2_q90_h20_match%d" % _s] = make("ma120m2", "90", 20, mode="matched", seed=_s)
    REGISTRY["anti_rsi30_q90_h20_match%d" % _s] = make("rsi30", "90", 20, mode="matched", seed=_s)

# --- control: a STRICTER dip filter with no detector, to compare at the same exposure ---
for _p in ("ma120m3", "ma120m4", "ma120m5"):
    REGISTRY["anti_%s_qoff_h20" % _p] = make(_p, "off", 20)
REEXTRA = REGISTRY["anti_ma120m7_qoff_h20"] = make("ma120m7", "off", 20)
