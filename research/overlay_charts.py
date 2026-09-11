"""Chart the anticipation overlay leg by leg: where it buys, what the setup looked like, what followed.

Panels: price with the formal strategy's position and the overlay's legs marked and labelled with each
leg's realised return; the detector score against its frozen threshold; the dip filter in ATR units.
Every state is causal - the detector's threshold is the FROZEN training quantile, never recomputed here.
"""
import argparse, os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from hooked import load_registry
from harness import resolve_pool, OUT

CAND = "anti_ma120m2_q90_h20"


def _detector(df):
    """Frozen launch detector score, exactly as the candidate uses it."""
    import json
    from launch_features import build_features
    m = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_model.json")))["full"]
    f = build_features(df)
    cols = m["cols"]
    mu = np.array([m["mu"][c] for c in cols]); sd = np.array([m["sd"][c] for c in cols])
    co = np.array([m["coef"][c] for c in cols])
    z = ((f[cols].to_numpy() - mu) / sd).clip(-5, 5)
    p = 1.0 / (1.0 + np.exp(-(z @ co + m["intercept"])))
    return pd.Series(p, index=f.index), float(m["thr90"])


def one(args):
    path, outdir = args
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    code = os.path.basename(path).split("_")[0]
    mk = "HK" if len(code) == 5 else "CN-A"
    raw = pd.read_csv(path)
    reg = load_registry()
    db, _ = reg["baseline"](market=mk, stock_code=code).analyze(raw.copy())
    dc, _ = reg[CAND](market=mk, stock_code=code).analyze(raw.copy())
    c = db["close"]; dates = pd.to_datetime(db["date"])
    pb = db["buy_signal"].astype(bool); pc = dc["buy_signal"].astype(bool)
    score, thr = _detector(raw)
    score = score.reindex(range(len(c)))
    ma120 = c.rolling(120).mean()
    atr = db["atr"]
    dip = ((c - ma120) / atr.replace(0, np.nan))

    # overlay legs = bars the candidate holds but the baseline does not
    extra = (pc & ~pb).to_numpy()
    d = np.diff(extra.astype(int), prepend=0, append=0)
    legs = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

    fig, (ax, a2, a3) = plt.subplots(3, 1, figsize=(19, 10), sharex=True,
                                     gridspec_kw={"height_ratios": [5, 1.2, 1.2]})
    ax.plot(dates, c, lw=0.9, color="#222", zorder=3)
    ax.plot(dates, ma120, lw=0.9, color="#1f77b4", label="MA120")
    lo, hi = c.min(), c.max()
    ax.fill_between(dates, lo, hi, where=pb.to_numpy(), color="#2ca02c", alpha=0.10,
                    label="formal strategy holding")
    won = lost = hand = 0
    for a, b in legs:
        b = min(b, len(c) - 1)
        r = (c.iloc[b] / c.iloc[a] - 1) * 100
        handover = bool(pb.iloc[min(b + 1, len(c) - 1)])
        won += r > 0; lost += r <= 0; hand += handover
        ax.axvspan(dates.iloc[a], dates.iloc[b], color="#d62728" if r <= 0 else "#ff7f0e",
                   alpha=0.30 if not handover else 0.55, zorder=2)
        ax.annotate(f"{r:+.0f}%" + ("→" if handover else ""), (dates.iloc[a], c.iloc[a]),
                    textcoords="offset points", xytext=(0, -14), fontsize=7, ha="center",
                    color="#2ca02c" if r > 0 else "#d62728", fontweight="bold")
    bh = (c.iloc[-1] / c.iloc[0] - 1) * 100
    from harness import OUT as _o
    sb = reg["baseline"](market=mk, stock_code=code).backtest(db, 10000.0)["total_return"]
    sc = reg[CAND](market=mk, stock_code=code).backtest(dc, 10000.0)["total_return"]
    ax.set_title(f"{code} {mk}   buy&hold {bh:+.0f}%   formal {sb:+.0f}%   +overlay {sc:+.0f}%   |   "
                 f"{len(legs)} overlay legs ({won} up / {lost} down), {hand} hand a live long to the trend system (→)   |   "
                 f"orange/red = overlay leg, darker = handover")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.2)

    a2.plot(dates, score, lw=0.8, color="#9467bd")
    a2.axhline(thr, color="#d62728", lw=0.9, ls="--")
    a2.fill_between(dates, thr, 1.0, where=(score >= thr).to_numpy(), color="#9467bd", alpha=0.25)
    a2.set_ylabel("detector\n(frozen q90)", fontsize=7); a2.grid(alpha=0.2)

    a3.plot(dates, dip, lw=0.8, color="#8c564b")
    a3.axhline(-2.0, color="#d62728", lw=0.9, ls="--")
    a3.fill_between(dates, dip.min(), -2.0, where=(dip <= -2.0).to_numpy(), color="#8c564b", alpha=0.25)
    a3.set_ylabel("(close-MA120)\n/ ATR", fontsize=7); a3.grid(alpha=0.2)

    fig.tight_layout(); os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{code}.png"), dpi=88); plt.close(fig)
    rets = [(c.iloc[min(b, len(c) - 1)] / c.iloc[a] - 1) * 100 for a, b in legs]
    hands = [bool(pb.iloc[min(b + 1, len(c) - 1)]) for a, b in legs]
    return dict(code=code, mkt=mk, bh=bh, formal=sb, overlay=sc, legs=len(legs),
                leg_mean=np.mean(rets) if rets else np.nan,
                leg_med=np.median(rets) if rets else np.nan,
                win=np.mean([r > 0 for r in rets]) * 100 if rets else np.nan,
                handover=np.mean(hands) * 100 if hands else np.nan,
                hand_mean=np.mean([r for r, h in zip(rets, hands) if h]) if any(hands) else np.nan,
                nohand_mean=np.mean([r for r, h in zip(rets, hands) if not h]) if any(not h for h in hands) else np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stocks", nargs="+", required=True)
    ap.add_argument("--pool", default="pool_b_dev"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(resolve_pool(a.pool), "*_hfq.csv")))
    sel = [f for f in files if os.path.basename(f).split("_")[0] in set(a.stocks)]
    outdir = os.path.join(OUT, "overlay_charts")
    with Pool(a.workers) as p:
        rows = p.map(one, [(f, outdir) for f in sel], chunksize=1)
    T = pd.DataFrame(rows)
    print(f"charts -> {outdir}\n")
    print(f"{'code':<8}{'mkt':<6}{'B&H':>9}{'formal':>9}{'+overlay':>10}{'legs':>6}{'leg mean':>10}{'win%':>7}"
          f"{'handover%':>11}{'hand leg':>10}{'no-hand':>9}")
    for _, r in T.sort_values("bh", ascending=False).iterrows():
        print(f"{r.code:<8}{r.mkt:<6}{r.bh:>+8.0f}%{r.formal:>+8.0f}%{r.overlay:>+9.0f}%{r.legs:>6.0f}"
              f"{r.leg_mean:>+9.2f}%{r.win:>6.0f}%{r.handover:>10.0f}%{r.hand_mean:>+9.1f}%{r.nohand_mean:>+8.1f}%")
    print(f"\n  pooled: {T.legs.sum():.0f} legs, mean {T.leg_mean.mean():+.2f}%, handover {T.handover.mean():.0f}%,"
          f"  handover legs {T.hand_mean.mean():+.1f}% vs non-handover {T.nohand_mean.mean():+.1f}%")


if __name__ == "__main__":
    main()
