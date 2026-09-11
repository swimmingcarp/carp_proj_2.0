"""Why do 'main advance' templates lose money here? Chart what they call an advance, and what follows.

Overlays, per stock:
  - Weinstein/Minervini stage-2 template (the canonical "main advance" definition)
  - the formal strategy's position
  - at every fresh template fire, the realised forward 60-bar return, annotated on the chart
All state is causal: computed from bars <= t.
"""
import argparse, os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from hooked import load_registry
from harness import resolve_pool, OUT

FWD = 60


def stage2(d):
    """Weinstein/Minervini trend template, causal. Returns the boolean state plus its components."""
    c = d["close"]
    ma50, ma150, ma200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    hi52, lo52 = c.rolling(250, min_periods=150).max(), c.rolling(250, min_periods=150).min()
    comp = {
        "c>ma50": c > ma50,
        "c>ma150": c > ma150,
        "ma50>ma150": ma50 > ma150,
        "ma150 rising": ma150 > ma150.shift(21),
        "within 25% of 52w high": (c / hi52 - 1) * 100 > -25,
        ">25% above 52w low": (c / lo52 - 1) * 100 > 25,
    }
    st = comp["c>ma50"]
    for k, v in comp.items():
        st = st & v
    return st.fillna(False), comp, ma50, ma150


def one(args):
    name, path, outdir = args
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    code = os.path.basename(path).split("_")[0]
    mk = "HK" if len(code) == 5 else "CN-A"
    d, _ = load_registry()[name](market=mk, stock_code=code).analyze(pd.read_csv(path))
    c = d["close"]; dates = pd.to_datetime(d["date"])
    st, comp, ma50, ma150 = stage2(d)
    pos = d["buy_signal"].astype(bool)
    fwd = (c.shift(-FWD) / c - 1) * 100
    fresh = st & ~st.shift(1).fillna(False)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(19, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax.plot(dates, c, lw=0.9, color="#222", zorder=3)
    ax.plot(dates, ma50, lw=0.8, color="#1f77b4", label="MA50")
    ax.plot(dates, ma150, lw=0.9, color="#d62728", label="MA150")
    lo, hi = c.min(), c.max()
    ax.fill_between(dates, lo, hi, where=st.to_numpy(), color="#d62728", alpha=0.10,
                    label="Weinstein stage 2 (\"main advance\")")
    ax.fill_between(dates, lo, lo + (hi - lo) * 0.05, where=pos.to_numpy(), color="#2ca02c", alpha=0.55,
                    label="strategy holding")
    good = bad = 0
    for i in np.where(fresh.to_numpy())[0]:
        if np.isnan(fwd.iloc[i]): continue
        v = fwd.iloc[i]; good += v > 0; bad += v <= 0
        ax.annotate(f"{v:+.0f}%", (dates.iloc[i], c.iloc[i]), textcoords="offset points",
                    xytext=(0, 10), fontsize=7, ha="center",
                    color="#2ca02c" if v > 0 else "#d62728", fontweight="bold")
        ax.scatter(dates.iloc[i], c.iloc[i], marker="o", s=26, zorder=5,
                   color="#2ca02c" if v > 0 else "#d62728")
    bh = (c.iloc[-1] / c.iloc[0] - 1) * 100
    med = fwd[fresh].median()
    ax.set_title(f"{code} {mk}   buy&hold {bh:+.0f}%   |   stage-2 state is on {st.mean()*100:.0f}% of bars; "
                 f"{int(fresh.sum())} fresh fires, median forward-{FWD} return {med:+.1f}% "
                 f"({good} up / {bad} down)   |   dots = a fire, label = what happened next")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.2)

    share = pd.DataFrame({k: v.astype(float) for k, v in comp.items()}).rolling(20, min_periods=1).mean()
    for k in comp:
        ax2.plot(dates, share[k], lw=0.7, label=k)
    ax2.set_ylim(-0.02, 1.02); ax2.legend(loc="upper left", fontsize=6, ncol=3); ax2.grid(alpha=0.2)
    ax2.set_ylabel("template parts\n(20d mean)", fontsize=7)
    fig.tight_layout(); os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{code}.png"), dpi=90); plt.close(fig)

    uncond = fwd.median()
    return dict(code=code, mkt=mk, bh=bh, on_share=st.mean() * 100, fires=int(fresh.sum()),
                fwd_med=med, fwd_mean=fwd[fresh].mean(), uncond=uncond, excess=med - uncond,
                in_state_fwd=fwd[st].median(), off_state_fwd=fwd[~st].median(),
                strat_expo_in=pos[st].mean() * 100 if st.any() else np.nan,
                strat_expo_out=pos[~st].mean() * 100 if (~st).any() else np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stocks", nargs="+", required=True)
    ap.add_argument("--pool", default="dev"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(resolve_pool(a.pool), "*_hfq.csv")))
    sel = [f for f in files if os.path.basename(f).split("_")[0] in set(a.stocks)]
    outdir = os.path.join(OUT, "advance_charts")
    with Pool(a.workers) as p:
        rows = p.map(one, [("baseline", f, outdir) for f in sel], chunksize=1)
    T = pd.DataFrame(rows)
    print(f"charts -> {outdir}\n")
    print(f"{'code':<8}{'mkt':<6}{'B&H':>9}{'state on':>10}{'fires':>7}{'fwd60 at fire':>15}{'uncond':>9}"
          f"{'excess':>9}{'in-state':>10}{'off-state':>11}{'strat expo in/out':>20}")
    for _, r in T.sort_values("bh", ascending=False).iterrows():
        print(f"{r.code:<8}{r.mkt:<6}{r.bh:>+8.0f}%{r.on_share:>9.0f}%{r.fires:>7.0f}{r.fwd_med:>+14.1f}%"
              f"{r.uncond:>+8.1f}%{r.excess:>+8.1f}%{r.in_state_fwd:>+9.1f}%{r.off_state_fwd:>+10.1f}%"
              f"{r.strat_expo_in:>13.0f}%/{r.strat_expo_out:<5.0f}%")
    print(f"\n  median across these {len(T)} stocks: excess of the stage-2 state = {T.excess.median():+.2f}pp;"
          f"  in-state fwd {T.in_state_fwd.median():+.1f}% vs off-state {T.off_state_fwd.median():+.1f}%")


if __name__ == "__main__":
    main()
