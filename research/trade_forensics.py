"""Trade-level forensics: every entry and exit of a strategy on every stock, with the state that was
known at the time, what happened during the trade, and what happened after the exit.

  python3 research/trade_forensics.py baseline --pool pool_b_dev --workers 8 --charts 10
  python3 research/trade_forensics.py baseline --pool pool_b_dev --report

Writes research/out/forensics/<strategy>__<pool>/ledger.csv, stocks.csv and charts/<code>.png.
States are computed from the stock's own history at the entry bar's close (known when the order is
placed; the trade's return accrues from the next bar), so no state contains the trade's own outcome.
"""
import argparse, os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from hooked import load_registry
from harness import resolve_pool, pool_label, OUT

POST = 20   # bars after the exit: did the stock keep going?


def _states(d):
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
    lr = np.log(close).diff()
    rv20 = lr.rolling(20).std() * np.sqrt(252) * 100
    ma120 = close.rolling(120).mean()
    slope = (ma120 / ma120.shift(20) - 1) * 100
    hi60, lo60 = high.rolling(60).max(), low.rolling(60).min()
    rng = (high - low).replace(0, np.nan)
    delta = close.diff(); up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / dn.rolling(14).mean().replace(0, np.nan)
    return pd.DataFrame({
        "vol_state": rv20.rolling(250, min_periods=120).rank(pct=True),
        "slope_state": slope.rolling(250, min_periods=120).rank(pct=True),
        "above_ma120": (close > ma120).astype(float),
        "range_pos60": ((close - lo60) / (hi60 - lo60).replace(0, np.nan)),
        "dist_hi250": (close / close.rolling(250, min_periods=120).max() - 1) * 100,
        "relvol": vol / vol.rolling(60).median(),
        "ibs": ((close - low) / rng),
        "rsi14": 100 - 100 / (1 + rs),
        "ret20": (close / close.shift(20) - 1) * 100,
    })


def one(args):
    name, path = args
    code = os.path.basename(path).split("_")[0]
    mk = "HK" if len(code) == 5 else "CN-A"
    df = pd.read_csv(path)
    st = load_registry()[name](market=mk, stock_code=code)
    d, _ = st.analyze(df)
    bt = st.backtest(d, 10000.0)
    S = _states(d)
    dates = pd.to_datetime(d["date"]); idx = {str(x)[:10]: i for i, x in enumerate(dates)}
    close = d["close"].to_numpy()
    rows = []
    for t in bt["trades"]:
        b = idx.get(str(t["buy_date"])[:10]); s = idx.get(str(t["sell_date"])[:10])
        if b is None or s is None: continue
        seg = close[b:s + 1]
        post = close[min(s + POST, len(close) - 1)] / close[s] - 1
        rows.append(dict(code=code, mkt=mk, buy_date=dates[b].date(), sell_date=dates[s].date(),
                         buy_price=t["buy_price"], sell_price=t["sell_price"], ret_pct=t["profit_rate"] * 100,
                         hold_bars=s - b, entry_reason=d["entry_reason"].iloc[b], exit_reason=d["exit_reason"].iloc[s],
                         mfe_pct=(seg.max() / close[b] - 1) * 100, mae_pct=(seg.min() / close[b] - 1) * 100,
                         post20_pct=post * 100, period="discover" if dates[b] < pd.Timestamp("2024-01-01") else "validate",
                         **{k: float(S[k].iloc[b]) for k in S.columns}))
    bh = (close[-1] / close[0] - 1) * 100
    stock = dict(code=code, mkt=mk, strat_ret=bt["total_return"], bh_ret=bh, excess=bt["total_return"] - bh,
                 n_trades=len(rows), win_rate=np.mean([r["ret_pct"] > 0 for r in rows]) * 100 if rows else np.nan,
                 mdd=bt.get("max_drawdown", np.nan))
    return rows, stock


def daily_one(args):
    """Every bar of one stock: price, the causal state features, the position and the signals."""
    name, path = args
    code = os.path.basename(path).split("_")[0]
    mk = "HK" if len(code) == 5 else "CN-A"
    st = load_registry()[name](market=mk, stock_code=code)
    d, _ = st.analyze(pd.read_csv(path))
    S = _states(d)
    out = pd.DataFrame({
        "code": code, "mkt": mk, "date": pd.to_datetime(d["date"]),
        "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"], "volume": d["volume"],
        "ret": d["close"].pct_change() * 100,
        "pos": d["buy_signal"], "entry": d["entry_signal"], "exit": d["exit_signal"],
        "entry_reason": d["entry_reason"], "exit_reason": d["exit_reason"],
        "trend_dir": d["trend_direction"], "heikin_bull": d["is_heikin_bullish"].astype(int),
        "dist_ma60": d["dist_ma60"], "atr_pct": d["atr"] / d["close"] * 100,
        "atr_expanding": d["atr_expanding"].astype(int), "m_top": d["is_m_top"].astype(int),
        "volume_weak": d["volume_weak"].astype(int),
    })
    for c in S.columns:
        out[c] = S[c].values
    # forward outcomes, for labelling only - never a feature
    c = d["close"]
    out["fwd20_max"] = (c.shift(-1).rolling(20, min_periods=1).max().shift(-19) / c - 1) * 100
    out["fwd60_max"] = (c.shift(-1).rolling(60, min_periods=1).max().shift(-59) / c - 1) * 100
    return out


def chart(name, path, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    code = os.path.basename(path).split("_")[0]
    mk = "HK" if len(code) == 5 else "CN-A"
    df = pd.read_csv(path)
    st = load_registry()[name](market=mk, stock_code=code)
    d, _ = st.analyze(df); bt = st.backtest(d, 10000.0)
    S = _states(d); dates = pd.to_datetime(d["date"]); close = d["close"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax.plot(dates, close, lw=0.8, color="#333"); ax.plot(dates, close.rolling(120).mean(), lw=0.8, color="#1f77b4", label="MA120")
    pos = d["buy_signal"].astype(bool).to_numpy()
    ax.fill_between(dates, close.min(), close.max(), where=pos, color="#2ca02c", alpha=0.07, label="in position")
    idx = {str(x)[:10]: i for i, x in enumerate(dates)}
    colour = lambda r: "#d62728" if "止损" in r else ("#ff7f0e" if ("转空" in r or "趋势" in r) else "#9467bd")
    for t in bt["trades"]:
        b = idx.get(str(t["buy_date"])[:10]); s = idx.get(str(t["sell_date"])[:10])
        if b is None or s is None: continue
        ax.scatter(dates[b], close[b], marker="^", s=70, color="#2ca02c", zorder=5)
        ax.scatter(dates[s], close[s], marker="v", s=70, color=colour(d["exit_reason"].iloc[s]), zorder=5)
        ax.annotate(f"{t['profit_rate']*100:+.0f}%", (dates[s], close[s]), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.set_title(f"{code} {mk}  {name}: strategy {bt['total_return']:+.1f}%  vs buy&hold {(close.iloc[-1]/close.iloc[0]-1)*100:+.1f}%   "
                 f"({len(bt['trades'])} trades)   ▲ entry  ▼ exit (red=stop, orange=trend, purple=other)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.2)
    ax2.plot(dates, S["vol_state"], lw=0.8, color="#8c564b", label="vol state (own pct)")
    ax2.fill_between(dates, 0, 1, where=(S["vol_state"] > 2 / 3).to_numpy(), color="#8c564b", alpha=0.15, label="high vol")
    ax2.plot(dates, S["slope_state"], lw=0.8, color="#1f77b4", label="MA120 slope (own pct)")
    ax2.set_ylim(0, 1); ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.2)
    fig.tight_layout(); os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{code}.png"), dpi=90); plt.close(fig)


def bucket(x, edges, labels):
    return pd.cut(x, edges, labels=labels, include_lowest=True)


def report(L, S):
    pd.set_option("display.width", 200)
    def agg(g):
        pos = g.ret_pct[g.ret_pct > 0].sum(); neg = -g.ret_pct[g.ret_pct <= 0].sum()
        return pd.Series({"n": len(g), "win%": (g.ret_pct > 0).mean() * 100, "mean%": g.ret_pct.mean(),
                          "tPF": pos / neg if neg > 0 else np.nan, "share_of_sum%": g.ret_pct.sum(),
                          "mfe%": g.mfe_pct.mean(), "mae%": g.mae_pct.mean(), "post20%": g.post20_pct.mean(), "hold": g.hold_bars.mean()})
    tot = L.ret_pct.sum()
    def show(title, key):
        T = L.groupby(key, observed=True).apply(agg); T["share_of_sum%"] = T["share_of_sum%"] / tot * 100
        print(f"\n=== {title} ===\n" + T.round(2).to_string())
    print(f"ledger: {len(L)} trades on {L.code.nunique()} stocks; sum of trade returns {tot:.0f} pp; tPF {L.ret_pct[L.ret_pct>0].sum()/-L.ret_pct[L.ret_pct<=0].sum():.3f}")
    show("by EXIT reason  (post20% = what the stock did in the 20 bars AFTER the exit)", "exit_reason")
    show("by ENTRY reason", "entry_reason")
    L["vol_t"] = bucket(L.vol_state, [0, 1/3, 2/3, 1], ["low", "mid", "high"]); show("by VOL state at entry (own pct)", "vol_t")
    L["slope_t"] = bucket(L.slope_state, [0, 1/3, 2/3, 1], ["falling", "flat", "rising"]); show("by MA120 SLOPE state at entry", "slope_t")
    L["rng_t"] = bucket(L.range_pos60, [0, 0.25, 0.75, 1], ["bottom", "middle", "top"]); show("by POSITION in 60d range at entry", "rng_t")
    L["hi_t"] = bucket(L.dist_hi250, [-100, -30, -15, -5, 0.01], ["<-30%", "-30..-15", "-15..-5", "near high"]); show("by DISTANCE below 250d high at entry", "hi_t")
    L["rv_t"] = bucket(L.relvol, [0, 0.8, 1.5, 3, 100], ["quiet", "normal", "elevated", "climax"]); show("by RELATIVE VOLUME at entry", "rv_t")
    L["r20_t"] = bucket(L.ret20, [-100, -10, 0, 10, 1000], ["<-10%", "-10..0", "0..10", ">10%"]); show("by TRAILING 20d return at entry", "r20_t")
    # sign consistency inside this pool: CN-A/HK x discover/validate
    print("\n=== 4-cell sign check inside this pool (mean trade % by state; a real effect keeps its ORDER in all 4) ===")
    for key in ("vol_t", "slope_t", "rng_t", "hi_t", "rv_t", "r20_t"):
        T = L.groupby(["mkt", "period", key], observed=True).ret_pct.mean().unstack(key).round(2)
        print(f"\n[{key}]\n" + T.to_string())
    print("\n=== STOCK level: best and worst 10 by excess over buy&hold ===")
    S = S.sort_values("excess")
    print(S.tail(10).round(1).to_string(index=False)); print("..."); print(S.head(10).round(1).to_string(index=False))
    print(f"\nstocks: beats B&H {(S.excess>0).mean()*100:.1f}%  median excess {S.excess.median():+.1f}pp  "
          f"top 10 stocks hold {S.strat_ret.nlargest(10).sum()/S.strat_ret.sum()*100:.0f}% of summed strategy return")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("strategy"); ap.add_argument("--pool", default="pool_b_dev")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--charts", type=int, default=0, help="chart the N best and N worst stocks")
    ap.add_argument("--stocks", nargs="*", help="chart these codes"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--all-charts", action="store_true", help="chart every stock in the pool")
    ap.add_argument("--daily", action="store_true", help="also dump the per-day feature panel")
    a = ap.parse_args()
    data_dir = resolve_pool(a.pool); outdir = os.path.join(OUT, "forensics", f"{a.strategy}__{pool_label(data_dir)}")
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*_hfq.csv")))
    lp, sp = os.path.join(outdir, "ledger.csv"), os.path.join(outdir, "stocks.csv")
    if not (os.path.exists(lp) and os.path.exists(sp)) or not a.report:
        with Pool(a.workers) as p:
            res = p.map(one, [(a.strategy, f) for f in files], chunksize=4)
        L = pd.DataFrame([r for rows, _ in res for r in rows]); S = pd.DataFrame([s for _, s in res])
        L.to_csv(lp, index=False); S.to_csv(sp, index=False); print(f"wrote {lp} ({len(L)} trades) and {sp}")
    L = pd.read_csv(lp, dtype={"code": str}); S = pd.read_csv(sp, dtype={"code": str})
    codes = list(a.stocks or [])
    if a.charts:
        S2 = S.sort_values("excess"); codes += list(S2.head(a.charts).code) + list(S2.tail(a.charts).code)
    if a.all_charts:
        codes = [os.path.basename(f).split("_")[0] for f in files]
    if a.daily:
        with Pool(a.workers) as p:
            rows = p.map(daily_one, [(a.strategy, f) for f in files], chunksize=4)
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(outdir, "daily.csv"), index=False)
        print("wrote", os.path.join(outdir, "daily.csv"))
    for c in codes:
        f = [x for x in files if os.path.basename(x).startswith(c + "_")]
        if f: chart(a.strategy, f[0], os.path.join(outdir, "charts")); print("chart", c)
    if a.report: report(L, S)


if __name__ == "__main__":
    main()
