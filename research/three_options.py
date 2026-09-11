"""Compare the three remaining options on identical stocks, days and fees.

  A  BLEND      w of buy-and-hold + (1-w) of the strategy, per stock
  B  RISK ONLY  the strategy as-is, judged on risk-adjusted metrics rather than return
  C  REDEPLOY   pool capital ACROSS stocks: when a stock is flat its capital goes to another signalled
                stock. No signal is changed - only capital utilisation.

Fees follow the project model: CN-A 1.5bp buy / 6.5bp sell, HK 14.5bp per side, charged on |dpos|.
"""
import sys, os, glob, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from hooked import load_registry
from harness import resolve_pool

FEE = {"CN-A": (1.5e-4, 6.5e-4), "HK": (14.5e-4, 14.5e-4)}


def load_one(path):
    code = os.path.basename(path).split("_")[0]
    mk = "HK" if len(code) == 5 else "CN-A"
    d, _ = load_registry()["baseline"](market=mk, stock_code=code).analyze(pd.read_csv(path))
    return pd.DataFrame({"date": pd.to_datetime(d["date"]), "code": code, "mkt": mk,
                         "r": d["close"].pct_change().fillna(0.0),
                         "pos": d["buy_signal"].shift(1).fillna(0.0)})


def net(r, pos, mk):
    """Net daily return of a position series, fees on turnover."""
    buy, sell = FEE[mk]
    dp = np.diff(pos, prepend=0.0)
    cost = np.where(dp > 0, dp * buy, -dp * sell)
    return pos * r - cost


def stats(daily, yrs):
    eq = np.cumprod(1 + daily)
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100
    mdd = ((eq / np.maximum.accumulate(eq) - 1).min()) * 100
    sd = daily.std() * np.sqrt(252)
    dn = daily[daily < 0].std() * np.sqrt(252) if (daily < 0).any() else np.nan
    return dict(cagr=cagr, mdd=mdd, vol=sd * 100,
                sharpe=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
                sortino=daily.mean() * 252 / dn if dn and dn > 0 else np.nan,
                calmar=cagr / abs(mdd) if mdd < 0 else np.nan,
                total=(eq[-1] - 1) * 100)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pools", nargs="*", default=["pool_b_dev", "oos"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    WS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for pool in a.pools:
        files = sorted(glob.glob(os.path.join(resolve_pool(pool), "*_hfq.csv")))
        with Pool(a.workers) as p:
            parts = p.map(load_one, files, chunksize=4)
        D = pd.concat(parts, ignore_index=True)
        dates = np.sort(D.date.unique()); yrs = (dates[-1] - dates[0]) / np.timedelta64(365, "D")
        R = D.pivot(index="date", columns="code", values="r").reindex(dates).fillna(0.0)
        P = D.pivot(index="date", columns="code", values="pos").reindex(dates).ffill().fillna(0.0)
        MK = D.groupby("code").mkt.first()
        n = R.shape[1]
        print(f"\n{'='*104}\n{pool}: {n} stocks, {len(dates)} days, {yrs:.1f} years\n{'='*104}")

        # ---------- per-stock net daily matrices ----------
        def netmat(pos):
            out = np.empty_like(pos.to_numpy(dtype=float))
            for k, c in enumerate(pos.columns):
                out[:, k] = net(R[c].to_numpy(), pos[c].to_numpy(), MK[c])
            return pd.DataFrame(out, index=pos.index, columns=pos.columns)

        bh_pos = pd.DataFrame(1.0, index=P.index, columns=P.columns)
        NB = netmat(bh_pos); NS = netmat(P)

        print("\nA  BLEND: w of buy-and-hold + (1-w) of the strategy, equal-weight portfolio, net of fees")
        print(f"  {'w':>5}{'capital used':>14}{'CAGR':>8}{'vol':>8}{'MDD':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}"
              f"{'per-stock median':>18}{'beats B&H':>11}")
        blend_rows = {}
        for w in WS:
            pos = (1 - w) * P + w * bh_pos
            N = netmat(pos)
            port = N.mean(axis=1).to_numpy()
            st = stats(port, yrs); blend_rows[w] = st
            per = (np.expm1(np.log1p(N).sum())) * 100
            bhper = (np.expm1(np.log1p(NB).sum())) * 100
            print(f"  {w:>5.1f}{pos.mean(axis=1).mean()*100:>13.1f}%{st['cagr']:>7.2f}%{st['vol']:>7.1f}%{st['mdd']:>8.1f}%"
                  f"{st['sharpe']:>8.2f}{st['sortino']:>9.2f}{st['calmar']:>8.2f}{per.median():>17.2f}%{(per>bhper).mean()*100:>10.1f}%")

        print("\nB  RISK-ONLY VIEW: the strategy as-is, judged on risk rather than return")
        pb = NB.mean(axis=1).to_numpy(); ps = NS.mean(axis=1).to_numpy()
        sb, ss = stats(pb, yrs), stats(ps, yrs)
        print(f"  {'':<22}{'buy&hold':>12}{'strategy':>12}{'change':>12}")
        for k, lab in (("cagr", "CAGR %"), ("vol", "vol %"), ("mdd", "max drawdown %"), ("sharpe", "Sharpe"),
                       ("sortino", "Sortino"), ("calmar", "Calmar")):
            print(f"  {lab:<22}{sb[k]:>12.2f}{ss[k]:>12.2f}{ss[k]-sb[k]:>+12.2f}")
        pm = pd.DataFrame({"bh": (np.expm1(np.log1p(NB).sum())) * 100, "st": (np.expm1(np.log1p(NS).sum())) * 100})
        mddb = NB.apply(lambda x: ((np.cumprod(1+x)/np.maximum.accumulate(np.cumprod(1+x))-1).min())*100)
        mdds = NS.apply(lambda x: ((np.cumprod(1+x)/np.maximum.accumulate(np.cumprod(1+x))-1).min())*100)
        print(f"  per-stock: median return {pm.bh.median():+.2f}% -> {pm.st.median():+.2f}%;  "
              f"median max drawdown {mddb.median():.1f}% -> {mdds.median():.1f}%  "
              f"(shallower on {(mdds>mddb).mean()*100:.0f}% of stocks)")

        print("\nC  REDEPLOY: pool the idle capital across stocks (same signals, only utilisation changes)")
        sig = P.to_numpy(); cnt = sig.sum(axis=1)
        print(f"  {'variant':<34}{'capital used':>14}{'CAGR':>8}{'vol':>8}{'MDD':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}")
        for cap, lab in ((None, "C1 full redeploy (no cap)"), (0.10, "C2 cap 10% per stock"), (0.05, "C3 cap 5% per stock")):
            W = np.zeros_like(sig)
            live = cnt > 0
            if cap is None:
                W[live] = sig[live] / cnt[live][:, None]
            else:
                base = np.zeros_like(sig)
                base[live] = np.minimum(cap, 1.0 / cnt[live][:, None] * 0 + cap)
                base = np.where(sig > 0, cap, 0.0)
                tot = base.sum(axis=1, keepdims=True)
                W = np.where(tot > 1.0, base / np.maximum(tot, 1e-9), base)
            Wdf = pd.DataFrame(W, index=P.index, columns=P.columns)
            NW = netmat(Wdf)
            port = NW.sum(axis=1).to_numpy()
            st = stats(port, yrs)
            print(f"  {lab:<34}{W.sum(axis=1).mean()*100:>13.1f}%{st['cagr']:>7.2f}%{st['vol']:>7.1f}%{st['mdd']:>8.1f}%"
                  f"{st['sharpe']:>8.2f}{st['sortino']:>9.2f}{st['calmar']:>8.2f}")
        print(f"  reference  equal-weight buy&hold        {100.0:>12.1f}%{sb['cagr']:>7.2f}%{sb['vol']:>7.1f}%{sb['mdd']:>8.1f}%"
              f"{sb['sharpe']:>8.2f}{sb['sortino']:>9.2f}{sb['calmar']:>8.2f}")
        print(f"  reference  strategy, no redeploy        {P.mean(axis=1).mean()*100:>12.1f}%{ss['cagr']:>7.2f}%{ss['vol']:>7.1f}%"
              f"{ss['mdd']:>8.1f}%{ss['sharpe']:>8.2f}{ss['sortino']:>9.2f}{ss['calmar']:>8.2f}")
        print(f"  signalled stocks per day: median {np.median(cnt):.0f} of {n}  ({np.median(cnt)/n*100:.0f}%)")


if __name__ == "__main__":
    main()
