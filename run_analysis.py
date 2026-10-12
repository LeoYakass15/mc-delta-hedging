"""Run every study and write the charts and tables to results/.

Usage:
    python run_analysis.py                 # default parameters
    python run_analysis.py --paths 5000    # faster
    python run_analysis.py --tc 0.002      # 20 bp transaction cost
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mchedge import analysis, plots
from mchedge.black_scholes import bs_price
from mchedge.hedging import simulate_delta_hedge
from mchedge.monte_carlo import mc_european_price


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--S0", type=float, default=100.0, help="spot price")
    p.add_argument("--K", type=float, default=100.0, help="strike")
    p.add_argument("--T", type=float, default=1.0, help="maturity in years")
    p.add_argument("--r", type=float, default=0.02, help="risk-free rate")
    p.add_argument("--sigma", type=float, default=0.20, help="reference volatility")
    p.add_argument("--option", choices=["call", "put"], default="call")
    p.add_argument("--paths", type=int, default=20_000, help="paths per hedging simulation")
    p.add_argument("--tc", type=float, default=0.001, help="proportional transaction cost")
    p.add_argument("--seed", type=int, default=42, help="random seed (fixed for reproducibility)")
    p.add_argument("--out", default="results", help="output folder")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    base = dict(S0=a.S0, K=a.K, T=a.T, r=a.r, option=a.option)

    # 1. Monte Carlo pricer
    print("== 1. Monte Carlo pricer ==")
    bs = float(bs_price(a.S0, a.K, a.T, a.r, a.sigma, a.option))
    mc = mc_european_price(a.S0, a.K, a.T, a.r, a.sigma, a.option, n_paths=500_000, seed=a.seed)
    print(f"Black-Scholes : {bs:.4f}\nMonte Carlo   : {mc}")
    conv = analysis.mc_convergence(a.S0, a.K, a.T, a.r, a.sigma, a.option, seed=a.seed)
    conv.to_csv(out / "mc_convergence.csv", index=False)
    plots.plot_mc_convergence(conv, out / "01_mc_convergence.png")
    std = conv.pivot(index="n_paths", columns="method", values="std_error")
    gains = (std["standard"].values[:, None] ** 2 / std.values ** 2).mean(axis=0)
    print("Variance reduction factor vs standard MC:")
    for name, g in zip(std.columns, gains):
        print(f"  {name:<22} x{g:.1f}")

    # 1b. MC vs Black-Scholes validation over a grid of options
    val = analysis.mc_vs_bs_table(S0=a.S0, r=a.r, seed=a.seed)
    val.to_csv(out / "mc_vs_bs.csv", index=False)
    plots.plot_mc_vs_bs(val, out / "01b_mc_vs_bs.png")
    live = val[~val["degenerate"]]
    print(f"\n== 1b. MC vs Black-Scholes on {len(val)} options "
          "(5 strikes x 3 vols x 2 maturities x call/put, 200k paths) ==")
    print(f"  BS price inside the 95% CI : {val['bs_in_ci95'].sum()} / {len(val)} (expected ~95%)")
    print(f"  mean z {live['z'].mean():+.2f} (expected 0), std of z {live['z'].std():.2f} (expected 1)")
    print(f"  largest absolute error     : {val['abs_error'].abs().max():.4f}")
    if val["degenerate"].any():
        print(f"  {val['degenerate'].sum()} degenerate option(s) (zero payoff on every path, BS price ~0)")
    print(val[(val["T"] == 1.0) & (val["sigma"] == 0.2)]
          [["option", "K", "bs_price", "mc_price", "std_error", "z"]].round(4).to_string(index=False))

    # 2. Hedging error: (sigma, N) grid
    print("\n== 2. Hedging error vs sigma and N (implied vol = realised vol) ==")
    sigmas = [0.10, 0.20, 0.30, 0.40, 0.60]
    freqs = [4, 12, 26, 52, 104, 252, 504, 1008]
    grid = analysis.hedging_grid(sigmas, freqs, n_paths=a.paths, seed=a.seed, **base)
    grid.to_csv(out / "hedging_grid.csv", index=False)
    plots.plot_std_vs_rebal(grid, out / "02_std_vs_rebal.png")
    plots.plot_std_vs_sigma(grid, out / "03_std_vs_sigma.png")
    plots.plot_heatmap(grid, out / "04_heatmap_rel_std.png")
    slopes = analysis.convergence_slopes(grid)
    print(slopes.round(3).to_string(index=False))
    print(f"Mean slope of log(std) vs log(N): {slopes['slope'].mean():.3f}  (theory: -0.5)")
    ratio = grid["std_pnl"] / grid["derman_kamal_std"]
    print(f"Simulated / Derman-Kamal ratio: mean {ratio.mean():.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}")

    # 3. P&L distributions
    dists = {n: simulate_delta_hedge(sigma_real=a.sigma, n_rebal=n, n_paths=a.paths, seed=a.seed, **base)
             for n in (12, 52, 252)}
    plots.plot_pnl_distributions(dists, out / "05_pnl_distributions.png")

    # 4. Realised vol != implied vol
    print("\n== 3. Realised vol != implied vol ==")
    mism = analysis.vol_mismatch(a.sigma, np.round(np.linspace(0.10, 0.30, 9), 4), n_rebal=252,
                                 n_paths=a.paths, seed=a.seed, **base)
    mism.to_csv(out / "vol_mismatch.csv", index=False)
    plots.plot_vol_mismatch(mism, out / "06_vol_mismatch.png")
    print(mism[["sigma_real", "mean_pnl", "theory_mean", "std_pnl"]].round(3).to_string(index=False))

    # 5. Transaction costs
    print(f"\n== 4. Transaction costs ({a.tc:.2%}) ==")
    tc = analysis.transaction_cost_tradeoff([4, 12, 26, 52, 104, 252, 504, 1008], a.tc, sigma=a.sigma,
                                            n_paths=a.paths, seed=a.seed, **base)
    tc.to_csv(out / "tc_tradeoff.csv", index=False)
    plots.plot_tc_tradeoff(tc, out / "07_tc_tradeoff.png")
    print(tc[["n_rebal", "std_pnl", "mean_costs", "objective"]].round(3).to_string(index=False))

    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
