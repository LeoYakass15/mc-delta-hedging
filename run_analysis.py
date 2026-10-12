"""Lance toutes les études et écrit graphiques + tableaux dans results/.

Usage :
    python run_analysis.py                 # paramètres par défaut
    python run_analysis.py --paths 5000    # plus rapide
    python run_analysis.py --tc 0.002      # coût de transaction de 20 pb
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
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--sigma", type=float, default=0.20, help="vol de référence")
    p.add_argument("--option", choices=["call", "put"], default="call")
    p.add_argument("--paths", type=int, default=20_000, help="trajectoires par simulation de couverture")
    p.add_argument("--tc", type=float, default=0.001, help="coût de transaction proportionnel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    base = dict(S0=a.S0, K=a.K, T=a.T, r=a.r, option=a.option)

    # 1. Pricer Monte Carlo
    print("== 1. Pricer Monte Carlo ==")
    bs = float(bs_price(a.S0, a.K, a.T, a.r, a.sigma, a.option))
    mc = mc_european_price(a.S0, a.K, a.T, a.r, a.sigma, a.option, n_paths=500_000, seed=a.seed)
    print(f"Black-Scholes : {bs:.4f}\nMonte Carlo   : {mc}")
    conv = analysis.mc_convergence(a.S0, a.K, a.T, a.r, a.sigma, a.option, seed=a.seed)
    conv.to_csv(out / "mc_convergence.csv", index=False)
    plots.plot_mc_convergence(conv, out / "01_mc_convergence.png")
    std = conv.pivot(index="n_paths", columns="method", values="std_error")
    print("Facteur de réduction de variance (vs standard) :")
    print((std["standard"].values[:, None] ** 2 / std.values ** 2).mean(axis=0).round(1),
          "pour", list(std.columns))

    # 2. Erreur de couverture : grille (σ, N)
    print("\n== 2. Erreur de couverture vs σ et N (vol implicite = vol réelle) ==")
    sigmas = [0.10, 0.20, 0.30, 0.40, 0.60]
    freqs = [4, 12, 26, 52, 104, 252, 504, 1008]
    grid = analysis.hedging_grid(sigmas, freqs, n_paths=a.paths, seed=a.seed, **base)
    grid.to_csv(out / "hedging_grid.csv", index=False)
    plots.plot_std_vs_rebal(grid, out / "02_std_vs_rebal.png")
    plots.plot_std_vs_sigma(grid, out / "03_std_vs_sigma.png")
    plots.plot_heatmap(grid, out / "04_heatmap_rel_std.png")
    slopes = analysis.convergence_slopes(grid)
    print(slopes.round(3).to_string(index=False))
    print(f"Pente moyenne log(std) ~ log(N) : {slopes['slope'].mean():.3f}  (théorie : -0.5)")
    ratio = (grid["std_pnl"] / grid["derman_kamal_std"])
    print(f"Ratio simulé / Derman-Kamal : moyenne {ratio.mean():.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}")

    # 3. Distributions du P&L
    dists = {n: simulate_delta_hedge(sigma_real=a.sigma, n_rebal=n, n_paths=a.paths, seed=a.seed, **base)
             for n in (12, 52, 252)}
    plots.plot_pnl_distributions(dists, out / "05_pnl_distributions.png")

    # 4. Vol réalisée ≠ vol implicite
    print("\n== 3. Vol réalisée ≠ vol implicite ==")
    mism = analysis.vol_mismatch(a.sigma, np.round(np.linspace(0.10, 0.30, 9), 4), n_rebal=252,
                                 n_paths=a.paths, seed=a.seed, **base)
    mism.to_csv(out / "vol_mismatch.csv", index=False)
    plots.plot_vol_mismatch(mism, out / "06_vol_mismatch.png")
    print(mism[["sigma_real", "mean_pnl", "theory_mean", "std_pnl"]].round(3).to_string(index=False))

    # 5. Coûts de transaction
    print(f"\n== 4. Coûts de transaction ({a.tc:.2%}) ==")
    tc = analysis.transaction_cost_tradeoff([4, 12, 26, 52, 104, 252, 504, 1008], a.tc, sigma=a.sigma,
                                            n_paths=a.paths, seed=a.seed, **base)
    tc.to_csv(out / "tc_tradeoff.csv", index=False)
    plots.plot_tc_tradeoff(tc, out / "07_tc_tradeoff.png")
    print(tc[["n_rebal", "std_pnl", "mean_costs", "objective"]].round(3).to_string(index=False))

    print(f"\nRésultats écrits dans {out.resolve()}")


if __name__ == "__main__":
    main()
