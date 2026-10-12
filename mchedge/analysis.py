"""Études paramétriques : convergence MC, erreur de couverture vs volatilité et fréquence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .black_scholes import bs_price
from .hedging import derman_kamal_std, simulate_delta_hedge
from .monte_carlo import mc_european_price


def mc_convergence(S0, K, T, r, sigma, option="call", path_counts=(1_000, 4_000, 16_000, 64_000, 256_000),
                   seed=0) -> pd.DataFrame:
    """Prix MC et erreur standard selon le nombre de trajectoires et la méthode."""
    ref = float(bs_price(S0, K, T, r, sigma, option))
    methods = {
        "standard": dict(antithetic=False, control_variate=False),
        "antithétique": dict(antithetic=True, control_variate=False),
        "contrôle": dict(antithetic=False, control_variate=True),
        "antithétique + contrôle": dict(antithetic=True, control_variate=True),
    }
    rows = []
    for n in path_counts:
        for name, kw in methods.items():
            res = mc_european_price(S0, K, T, r, sigma, option, n_paths=n, seed=seed, **kw)
            rows.append({"n_paths": n, "method": name, "price": res.price, "std_error": res.std_error,
                         "bs_price": ref, "error": res.price - ref})
    return pd.DataFrame(rows)


def hedging_grid(sigmas, rebal_freqs, S0=100.0, K=100.0, T=1.0, r=0.02, option="call",
                 n_paths=20_000, tc=0.0, seed=0) -> pd.DataFrame:
    """Erreur de couverture (vol implicite = vol réelle) sur une grille (σ, N)."""
    rows = []
    for i, sig in enumerate(sigmas):
        for j, n in enumerate(rebal_freqs):
            res = simulate_delta_hedge(S0, K, T, r, sig, None, n, n_paths, option, tc=tc,
                                       seed=None if seed is None else seed + 1000 * i + j)
            row = res.summary()
            row["derman_kamal_std"] = derman_kamal_std(S0, K, T, r, sig, n)
            rows.append(row)
    return pd.DataFrame(rows)


def convergence_slopes(grid: pd.DataFrame) -> pd.DataFrame:
    """Régression log(std) = a + b log(N) par volatilité ; théorie : b = -1/2."""
    out = []
    for sig, g in grid.groupby("sigma_real"):
        b, a = np.polyfit(np.log(g["n_rebal"]), np.log(g["std_pnl"]), 1)
        out.append({"sigma": sig, "slope": b, "intercept": a})
    return pd.DataFrame(out)


def vol_mismatch(sigma_implied, sigma_reals, n_rebal=252, S0=100.0, K=100.0, T=1.0, r=0.02,
                 option="call", n_paths=20_000, seed=0) -> pd.DataFrame:
    """P&L quand la vol réalisée diffère de la vol de pricing/couverture.

    Approximation analytique de l'espérance (vendeur d'option, delta à vol implicite) :
    E[P&L] ≈ prix_BS(σ_imp) - prix_BS(σ_real).
    """
    rows = []
    for i, sr in enumerate(sigma_reals):
        res = simulate_delta_hedge(S0, K, T, r, sr, sigma_implied, n_rebal, n_paths, option,
                                   seed=None if seed is None else seed + i)
        row = res.summary()
        row["theory_mean"] = float(bs_price(S0, K, T, r, sigma_implied, option)
                                   - bs_price(S0, K, T, r, sr, option))
        rows.append(row)
    return pd.DataFrame(rows)


def transaction_cost_tradeoff(rebal_freqs, tc, sigma=0.2, S0=100.0, K=100.0, T=1.0, r=0.02,
                              option="call", n_paths=20_000, risk_aversion=1.0, seed=0) -> pd.DataFrame:
    """Arbitrage risque / coûts : plus on rebalance, moins de risque mais plus de coûts.

    Critère moyenne-variance : coût total = coûts moyens + λ · écart-type du P&L.
    """
    rows = []
    for j, n in enumerate(rebal_freqs):
        res = simulate_delta_hedge(S0, K, T, r, sigma, None, n, n_paths, option, tc=tc,
                                   seed=None if seed is None else seed + j)
        row = res.summary()
        row["objective"] = row["mean_costs"] + risk_aversion * row["std_pnl"]
        rows.append(row)
    return pd.DataFrame(rows)
