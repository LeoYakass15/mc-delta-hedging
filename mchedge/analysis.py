"""Parametric studies: MC convergence and validation, hedging error vs volatility and frequency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .black_scholes import bs_price
from .hedging import derman_kamal_std, simulate_delta_hedge
from .monte_carlo import mc_european_price


def mc_convergence(S0, K, T, r, sigma, option="call", path_counts=(1_000, 4_000, 16_000, 64_000, 256_000),
                   seed=0) -> pd.DataFrame:
    """MC price and standard error as a function of the number of paths and the method."""
    ref = float(bs_price(S0, K, T, r, sigma, option))
    methods = {
        "standard": dict(antithetic=False, control_variate=False),
        "antithetic": dict(antithetic=True, control_variate=False),
        "control variate": dict(antithetic=False, control_variate=True),
        "antithetic + control": dict(antithetic=True, control_variate=True),
    }
    rows = []
    for n in path_counts:
        for name, kw in methods.items():
            res = mc_european_price(S0, K, T, r, sigma, option, n_paths=n, seed=seed, **kw)
            rows.append({"n_paths": n, "method": name, "price": res.price, "std_error": res.std_error,
                         "bs_price": ref, "error": res.price - ref})
    return pd.DataFrame(rows)


def mc_vs_bs_table(S0=100.0, r=0.02, strikes=(80, 90, 100, 110, 120), sigmas=(0.1, 0.2, 0.4),
                   maturities=(0.25, 1.0), options=("call", "put"), n_paths=200_000, seed=0) -> pd.DataFrame:
    """Validation of the MC pricer against Black-Scholes over a grid of options.

    z = (MC price - BS price) / standard error. If the pricer is unbiased, z is ~N(0, 1):
    about 95% of |z| should be below 1.96.

    Degenerate case: a deep out-of-the-money option can have a zero payoff on every path,
    hence a zero standard error. z is then set to 0 if the BS price is also ~0 (< 1e-4), and the
    option is flagged in the `degenerate` column.
    """
    rows, i = [], 0
    for option in options:
        for T in maturities:
            for sig in sigmas:
                for K in strikes:
                    res = mc_european_price(S0, K, T, r, sig, option, n_paths=n_paths,
                                            seed=None if seed is None else seed + i)
                    bs = float(bs_price(S0, K, T, r, sig, option))
                    err = res.price - bs
                    degenerate = res.std_error < 1e-12
                    if degenerate:
                        z = 0.0 if abs(err) < 1e-4 else np.sign(err) * np.inf
                    else:
                        z = err / res.std_error
                    rows.append({"option": option, "K": K, "T": T, "sigma": sig,
                                 "bs_price": bs, "mc_price": res.price, "std_error": res.std_error,
                                 "abs_error": err,
                                 "rel_error_pct": 100 * err / bs if bs > 0.01 else np.nan,
                                 "z": z, "bs_in_ci95": abs(z) <= 1.96, "degenerate": degenerate})
                    i += 1
    return pd.DataFrame(rows)


def hedging_grid(sigmas, rebal_freqs, S0=100.0, K=100.0, T=1.0, r=0.02, option="call",
                 n_paths=20_000, tc=0.0, seed=0) -> pd.DataFrame:
    """Hedging error (implied vol = realised vol) over a (sigma, N) grid."""
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
    """Regression log(std) = a + b log(N) for each volatility; theory: b = -1/2."""
    out = []
    for sig, g in grid.groupby("sigma_real"):
        b, a = np.polyfit(np.log(g["n_rebal"]), np.log(g["std_pnl"]), 1)
        out.append({"sigma": sig, "slope": b, "intercept": a})
    return pd.DataFrame(out)


def vol_mismatch(sigma_implied, sigma_reals, n_rebal=252, S0=100.0, K=100.0, T=1.0, r=0.02,
                 option="call", n_paths=20_000, seed=0) -> pd.DataFrame:
    """P&L when the realised vol differs from the pricing/hedging vol.

    Analytical approximation of the mean (option seller, delta at implied vol):
    E[P&L] ≈ BS_price(σ_imp) - BS_price(σ_real).
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
    """Risk / cost trade-off: more frequent rebalancing means less risk but higher costs.

    Mean-variance criterion: total cost = mean costs + λ · P&L standard deviation.
    """
    rows = []
    for j, n in enumerate(rebal_freqs):
        res = simulate_delta_hedge(S0, K, T, r, sigma, None, n, n_paths, option, tc=tc,
                                   seed=None if seed is None else seed + j)
        row = res.summary()
        row["objective"] = row["mean_costs"] + risk_aversion * row["std_pnl"]
        rows.append(row)
    return pd.DataFrame(rows)
