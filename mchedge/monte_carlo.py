"""Monte Carlo pricer for European options under Black-Scholes (GBM).

- Exact GBM simulation (no discretisation error for a terminal payoff).
- Variance reduction: antithetic variates and/or a control variate (discounted underlying).
- Returns the price, its standard error and the 95% confidence interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"{self.price:.4f} ± {1.96 * self.std_error:.4f} (95% CI [{lo:.4f}, {hi:.4f}], n={self.n_paths})"


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0, rng=None, antithetic=False):
    """GBM paths on a uniform grid, shape (n_paths, n_steps + 1).

    Exact scheme: S_{t+dt} = S_t * exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z).
    With antithetic=True, the second half of the paths uses -Z.
    """
    rng = np.random.default_rng(rng)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal(((n_paths + 1) // 2, n_steps))
        Z = np.concatenate([half, -half])[:n_paths]
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def mc_european_price(S0, K, T, r, sigma, option="call", q=0.0, n_paths=200_000,
                      antithetic=True, control_variate=True, seed=None) -> MCResult:
    """Monte Carlo price of a European option.

    Control variate: X = e^{-rT} S_T, whose expectation S0 e^{-qT} is known.
    With antithetic variates, the i.i.d. samples are the pair averages, so both the
    standard error and the control-variate coefficient are estimated on them.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    rng = np.random.default_rng(seed)
    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    disc = np.exp(-r * T)
    payoff = disc * (np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0))

    X = disc * ST
    if antithetic:  # i.i.d. samples = pair averages
        m = len(Z) // 2
        payoff, X = 0.5 * (payoff[:m] + payoff[m:]), 0.5 * (X[:m] + X[m:])

    if control_variate and payoff.var() > 0:
        # beta is estimated on the same i.i.d. samples (after pairing); estimating it on
        # individual draws makes antithetic + control worse than the control alone.
        cov = np.cov(payoff, X)
        beta = cov[0, 1] / cov[1, 1]
        payoff = payoff - beta * (X - S0 * np.exp(-q * T))

    return MCResult(float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(len(payoff))), len(Z))
