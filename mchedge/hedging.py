"""Simulation of discrete delta hedging of a short European option.

The trader sells the option at the Black-Scholes price computed with the implied
vol `sigma_implied`, then delta-hedges (BS deltas with the same vol) at `n_rebal`
equally spaced dates. The underlying follows a GBM with realised vol `sigma_real`
(real-world drift `mu`, r by default).

Final P&L (discounted to t=0) = value of the hedge portfolio - option payoff.
- If sigma_real = sigma_implied, the mean is ~0 and the standard deviation decays
  as O(1/sqrt(N)): this is the discrete hedging error (Derman & Kamal, 1999).
- If sigma_real != sigma_implied, a bias ~ 0.5 * ∫ Γ S² (σ_imp² - σ_real²) dt appears.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_delta, bs_price, bs_vega


@dataclass
class HedgeResult:
    pnl: np.ndarray          # discounted P&L per path
    premium: float           # premium received at t=0
    n_rebal: int
    sigma_real: float
    sigma_implied: float
    total_costs: np.ndarray  # discounted transaction costs per path

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))

    def var(self, level: float = 0.95) -> float:
        """Value-at-Risk (as a positive loss) at the given level."""
        return float(-np.quantile(self.pnl, 1 - level))

    def cvar(self, level: float = 0.95) -> float:
        """Expected shortfall (average loss beyond the VaR)."""
        cutoff = np.quantile(self.pnl, 1 - level)
        return float(-self.pnl[self.pnl <= cutoff].mean())

    def summary(self) -> dict:
        return {
            "n_rebal": self.n_rebal,
            "sigma_real": self.sigma_real,
            "sigma_implied": self.sigma_implied,
            "premium": self.premium,
            "mean_pnl": self.mean,
            "std_pnl": self.std,
            "rel_std_pnl": self.std / self.premium,
            "VaR95": self.var(0.95),
            "CVaR95": self.cvar(0.95),
            "mean_costs": float(self.total_costs.mean()),
        }


def simulate_delta_hedge(S0=100.0, K=100.0, T=1.0, r=0.02, sigma_real=0.2, sigma_implied=None,
                         n_rebal=52, n_paths=20_000, option="call", q=0.0, mu=None,
                         tc=0.0, seed=None) -> HedgeResult:
    """Simulate discrete delta hedging of a short option.

    Parameters
    ----------
    sigma_real    : realised vol of the underlying (drives the paths).
    sigma_implied : vol used for the sale price and the deltas (default = sigma_real).
    n_rebal       : number of rebalancing dates over [0, T] (the first one at t=0).
    mu            : real-world drift (default = r, i.e. risk-neutral measure).
    tc            : proportional transaction cost (e.g. 0.001 = 10 bp of traded notional).
    """
    sigma_implied = sigma_real if sigma_implied is None else sigma_implied
    mu = r if mu is None else mu
    rng = np.random.default_rng(seed)
    dt = T / n_rebal
    growth = np.exp(r * dt)

    S = np.full(n_paths, float(S0))
    premium = float(bs_price(S0, K, T, r, sigma_implied, option, q))
    delta = np.full(n_paths, float(bs_delta(S0, K, T, r, sigma_implied, option, q)))
    costs = tc * np.abs(delta) * S
    cash = premium - delta * S - costs

    for i in range(1, n_rebal + 1):
        Z = rng.standard_normal(n_paths)
        S_new = S * np.exp((mu - q - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * Z)
        # carry: interest on cash + dividends on the shares held
        cash = cash * growth + delta * S * (np.exp(q * dt) - 1.0)
        S = S_new
        if i < n_rebal:
            new_delta = bs_delta(S, K, T - i * dt, r, sigma_implied, option, q)
            trade_cost = tc * np.abs(new_delta - delta) * S
            cash -= (new_delta - delta) * S + trade_cost
            costs = costs * growth + trade_cost
            delta = new_delta
        else:
            costs = costs * growth

    payoff = np.maximum(S - K, 0.0) if option == "call" else np.maximum(K - S, 0.0)
    disc = np.exp(-r * T)
    pnl = disc * (cash + delta * S - payoff)
    return HedgeResult(pnl, premium, n_rebal, sigma_real, sigma_implied, disc * costs)


def derman_kamal_std(S0, K, T, r, sigma, n_rebal, q=0.0) -> float:
    """Approximate standard deviation of the hedging error (correct vol):
    σ_PnL ≈ sqrt(π/4) · vega · σ / sqrt(N)   (Derman & Kamal, 1999)."""
    return float(np.sqrt(np.pi / 4) * bs_vega(S0, K, T, r, sigma, q) * sigma / np.sqrt(n_rebal))
