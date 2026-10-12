"""Simulation de couverture delta discrète d'une option européenne vendue.

Le trader vend l'option au prix Black-Scholes calculé avec la vol implicite
`sigma_implied`, puis se couvre en delta (deltas BS avec la même vol) à
`n_rebal` dates équidistantes. Le sous-jacent suit un GBM de vol réelle
`sigma_real` (dérive réelle `mu`, par défaut r).

P&L final (actualisé en t=0) = valeur du portefeuille de couverture - payoff.
- Si sigma_real = sigma_implied, l'espérance est ~0 et l'écart-type décroît en
  O(1/sqrt(N)) : c'est l'erreur de couverture discrète (Derman & Kamal, 1999).
- Si sigma_real != sigma_implied, apparaît en plus un biais ~ 0.5 * ∫ Γ S² (σ_imp² - σ_real²) dt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_delta, bs_price, bs_vega


@dataclass
class HedgeResult:
    pnl: np.ndarray          # P&L actualisé par trajectoire
    premium: float           # prime reçue en t=0
    n_rebal: int
    sigma_real: float
    sigma_implied: float
    total_costs: np.ndarray  # coûts de transaction actualisés par trajectoire

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))

    def var(self, level: float = 0.95) -> float:
        """Value-at-Risk (perte positive) au niveau donné."""
        return float(-np.quantile(self.pnl, 1 - level))

    def cvar(self, level: float = 0.95) -> float:
        """Expected shortfall (perte moyenne au-delà de la VaR)."""
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
    """Simule la couverture delta discrète d'une option vendue.

    Paramètres
    ----------
    sigma_real    : vol réelle du sous-jacent (génère les trajectoires).
    sigma_implied : vol utilisée pour le prix de vente et les deltas (défaut = sigma_real).
    n_rebal       : nombre de rebalancements sur [0, T] (le premier en t=0).
    mu            : dérive réelle (défaut = r, i.e. mesure risque-neutre).
    tc            : coût de transaction proportionnel (ex. 0.001 = 10 pb du nominal échangé).
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
        # portage : intérêts sur le cash + dividendes sur les actions détenues
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
    """Approximation de l'écart-type de l'erreur de couverture (vol correcte) :
    σ_PnL ≈ sqrt(π/4) · vega · σ / sqrt(N)   (Derman & Kamal, 1999)."""
    return float(np.sqrt(np.pi / 4) * bs_vega(S0, K, T, r, sigma, q) * sigma / np.sqrt(n_rebal))
