"""Pricer Monte Carlo pour options européennes sous Black-Scholes (GBM).

- Simulation exacte du GBM (pas d'erreur de discrétisation pour un payoff terminal).
- Réduction de variance : variables antithétiques et/ou variable de contrôle (sous-jacent actualisé).
- Retourne le prix, l'erreur standard et l'intervalle de confiance à 95 %.
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
        return f"{self.price:.4f} ± {1.96 * self.std_error:.4f} (IC95 [{lo:.4f}, {hi:.4f}], n={self.n_paths})"


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0, rng=None, antithetic=False):
    """Trajectoires GBM sur une grille uniforme, forme (n_paths, n_steps + 1).

    Schéma exact : S_{t+dt} = S_t * exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z).
    Avec antithetic=True, la seconde moitié des trajectoires utilise -Z.
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
    """Prix Monte Carlo d'une option européenne.

    Variable de contrôle : X = e^{-rT} S_T, d'espérance connue S0 e^{-qT}.
    Avec les antithétiques, l'erreur standard est calculée sur les moyennes de paires
    (les deux tirages d'une paire ne sont pas indépendants).
    """
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
    if antithetic:  # échantillons i.i.d. = moyennes de paires
        m = len(Z) // 2
        payoff, X = 0.5 * (payoff[:m] + payoff[m:]), 0.5 * (X[:m] + X[m:])

    if control_variate:
        # β estimé sur les mêmes échantillons i.i.d. (après appariement), sinon la
        # combinaison antithétique + contrôle est sous-optimale.
        cov = np.cov(payoff, X)
        beta = cov[0, 1] / cov[1, 1]
        payoff = payoff - beta * (X - S0 * np.exp(-q * T))

    return MCResult(float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(len(payoff))), len(Z))
