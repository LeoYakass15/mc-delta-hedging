"""Formules fermées de Black-Scholes (prix et grecques) pour options européennes.

Servent de référence pour valider le pricer Monte Carlo et de deltas pour la couverture.
Toutes les fonctions sont vectorisées : S, T, sigma peuvent être des tableaux numpy.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, T, sigma = np.asarray(S, float), np.asarray(T, float), np.asarray(sigma, float)
    sqrtT = np.sqrt(np.maximum(T, 1e-16))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return d1, d1 - sigma * sqrtT


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """Prix Black-Scholes d'un call ou put européen (T = 0 -> payoff)."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    payoff = np.maximum(S - K, 0.0) if option == "call" else np.maximum(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if option == "call":
        px = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif option == "put":
        px = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    else:
        raise ValueError("option doit valoir 'call' ou 'put'")
    return np.where(T > 0, px, payoff)


def bs_delta(S, K, T, r, sigma, option="call", q=0.0):
    """Delta Black-Scholes. À maturité, delta = indicatrice d'exercice."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        live, expiry = np.exp(-q * T) * norm.cdf(d1), (S > K).astype(float)
    elif option == "put":
        live, expiry = np.exp(-q * T) * (norm.cdf(d1) - 1.0), -(S < K).astype(float)
    else:
        raise ValueError("option doit valoir 'call' ou 'put'")
    return np.where(T > 0, live, expiry)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega (dérivée par rapport à sigma, pour 1.00 de vol, pas 1%)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
