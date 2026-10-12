"""Tests (pytest). Lancer : python -m pytest -q"""
import numpy as np

from mchedge import (bs_delta, bs_price, derman_kamal_std, mc_european_price,
                     simulate_delta_hedge, simulate_gbm_paths)

P = dict(S0=100.0, K=105.0, T=1.0, r=0.03, sigma=0.25)


def test_put_call_parity():
    c = bs_price(P["S0"], P["K"], P["T"], P["r"], P["sigma"], "call")
    p = bs_price(P["S0"], P["K"], P["T"], P["r"], P["sigma"], "put")
    assert np.isclose(c - p, P["S0"] - P["K"] * np.exp(-P["r"] * P["T"]))


def test_delta_matches_finite_difference():
    h = 1e-4
    up = bs_price(P["S0"] + h, P["K"], P["T"], P["r"], P["sigma"])
    dn = bs_price(P["S0"] - h, P["K"], P["T"], P["r"], P["sigma"])
    assert np.isclose(bs_delta(P["S0"], P["K"], P["T"], P["r"], P["sigma"]), (up - dn) / (2 * h), atol=1e-6)


def test_mc_price_within_confidence_interval():
    for option in ("call", "put"):
        for kw in (dict(antithetic=False, control_variate=False), dict()):
            ref = bs_price(P["S0"], P["K"], P["T"], P["r"], P["sigma"], option)
            res = mc_european_price(**P, option=option, n_paths=200_000, seed=1, **kw)
            assert abs(res.price - ref) < 4 * res.std_error


def test_variance_reduction_helps():
    plain = mc_european_price(**P, n_paths=100_000, antithetic=False, control_variate=False, seed=2)
    best = mc_european_price(**P, n_paths=100_000, seed=2)
    assert best.std_error < 0.5 * plain.std_error


def test_gbm_paths_martingale():
    paths = simulate_gbm_paths(100.0, 0.03, 0.25, 1.0, 50, 100_000, rng=3, antithetic=True)
    assert paths.shape == (100_000, 51)
    assert np.isclose(np.exp(-0.03) * paths[:, -1].mean(), 100.0, rtol=5e-3)


def test_hedge_unbiased_and_std_close_to_theory():
    res = simulate_delta_hedge(sigma_real=0.2, n_rebal=252, n_paths=20_000, seed=4)
    assert abs(res.mean) < 4 * res.std / np.sqrt(20_000) + 0.02
    th = derman_kamal_std(100.0, 100.0, 1.0, 0.02, 0.2, 252)
    assert 0.85 < res.std / th < 1.15


def test_hedge_error_scales_as_inverse_sqrt_n():
    s1 = simulate_delta_hedge(n_rebal=50, n_paths=20_000, seed=5).std
    s2 = simulate_delta_hedge(n_rebal=800, n_paths=20_000, seed=6).std
    assert 3.4 < s1 / s2 < 4.6  # théorie : sqrt(16) = 4


def test_vol_mismatch_sign():
    # vendre à 20 % quand la vol réalisée est 30 % fait perdre de l'argent
    res = simulate_delta_hedge(sigma_real=0.30, sigma_implied=0.20, n_rebal=252, n_paths=10_000, seed=7)
    expected = bs_price(100, 100, 1, 0.02, 0.2) - bs_price(100, 100, 1, 0.02, 0.3)
    assert res.mean < 0
    assert abs(res.mean - expected) < 0.15 * abs(expected)


def test_transaction_costs_reduce_pnl():
    a = simulate_delta_hedge(n_rebal=52, n_paths=5_000, seed=8)
    b = simulate_delta_hedge(n_rebal=52, n_paths=5_000, seed=8, tc=0.002)
    assert b.mean < a.mean
    assert np.isclose(a.mean - b.mean, b.total_costs.mean(), rtol=1e-8)


def test_put_hedge_works():
    res = simulate_delta_hedge(option="put", n_rebal=252, n_paths=10_000, seed=9)
    assert res.std < 0.2 * res.premium
