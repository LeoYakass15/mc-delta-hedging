"""Graphiques matplotlib des études (palette validée daltonisme, un seul axe y par graphique)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "font.size": 10,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mc_convergence(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for color, (name, g) in zip(SERIES, df.groupby("method", sort=False)):
        ax.loglog(g["n_paths"], g["std_error"], "o-", color=color, ms=5, label=name)
    n = df["n_paths"].unique()
    ref = df[df["method"] == "standard"]["std_error"].iloc[0] * np.sqrt(n[0] / n)
    ax.loglog(n, ref, ":", color=INK_2, lw=1.2, label=r"pente $-1/2$")
    ax.set(xlabel="Nombre de trajectoires", ylabel="Erreur standard du prix",
           title="Convergence du pricer Monte Carlo")
    ax.legend()
    _save(fig, path)


def plot_std_vs_rebal(grid: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for color, (sig, g) in zip(SERIES, grid.groupby("sigma_real")):
        ax.loglog(g["n_rebal"], g["std_pnl"], "o", color=color, ms=6, label=f"σ = {sig:.0%}")
        ax.loglog(g["n_rebal"], g["derman_kamal_std"], "-", color=color, lw=1.5, alpha=0.8)
    ax.plot([], [], "-", color=INK_2, lw=1.5, label="Derman–Kamal (théorie)")
    ax.set(xlabel="Nombre de rebalancements N (échelle log)",
           ylabel="Écart-type du P&L de couverture",
           title="Erreur de couverture vs fréquence de rebalancement")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, path)


def plot_std_vs_sigma(grid: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for color, (n, g) in zip(SERIES, grid.groupby("n_rebal")):
        g = g.sort_values("sigma_real")
        ax.plot(g["sigma_real"] * 100, g["std_pnl"], "o-", color=color, ms=5, label=f"N = {n}")
    ax.set(xlabel="Volatilité (%)", ylabel="Écart-type du P&L de couverture",
           title="Erreur de couverture vs volatilité")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, path)


def plot_heatmap(grid: pd.DataFrame, path, value="rel_std_pnl"):
    piv = grid.pivot(index="sigma_real", columns="n_rebal", values=value)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(piv.values * 100, cmap=SEQ_BLUE, aspect="auto", origin="lower")
    ax.grid(False)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), [f"{s:.0%}" for s in piv.index])
    vmax = np.nanmax(piv.values * 100)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else INK)
    ax.set(xlabel="Nombre de rebalancements N", ylabel="Volatilité σ",
           title="Écart-type du P&L en % de la prime")
    fig.colorbar(im, ax=ax, label="% de la prime")
    _save(fig, path)


def plot_pnl_distributions(results: dict, path):
    """results : {N: HedgeResult}"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lim = max(np.quantile(np.abs(r.pnl), 0.995) for r in results.values())
    bins = np.linspace(-lim, lim, 90)
    for color, (n, res) in zip(SERIES, results.items()):
        ax.hist(res.pnl, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"N = {n} (σ = {res.std:.2f})")
    ax.axvline(0, color=INK_2, lw=1, ls=":")
    ax.set(xlabel="P&L de couverture actualisé", ylabel="Densité",
           title="Distribution du P&L de couverture")
    ax.legend()
    _save(fig, path)


def plot_vol_mismatch(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = df["sigma_real"] * 100
    ax.fill_between(x, df["mean_pnl"] - df["std_pnl"], df["mean_pnl"] + df["std_pnl"],
                    color=SERIES[0], alpha=0.15, lw=0, label="± 1 écart-type")
    ax.plot(x, df["mean_pnl"], "o", color=SERIES[0], ms=6, label="P&L moyen simulé")
    ax.plot(x, df["theory_mean"], "-", color=SERIES[1], lw=2,
            label=r"BS($\sigma_{imp}$) − BS($\sigma_{réel}$)")
    ax.axhline(0, color=INK_2, lw=1, ls=":")
    ax.axvline(df["sigma_implied"].iloc[0] * 100, color=INK_2, lw=1, ls="--")
    ax.set(xlabel="Volatilité réalisée (%)", ylabel="P&L de couverture (vendeur)",
           title=f"Vol réalisée ≠ vol implicite ({df['sigma_implied'].iloc[0]:.0%})")
    ax.legend()
    _save(fig, path)


def plot_tc_tradeoff(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogx(df["n_rebal"], df["std_pnl"], "o-", color=SERIES[0], ms=5, label="Écart-type du P&L (risque)")
    ax.semilogx(df["n_rebal"], df["mean_costs"], "o-", color=SERIES[1], ms=5, label="Coûts moyens")
    ax.semilogx(df["n_rebal"], df["objective"], "o-", color=SERIES[2], ms=5, label="Coûts + risque")
    best = df.loc[df["objective"].idxmin()]
    ax.annotate(f"optimum ≈ N = {int(best['n_rebal'])}", (best["n_rebal"], best["objective"]),
                xytext=(0, 14), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set(xlabel="Nombre de rebalancements N", ylabel="Montant (même unité que la prime)",
           title="Arbitrage risque / coûts de transaction")
    ax.legend()
    _save(fig, path)
