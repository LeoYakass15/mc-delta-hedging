"""Matplotlib charts for the studies (colour-blind-safe palette, one y-axis per chart)."""
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
    ax.loglog(n, ref, ":", color=INK_2, lw=1.2, label=r"slope $-1/2$")
    ax.set(xlabel="Number of paths", ylabel="Standard error of the price",
           title="Monte Carlo pricer convergence")
    ax.legend()
    _save(fig, path)


def plot_mc_vs_bs(df: pd.DataFrame, path):
    """MC - BS gap in number of standard errors (z), by strike, vol and maturity."""
    opts = list(df["option"].unique())
    sigmas, mats = sorted(df["sigma"].unique()), sorted(df["T"].unique())
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, len(opts), figsize=(10, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 * np.diff(sorted(df["K"].unique())).min()
    n_groups = len(sigmas) * len(mats)
    for ax, opt in zip(axes, opts):
        ax.axhspan(-1.96, 1.96, color=SERIES[0], alpha=0.08, lw=0)
        for y in (-1.96, 1.96):
            ax.axhline(y, color=INK_2, lw=1, ls="--")
        ax.axhline(0, color=INK_2, lw=1, ls=":")
        g_idx = 0
        for mi, T in enumerate(mats):
            for si, sig in enumerate(sigmas):
                g = df[(df["option"] == opt) & (df["T"] == T) & (df["sigma"] == sig)]
                off = (g_idx - (n_groups - 1) / 2) * width / n_groups
                ax.plot(g["K"] + off, g["z"].clip(-4, 4), markers[mi], color=SERIES[si], ms=7,
                        mec=SURFACE, mew=1.2, ls="none")
                g_idx += 1
        ax.set(xlabel="Strike K (S0 = 100)", title=f"{opt.capitalize()}s", ylim=(-4, 4))
        ax.set_xticks(sorted(df["K"].unique()))
    axes[0].set_ylabel("(MC price − BS price) / std error")
    handles = [plt.Line2D([], [], marker="o", color=SERIES[i], ls="none", ms=7, label=f"σ = {s:.0%}")
               for i, s in enumerate(sigmas)]
    handles += [plt.Line2D([], [], marker=markers[i], color=INK_2, ls="none", ms=7,
                           label=f"T = {t:g} yr") for i, t in enumerate(mats)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=SERIES[0], alpha=0.15, label="95% CI (|z| < 1.96)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    fig.suptitle(f"Monte Carlo vs Black-Scholes: BS price inside the 95% CI for "
                 f"{df['bs_in_ci95'].sum()} of {len(df)} options",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_std_vs_rebal(grid: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for color, (sig, g) in zip(SERIES, grid.groupby("sigma_real")):
        ax.loglog(g["n_rebal"], g["std_pnl"], "o", color=color, ms=6, label=f"σ = {sig:.0%}")
        ax.loglog(g["n_rebal"], g["derman_kamal_std"], "-", color=color, lw=1.5, alpha=0.8)
    ax.plot([], [], "-", color=INK_2, lw=1.5, label="Derman–Kamal (theory)")
    ax.set(xlabel="Number of rebalancing dates N (log scale)",
           ylabel="Std dev of hedging P&L",
           title="Hedging error vs rebalancing frequency")
    ax.legend(ncol=2, fontsize=9)
    _save(fig, path)


def plot_std_vs_sigma(grid: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for color, (n, g) in zip(SERIES, grid.groupby("n_rebal")):
        g = g.sort_values("sigma_real")
        ax.plot(g["sigma_real"] * 100, g["std_pnl"], "o-", color=color, ms=5, label=f"N = {n}")
    ax.set(xlabel="Volatility (%)", ylabel="Std dev of hedging P&L",
           title="Hedging error vs volatility")
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
    ax.set(xlabel="Number of rebalancing dates N", ylabel="Volatility σ",
           title="Hedging P&L std dev as % of the option premium")
    fig.colorbar(im, ax=ax, label="% of premium")
    _save(fig, path)


def plot_pnl_distributions(results: dict, path):
    """results: {N: HedgeResult}"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lim = max(np.quantile(np.abs(r.pnl), 0.995) for r in results.values())
    bins = np.linspace(-lim, lim, 90)
    for color, (n, res) in zip(SERIES, results.items()):
        ax.hist(res.pnl, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"N = {n} (std = {res.std:.2f})")
    ax.axvline(0, color=INK_2, lw=1, ls=":")
    ax.set(xlabel="Discounted hedging P&L", ylabel="Density",
           title="Distribution of the hedging P&L")
    ax.legend()
    _save(fig, path)


def plot_vol_mismatch(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = df["sigma_real"] * 100
    ax.fill_between(x, df["mean_pnl"] - df["std_pnl"], df["mean_pnl"] + df["std_pnl"],
                    color=SERIES[0], alpha=0.15, lw=0, label="± 1 std dev")
    ax.plot(x, df["mean_pnl"], "o", color=SERIES[0], ms=6, label="Simulated mean P&L")
    ax.plot(x, df["theory_mean"], "-", color=SERIES[1], lw=2,
            label=r"BS($\sigma_{impl}$) − BS($\sigma_{real}$)")
    ax.axhline(0, color=INK_2, lw=1, ls=":")
    ax.axvline(df["sigma_implied"].iloc[0] * 100, color=INK_2, lw=1, ls="--")
    ax.set(xlabel="Realised volatility (%)", ylabel="Hedging P&L (option seller)",
           title=f"Realised vol ≠ implied vol ({df['sigma_implied'].iloc[0]:.0%})")
    ax.legend()
    _save(fig, path)


def plot_tc_tradeoff(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogx(df["n_rebal"], df["std_pnl"], "o-", color=SERIES[0], ms=5, label="P&L std dev (risk)")
    ax.semilogx(df["n_rebal"], df["mean_costs"], "o-", color=SERIES[1], ms=5, label="Mean costs")
    ax.semilogx(df["n_rebal"], df["objective"], "o-", color=SERIES[2], ms=5, label="Costs + risk")
    best = df.loc[df["objective"].idxmin()]
    ax.annotate(f"optimum ≈ N = {int(best['n_rebal'])}", (best["n_rebal"], best["objective"]),
                xytext=(0, 14), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set(xlabel="Number of rebalancing dates N", ylabel="Amount (same unit as premium)",
           title="Risk / transaction cost trade-off")
    ax.legend()
    _save(fig, path)
