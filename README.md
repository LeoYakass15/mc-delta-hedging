# Monte Carlo pricing & discrete delta hedging

Pricer Monte Carlo pour options européennes, simulation de couverture delta discrète
et analyse de l'erreur de couverture en fonction de la volatilité et de la fréquence
de rebalancement.

## Structure

```
mchedge/
  black_scholes.py   formules fermées (prix, delta, gamma, vega) — référence
  monte_carlo.py     pricer MC : GBM exact, antithétiques, variable de contrôle, IC 95 %
  hedging.py         simulation de couverture delta discrète (vectorisée sur les trajectoires)
  analysis.py        études paramétriques (grilles σ × N, vol mal spécifiée, coûts)
  plots.py           graphiques matplotlib
run_analysis.py      lance toutes les études -> results/ (PNG + CSV)
tests/               tests pytest (parité call-put, IC du MC, loi en 1/√N, ...)
```

## Installation et lancement

```bash
pip install -r requirements.txt
python run_analysis.py              # ~25 s ; options : --paths, --sigma, --tc, --option put, ...
python -m pytest -q
```

Exemple d'utilisation directe :

```python
from mchedge import mc_european_price, simulate_delta_hedge

print(mc_european_price(S0=100, K=100, T=1, r=0.02, sigma=0.2, n_paths=500_000, seed=0))
res = simulate_delta_hedge(sigma_real=0.2, n_rebal=52, n_paths=20_000, seed=0)
print(res.summary())
```

## Méthodologie

**Pricer.** Sous la mesure risque-neutre, `S_T = S0 exp((r − q − σ²/2)T + σ√T Z)` est
simulé exactement. Deux réductions de variance : antithétiques (`Z, −Z`) et variable de
contrôle `X = e^{−rT} S_T` (espérance connue `S0 e^{−qT}`). Le β de contrôle est estimé
sur les moyennes de paires antithétiques — l'estimer sur les tirages individuels rend la
combinaison moins efficace que le contrôle seul.

**Couverture.** Le trader vend l'option au prix BS(σ_imp), achète Δ actions et place le
reste au taux r. À chacune des N dates équidistantes il rebalance au delta BS(σ_imp). Le
sous-jacent suit un GBM de vol réelle σ_réel. Le P&L final actualisé est

`P&L = e^{−rT} [ cash_T + Δ_{T−} S_T − payoff(S_T) ]`.

Coûts de transaction proportionnels `tc · |ΔΔ| · S` en option.

**Théorie de référence.**
- Vol correcte : P&L d'espérance ≈ 0 et d'écart-type
  `σ_PnL ≈ √(π/4) · vega · σ / √N` (Derman & Kamal, 1999).
- Vol mal spécifiée : `E[P&L] ≈ BS(σ_imp) − BS(σ_réel)`, soit
  `½ ∫ Γ S² (σ_imp² − σ_réel²) dt` avec gamma à σ_imp.

## Résultats (ATM, S0 = K = 100, T = 1, r = 2 %, 20 000 trajectoires)

| Étude | Constat |
|---|---|
| Pricer MC | 500k trajectoires : 8,909 ± 0,008 vs BS 8,916. Réduction de variance ×1,7 (antithétique), ×5,4 (contrôle), ×23 (les deux). |
| Erreur vs N | Pente log-log −0,487 pour toutes les vols (théorie −0,5). Simulé / Derman–Kamal ≈ 0,98 (0,92 à N = 4, où l'asymptotique est grossière). |
| Erreur vs σ | Écart-type absolu ≈ linéaire en σ (vega ATM quasi constant). En **% de la prime** il dépend presque uniquement de N : ~10 % à N = 52, ~5 % à N = 252, ~2,5 % à N = 1008. |
| Distribution | P&L centré, queues épaisses à faible N (le gamma concentre l'erreur près du strike à l'approche de la maturité). |
| σ_réel ≠ σ_imp | Biais simulé = BS(σ_imp) − BS(σ_réel) à 0,01 près ; vendre à 20 % une vol réalisée de 30 % coûte ≈ 3,9. La dispersion augmente aussi avec l'écart de vol. |
| Coûts (10 pb) | Risque ↓ en 1/√N, coûts ↑ ≈ √N : le compromis coûts + écart-type atteint son minimum vers N ≈ 250 (quotidien). Au-delà, les coûts dominent et ajoutent même de la variance. |

Les graphiques sont dans `results/` (`01_…` à `07_…`), les tableaux en CSV.

## Pistes d'extension

Vol stochastique (Heston) pour la vol réalisée, couverture à bandes (Whalley–Wilmott)
plutôt qu'à dates fixes, delta de Leland avec vol ajustée des coûts, options barrières.
