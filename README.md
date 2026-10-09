# From a Log-Correlated Gaussian Field to the Inverse Cubic Law: Multifractal Volatility in Financial Returns

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication code for the paper:

> Carvalho, J.L.P., Lima, L.S. (2026). From a Log-Correlated Gaussian Field to the Inverse Cubic Law: Multifractal Volatility in Financial Returns. *Physica A: Statistical Mechanics and its Applications*.

## Overview

This repository implements a discrete-time return model in which Gaussian innovations are modulated by a multifractal stochastic volatility field. The volatility is the exponential of a log-correlated Gaussian field sampled exactly through Cholesky factorization of its covariance matrix, reproducing the volatility cascade of the multifractal random walk (MRW) of Bacry, Delour and Muzy (2001). Persistence resides entirely in the volatility; the return innovations are standard Gaussian (H = 0.5), so raw returns are near-uncorrelated.

The model reproduces several stylized facts of financial returns:

- Heavy, power-law-like tails with effective CCDF exponents in the range of the inverse cubic law (α ≈ 2.6 to 3.0)
- Volatility clustering with strong finite-horizon persistence (H_DFA ≈ 0.84 for absolute returns)
- Near-random-walk behavior in raw returns (H_DFA ≈ 0.51)
- A nonlinear generalized Hurst spectrum whose multiscaling slope agrees with the MRW analytical prediction

All tail exponents follow the CCDF (survival-function) convention P(|R| > r) ~ r^(−α), which is the standard in the empirical econophysics literature and the convention of the inverse cubic law.

## What the code computes

| Component | Description |
|-----------|-------------|
| **Volatility field** | Log-correlated Gaussian field with covariance λ² ln⁺(L/\|t−s\|), sampled by Cholesky factorization |
| **Return simulation** | Discrete-time model r_t = (μ − ½σ²_t)Δt + σ_t ε_t √Δt with Gaussian innovations |
| **Tail analysis** | Clauset-Shalizi-Newman protocol (CSN) and Hill estimator, both in CCDF convention, with bootstrap CIs and threshold-stability diagnostics |
| **Scaling analysis** | DFA (n_max = T/4) and MF-DFA (n_max = T/10) computed per realization and aggregated with Monte Carlo uncertainty across 20 independent trajectories |
| **MRW benchmark** | Analytical prediction h(q) = 1/2 − λ²(q−1)/2 overlaid on the simulated H(q) spectrum |
| **Intermittency estimation** | λ² estimated from the autocovariance of log\|r\| against log(lag), evaluated separately on each simulated trajectory (mean ± s.d. reported) |
| **Kurtosis benchmark** | Closed-form excess kurtosis 3[(2L)^(4λ²) − 1] of the Gaussian-lognormal mixture, printed next to the simulated value |
| **Calibrated runs** | Simulations at the empirical λ² estimates (0.037 for S&P 500, 0.014 for Ibovespa) |
| **Empirical validation** | Full pipeline applied to S&P 500 and Ibovespa daily data (2000 to 2024) via Yahoo Finance, including rolling-window DFA |
| **Leverage extension** | Pochart-Bouchaud leverage kernel for tuning negative skewness |

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Installation

```bash
git clone https://github.com/Jaolukaz/fsde-financial-modeling.git
cd fsde-financial-modeling

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
python fsde_final.py
```

The script runs end-to-end in a single call to `main()`:

1. Simulates the model at the reference configuration (λ² = 0.10) with 20 realizations
2. Computes per-realization DFA, MF-DFA, and moments, pooled tail statistics with bootstrap CIs, and per-trajectory λ² estimates
3. Runs the sensitivity analysis across λ² ∈ [0.02, 0.15]
4. Runs the leverage sweep (Pochart-Bouchaud β from 0 to 2)
5. Simulates at the empirical λ² values (0.037 and 0.014)
6. Downloads empirical data from Yahoo Finance and applies the full pipeline
7. Prints a structured report and saves the figures

Execution time is approximately 25 to 40 minutes depending on hardware. For a quick test:

```python
from fsde_final import main
results = main(n_realizations=5, n_boot=50, make_figures=False,
               fetch_empirical=False, run_lambda_experiments=False)
```

## Data

Empirical data are downloaded automatically from Yahoo Finance:

- **S&P 500** (`^GSPC`): 2000 to 2024
- **Ibovespa** (`^BVSP`): 2000 to 2024

No manual data download is required. If Yahoo Finance cannot be reached, the empirical block is skipped and the report says so.

## Key parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Return Hurst exponent | H_B | 0.50 | Fixed at 1/2 (Gaussian innovations, no fractional noise) |
| Intermittency | λ² | 0.10 | Controls tail thickness; the effective tail exponent q* ≈ √(2/λ²) of the idealised cascade is a heuristic guide to its order of magnitude |
| Baseline volatility | σ₀ | 0.012 | Sets the volatility level |
| Integral scale | L | 252 | Longest correlation horizon (trading days); covariance = 0 for lags ≥ L |
| Path length | T | 2520 | Trading days per realization (≈ 10 years) |
| Realizations | n | 20 | Independent trajectories for pooling and per-realization analysis |
| Random seed | SEED | 42 | Fixes all simulations so that the reported numbers are reproducible |

## Output

The script produces:

- **Console report** with all numerical results (tail exponents, DFA/MF-DFA per-realization statistics, bootstrap CIs, MRW benchmark, closed-form kurtosis, per-trajectory λ² estimates, calibrated-λ² experiments, leverage sweep)
- **Figures** saved in `figures/` (PNG, 150 dpi): simulated series, CCDFs, Hill plots, DFA, MF-DFA with MRW overlay, sensitivity curves, rolling-window Hurst exponents, and empirical counterparts

## Methodological notes

- **Exponent convention.** All tail exponents are reported in the CCDF convention. The CSN routine estimates the density exponent internally and converts it as α = α_pdf − 1; the Hill estimator estimates the CCDF exponent directly. The two agree to within 0.01 on the simulated data.
- **Effective tail exponent.** `theoretical_alpha` returns q* = √(2/λ²), the effective tail exponent that multifractal extreme-value theory predicts for the idealised lognormal cascade (Muzy, Bacry and Kozhemyak 2006), in the CCDF convention. It is not a moment-divergence order. On the regularised daily grid all moments of the returns are finite, so the fitted exponents are effective slopes of lognormal-mixture tails over the sampled range.
- **Per-realization scaling and λ² estimation.** DFA, MF-DFA and the λ² estimator use lagged dependence, so they are evaluated on each trajectory separately and never on concatenated independent paths. Pooling is used only for marginal and tail statistics.
- **Approximate symmetry.** The return innovations and the volatility field are independent, so the baseline model has no leverage effect. The drift term −½σ²_t Δt induces a small, numerically negligible skewness, so the distribution is approximately rather than exactly symmetric.
- **Discrete-time formulation.** The model is implemented and analysed as a discrete-time return equation. The continuous-time Itô SDE serves as conceptual motivation but is not the operational model.
- **Finite-horizon persistence.** The covariance of the volatility field is exactly zero for lags ≥ L = 252. The resulting persistence is of finite horizon, not of genuine asymptotic long-memory type.
- **Leverage sweep.** The persistence check reported for each β is a DFA on the pooled absolute returns, used only to compare β values with one another.

## Citation

```bibtex
@article{carvalho2026multifractal,
  title={From a Log-Correlated Gaussian Field to the Inverse Cubic Law:
         Multifractal Volatility in Financial Returns},
  author={Carvalho, Jo{\~a}o Lucas de Pinho and Lima, Leonardo dos Santos},
  journal={Physica A: Statistical Mechanics and its Applications},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- **João Lucas de Pinho Carvalho**, Department of Mathematics, CEFET-MG
- **Leonardo dos Santos Lima**, Department of Physics, CEFET-MG

## Acknowledgments

This work is part of the doctoral thesis *Two Essays on Advanced Approaches in Econophysics* developed at the Federal Center for Technological Education of Minas Gerais (CEFET-MG), Brazil, with support from CAPES.
