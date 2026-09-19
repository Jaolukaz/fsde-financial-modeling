# From a Log-Correlated Gaussian Field to the Inverse Cubic Law: Multifractal Volatility in Financial Returns

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication code for the paper:

> Carvalho, J.L.P., Lima, L.S. (2026). From a Log-Correlated Gaussian Field to the Inverse Cubic Law: Multifractal Volatility in Financial Returns. *Physica A: Statistical Mechanics and its Applications*.

## Overview

This repository implements a discrete-time return model in which Gaussian innovations are modulated by a multifractal stochastic volatility field. The volatility is the exponential of a log-correlated Gaussian field sampled exactly through Cholesky factorization of its covariance matrix, reproducing the volatility cascade of the multifractal random walk (MRW) of Bacry, Delour and Muzy (2001). Persistence resides entirely in the volatility; the return innovations are standard Gaussian (H = 0.5), so raw returns are near-uncorrelated.

The model reproduces several stylized facts of financial returns:

- Heavy-tailed distributions consistent with the inverse cubic law (CCDF exponent α ≈ 2.6–3.0)
- Volatility clustering with strong finite-horizon persistence (H_DFA ≈ 0.84)
- Near-random-walk behavior in raw returns (H_DFA ≈ 0.50)
- Nonlinear generalized Hurst spectrum consistent with the MRW analytical prediction

All tail exponents follow the CCDF (survival-function) convention P(|R| > r) ~ r^{−α}, which is the standard in the empirical econophysics literature and the convention of the inverse cubic law.

## What the code computes

| Component | Description |
|-----------|-------------|
| **Volatility field** | Log-correlated Gaussian field with covariance λ² ln⁺(L/\|t−s\|), sampled by Cholesky factorization |
| **Return simulation** | Discrete-time model r_t = (μ − ½σ²_t)Δt + σ_t ε_t √Δt with Gaussian innovations |
| **Tail analysis** | Clauset-Shalizi-Newman protocol (CSN) and Hill estimator, both in CCDF convention, with bootstrap CIs and threshold-stability diagnostics |
| **Scaling analysis** | DFA and MF-DFA computed per realization (n_max = T/4), aggregated with Monte Carlo uncertainty across 20 independent trajectories |
| **MRW benchmark** | Analytical prediction h(q) = 1/2 − λ²(q−1)/2 overlaid on the simulated H(q) spectrum |
| **Calibrated runs** | Simulations at empirical λ² estimates (0.037 for S&P 500, 0.014 for Ibovespa) |
| **Empirical validation** | Full pipeline applied to S&P 500 and Ibovespa daily data (2000–2024) via Yahoo Finance |
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
python fsde_r2.py
```

The script runs end-to-end in a single call to `main()`:

1. Simulates the model at the reference configuration (λ² = 0.10) with 20 realizations
2. Computes per-realization DFA, MF-DFA, and tail statistics with bootstrap CIs
3. Runs sensitivity analysis across λ² ∈ [0.02, 0.15]
4. Runs the leverage sweep (Pochart-Bouchaud β = 0 to 2)
5. Simulates at empirical λ² values (0.037 and 0.014) with corrected methodology
6. Downloads empirical data from Yahoo Finance and applies the full pipeline
7. Prints a structured report and saves publication-quality figures

Execution time is approximately 25–40 minutes depending on hardware. For a quick test:

```python
from fsde_r2 import main
results = main(n_realizations=5, n_boot=50, make_figures=False,
               fetch_empirical=False, run_lambda_experiments=False)
```

## Data

Empirical data are downloaded automatically from Yahoo Finance:

- **S&P 500** (`^GSPC`): 2000–2024
- **Ibovespa** (`^BVSP`): 2000–2024

No manual data download is required.

## Key parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Return Hurst exponent | H_B | 0.50 | Fixed at 1/2 (Gaussian innovations, no fractional noise) |
| Intermittency | λ² | 0.10 | Controls tail thickness; α_CCDF ≈ √(2/λ²) − 1 |
| Baseline volatility | σ₀ | 0.012 | Sets the volatility level |
| Integral scale | L | 252 | Longest correlation horizon (trading days); covariance = 0 for lags ≥ L |
| Path length | T | 2520 | Trading days per realization (≈ 10 years) |
| Realizations | n | 20 | Independent trajectories for pooling and per-realization analysis |

## Output

The script produces:

- **Console report** with all numerical results (tail exponents, DFA/MF-DFA per-realization statistics, bootstrap CIs, MRW benchmark comparison, calibrated-λ² experiment results)
- **Figures** saved in `figures/` (PNG, 300 dpi): simulated series, CCDFs, Hill plots, DFA, MF-DFA with MRW overlay, sensitivity curves, rolling-window Hurst, and empirical counterparts
- **Summary table** comparing empirical data with simulated and calibrated model configurations

## Methodological notes

The following methodological corrections were implemented in the R2 revision:

- **Exponent convention.** The R1 code mixed density (CSN) and survival (Hill) exponent conventions, producing an apparent CSN-Hill gap of exactly 1.0 that was incorrectly interpreted as evidence of non-power-law behavior. All exponents are now reported in CCDF convention, and the two estimators agree to within 0.01.
- **Per-realization scaling.** DFA and MF-DFA were previously computed on concatenated trajectories, allowing fitting windows to span boundaries between independent realizations. They are now computed per realization with n_max = T/4, and aggregated with mean ± std across realizations.
- **Discrete-time formulation.** The model is implemented and analysed as a discrete-time return equation. The continuous-time Itô SDE serves as conceptual motivation but is not the operational model.
- **Finite-horizon persistence.** The covariance of the volatility field is exactly zero for lags ≥ L = 252. The resulting persistence is of finite horizon, not of genuine asymptotic long-memory type.

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

- **João Lucas de Pinho Carvalho** — Department of Mathematics, CEFET-MG
- **Leonardo dos Santos Lima** — Department of Physics, CEFET-MG

## Acknowledgments

This work is part of the doctoral thesis *Two Essays on Advanced Approaches in Econophysics* developed at the Federal Center for Technological Education of Minas Gerais (CEFET-MG), Brazil, with support from CAPES.
