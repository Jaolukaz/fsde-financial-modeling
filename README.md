# Modeling Stock Time Evolution Using a Fractional Stochastic Differential Equation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication code for the paper:

> Carvalho, J.L.P., Lima, L.S. (2026). Modeling Stock Time Evolution Using a Fractional Stochastic Differential Equation. *Journal of Financial Stability*.

## Overview

This repository contains the Python implementation of a Fractional Stochastic Differential Equation (FSDE) framework that combines fractional Brownian motion with multifractal stochastic volatility to model stock price dynamics. The model reproduces several stylized facts of financial markets:

- Heavy-tailed return distributions consistent with the inverse cubic law (α ≈ 3)
- Volatility clustering across multiple time scales
- Long-range dependence in absolute returns (H ≈ 0.85)
- Near-random-walk behavior in raw returns (H ≈ 0.50)

## Implementation Phases

The code is organized into six phases:

| Phase | Description |
|-------|-------------|
| 1 | Fractional Brownian Motion generation via Cholesky decomposition |
| 2 | Multifractal stochastic volatility via superposed Ornstein-Uhlenbeck processes |
| 3 | Complete FSDE simulation |
| 4 | Tail distribution analysis (CCDF, Hill estimator) |
| 5 | Detrended Fluctuation Analysis for Hurst exponent estimation |
| 6 | Empirical validation against S&P 500 and Ibovespa |

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/fsde-financial-modeling.git
cd fsde-financial-modeling

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
python fsde.py
```

The script will:
1. Generate synthetic FSDE trajectories with specified parameters
2. Perform tail analysis and DFA on simulated data
3. Download empirical data from Yahoo Finance (S&P 500 and Ibovespa, 2000-2024)
4. Validate model predictions against empirical stylized facts
5. Generate publication-quality figures

## Data

Empirical data are downloaded automatically from Yahoo Finance:
- **S&P 500** (`^GSPC`): 1999-2024
- **Ibovespa** (`^BVSP`): 2000-2024

No manual data download is required.

## Key Parameters

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| Hurst exponent | H | 0.50-0.65 | Controls long-range dependence in fBm |
| Intermittency | λ² | 0.04-0.12 | Controls tail thickness (α ≈ 1/λ²) |
| Integral scale | T | 252 | Longest correlation time scale (trading days) |
| O-U components | K | 10 | Number of superposed processes |

## Output

The script produces:
- Console output with statistical test results
- Figures saved in the working directory (PNG format, 300 dpi)
- Summary tables comparing empirical data with model predictions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{carvalho2026fsde,
  title={Modeling Stock Time Evolution Using a Fractional Stochastic Differential Equation},
  author={Carvalho, Jo{\~a}o Lucas de Pinho and Lima, Leonardo dos Santos},
  journal={Journal of Financial Stability},
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

This work is part of the doctoral thesis "Two Essays on Advanced Approaches in Econophysics" developed at the Federal Center for Technological Education of Minas Gerais (CEFET-MG), Brazil.
