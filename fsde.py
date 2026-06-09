#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================================
Modeling Stock Time Evolution Using a Fractional Stochastic Differential Equation:
Reproducing Stylized Facts Through Multifractal Volatility
===================================================================================

Supplementary Material (revised version, resubmission to Physica A, PHYSA-261157).

Authors:
    Joao Lucas de Pinho Carvalho
    Department of Mathematics - Federal Center for Technological Education of Minas Gerais (CEFET-MG)
    Belo Horizonte, MG, Brazil

    Leonardo dos Santos Lima
    Department of Physics - Federal Center for Technological Education of Minas Gerais (CEFET-MG)
    Belo Horizonte, MG, Brazil

Reference:
    CARVALHO, J.L.P., LIMA, L.S. (2026) - Modeling Stock Time Evolution Using
    a Fractional Stochastic Differential Equation: Reproducing Stylized Facts
    Through Multifractal Volatility. Physica A.
===================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis, linregress, skew, t as student_t, probplot, shapiro

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - single source of truth for all parameters
# =============================================================================

SEED = 42

# Master parameter configuration. Every phase reads from this object so that the
# numbers in the manuscript come from one documented configuration.
#
#   H_returns : Hurst exponent of the return-driving noise. Fixed at 0.5 so that
#               raw returns are near-uncorrelated, matching the data.
#   lambda_sq : intermittency parameter of the log-correlated volatility field.
#   sigma0    : baseline volatility level.
#   mu        : constant drift (negligible at the daily scale; kept explicit).
#   L         : integral scale of the volatility field, in trading days.
#               L = 252 corresponds to roughly one calendar year.


@dataclass
class FSDEParameters:
    """Parameters of the FSDE model.

    The fractional structure now lives in the volatility field; the return
    innovations are Brownian (H_returns = 0.5). H_returns is retained as an
    explicit field so that the Brownian limit is documented rather than implicit.
    """

    H_returns: float = 0.5
    lambda_sq: float = 0.10
    sigma0: float = 0.012
    mu: float = 0.00005
    L: float = 252.0

    def __post_init__(self):
        assert 0 < self.H_returns < 1, "H_returns must lie in (0,1)"
        assert self.lambda_sq >= 0, "lambda_sq must be non-negative"
        assert self.sigma0 > 0, "sigma0 must be positive"
        assert self.L > 0, "L must be positive"

    @property
    def theoretical_alpha(self) -> float:
        """Tail exponent predicted by the log-normal cascade moment-existence
        threshold: q*^2 ~ 2/lambda^2, hence alpha ~ sqrt(2/lambda^2).

        This replaces the arithmetically untenable alpha = 1/lambda^2 of the
        first submission (which gave alpha in [20,50] for lambda^2 in [0.02,0.05]).
        With lambda^2 = 0.10 this yields alpha ~ 4.5, compatible with the
        simulated regime.
        """
        return float(np.sqrt(2.0 / self.lambda_sq)) if self.lambda_sq > 0 else np.inf


# Master configuration instance used throughout.
CONFIG = FSDEParameters(
    H_returns=0.5,
    lambda_sq=0.10,
    sigma0=0.012,
    mu=0.00005,
    L=252.0,
)


@dataclass
class DFAResult:
    scales: np.ndarray
    fluctuations: np.ndarray
    H: float
    H_se: float
    R2: float
    intercept: float


@dataclass
class MFDFAResult:
    """Generalized Hurst spectrum H(q) from MF-DFA."""

    q_values: np.ndarray
    Hq: np.ndarray
    Hq_se: np.ndarray
    slope_B: float          # slope of H(q) = A + B q (multiscaling proxy)
    slope_B_se: float
    intercept_A: float


@dataclass
class TailResult:
    """Clauset-Shalizi-Newman + Hill tail-fit result."""

    tail: str
    x_min: float
    n_tail: int
    alpha_csn: float
    alpha_csn_ci: Tuple[float, float]
    ks_distance: float
    p_value: float
    hill_alpha: float
    hill_ci: Tuple[float, float]
    hill_k: int
    # arrays for plotting
    x_ccdf: np.ndarray = field(default_factory=lambda: np.array([]))
    ccdf: np.ndarray = field(default_factory=lambda: np.array([]))
    k_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    hill_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    xmin_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_stability: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MarketData:
    ticker: str
    name: str
    prices: np.ndarray
    returns: np.ndarray
    dates: np.ndarray

    @property
    def n_observations(self) -> int:
        return len(self.returns)


# =============================================================================
# FRACTIONAL BROWNIAN MOTION (general H; used for diagnostics / the H_vol field)
# =============================================================================

class FractionalBrownianMotion:
    """Exact fractional Gaussian noise via Cholesky factorization of the
    increment covariance. With H = 0.5 this reduces to i.i.d. Gaussian noise."""

    def __init__(self, H: float = 0.5):
        if not 0 < H < 1:
            raise ValueError(f"H must be in (0,1), got {H}")
        self.H = H
        self._cholesky_cache: Dict[int, np.ndarray] = {}

    def covariance_matrix(self, n: int) -> np.ndarray:
        H2 = 2 * self.H
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        k = np.abs(i_idx - j_idx)
        return 0.5 * (np.abs(k + 1) ** H2 - 2 * np.abs(k) ** H2 + np.abs(k - 1) ** H2)

    def _get_cholesky(self, n: int) -> np.ndarray:
        if n not in self._cholesky_cache:
            cov = self.covariance_matrix(n)
            cov += 1e-10 * np.eye(n)
            self._cholesky_cache[n] = cholesky(cov, lower=True)
        return self._cholesky_cache[n]

    def generate_increments(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        # Fast path: H = 0.5 is white noise; no Cholesky needed.
        if abs(self.H - 0.5) < 1e-12:
            return z
        L = self._get_cholesky(n)
        return L @ z

    def generate_path(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        increments = self.generate_increments(n, seed)
        path = np.zeros(n + 1)
        path[1:] = np.cumsum(increments)
        return path


# =============================================================================
# MULTIFRACTAL STOCHASTIC VOLATILITY (exact log-correlated field via Cholesky)
# =============================================================================

class MultifractalVolatility:
    """Log-correlated Gaussian field with exact covariance

        Cov[omega(t), omega(s)] = lambda^2 * ln^+( L / |t - s| ),

    sampled directly by Cholesky factorization. The field is centered by
    subtracting Var[omega]/2 so that E[exp(omega)] = 1 and sigma0 is the level of
    the volatility. There is no Ornstein-Uhlenbeck superposition: the construction
    is exact and free of the Jlambda^2 normalization issue of independent OU sums.
    """

    def __init__(self, sigma0: float = 0.012, lambda_sq: float = 0.10, L: float = 252.0):
        self.sigma0 = sigma0
        self.lambda_sq = lambda_sq
        self.L = L
        self._chol_cache: Dict[int, np.ndarray] = {}

    def covariance_matrix(self, n: int) -> np.ndarray:
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        tau = np.abs(i_idx - j_idx).astype(float)
        # Regularize the diagonal lag (|t-s| -> 1/2 day) so ln^+ is finite at lag 0.
        tau[tau == 0] = 0.5
        return self.lambda_sq * np.log(np.maximum(self.L / tau, 1.0))

    def _get_cholesky(self, n: int) -> np.ndarray:
        if n not in self._chol_cache:
            cov = self.covariance_matrix(n)
            cov += 1e-8 * np.eye(n)
            self._chol_cache[n] = cholesky(cov, lower=True)
        return self._chol_cache[n]

    def variance(self) -> float:
        """Stationary variance Var[omega] = lambda^2 ln(L / 0.5)."""
        return self.lambda_sq * np.log(self.L / 0.5)

    def generate(self, n: int, seed: Optional[int] = None,
                 return_omega: bool = False):
        rng = np.random.default_rng(seed)
        L_chol = self._get_cholesky(n)
        z = rng.standard_normal(n)
        omega = L_chol @ z
        omega -= self.variance() / 2.0   # center so E[exp(omega)] = 1
        sigma = self.sigma0 * np.exp(omega)
        if return_omega:
            return sigma, omega
        return sigma


# =============================================================================
# FSDE MODEL
# =============================================================================

class FSDEModel:
    """Log-price FSDE

        d log S(t) = (mu - 1/2 sigma(t)^2) dt + sigma(t) dB(t),

    where B is Brownian (H_returns = 0.5) and sigma(t) = sigma0 exp(omega(t)) is
    the exact log-correlated multifractal field. The return driver B and the
    volatility field omega are independent, so the model is symmetric (zero
    skewness) unless the leverage extension is enabled.
    """

    def __init__(self, params: Optional[FSDEParameters] = None,
                 leverage: float = 0.0):
        if params is None:
            params = CONFIG
        self.params = params
        self.leverage = leverage  # Pochart-Bouchaud leverage coefficient (0 => symmetric)
        self.fbm = FractionalBrownianMotion(H=params.H_returns)
        self.volatility = MultifractalVolatility(
            sigma0=params.sigma0,
            lambda_sq=params.lambda_sq,
            L=params.L,
        )

    def simulate(self, x0: float, T: int, dt: float = 1.0,
                 seed: Optional[int] = None,
                 return_components: bool = False) -> Tuple:
        ret_seed = seed
        vol_seed = seed + 1000 if seed is not None else None

        dB = self.fbm.generate_increments(T, seed=ret_seed)          # H = 0.5 -> white noise
        sigma_t, omega_t = self.volatility.generate(T, seed=vol_seed, return_omega=True)

        # Optional leverage: couple past returns into future log-volatility
        # (preliminary skewed-MRW-style demonstration). Kept off by default.
        if self.leverage != 0.0:
            sigma_t = self._apply_leverage(sigma_t, dB, self.leverage)

        log_prices = np.zeros(T + 1)
        log_prices[0] = np.log(x0)
        for i in range(T):
            drift = (self.params.mu - 0.5 * sigma_t[i] ** 2) * dt
            diffusion = sigma_t[i] * dB[i] * np.sqrt(dt)
            log_prices[i + 1] = log_prices[i] + drift + diffusion

        prices = np.exp(log_prices)
        returns = np.diff(log_prices)

        if return_components:
            components = {
                "returns": returns,
                "volatility": sigma_t,
                "omega": omega_t,
                "innovations": dB,
            }
            return prices, components
        return prices, returns, sigma_t

    @staticmethod
    def _apply_leverage(sigma_t: np.ndarray, innovations: np.ndarray,
                        beta: float, kernel_len: int = 60,
                        decay: float = 20.0) -> np.ndarray:
        """Pochart-Bouchaud-style leverage kernel: past (standardized) return
        innovations feed negatively into current log-volatility, generating
        negative return-volatility correlation and thus negative skewness.

        log sigma(t) -> log sigma(t) - beta * sum_{u>0} K(u) eps(t-u),
        with an exponential kernel K(u) ~ exp(-u/decay). beta > 0 makes
        downward moves raise subsequent volatility.
        """
        lags = np.arange(1, kernel_len + 1)
        kernel = np.exp(-lags / decay)
        kernel /= kernel.sum()
        T = len(sigma_t)
        log_sigma = np.log(sigma_t)
        adj = np.zeros(T)
        for t in range(1, T):
            m = min(kernel_len, t)
            adj[t] = -beta * np.dot(kernel[:m], innovations[t - 1::-1][:m])
        return np.exp(log_sigma + adj)


# =============================================================================
# BASIC STATISTICS
# =============================================================================

def compute_statistics(returns: np.ndarray) -> Dict:
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "skewness": float(skew(returns)),
        "kurtosis": float(kurtosis(returns, fisher=True)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "annualized_return": float(np.mean(returns) * 252),
        "annualized_volatility": float(np.std(returns) * np.sqrt(252)),
    }


# =============================================================================
# TAIL ANALYSIS: Clauset-Shalizi-Newman (x_min by KS) + Hill + bootstrap CIs
# =============================================================================

def _powerlaw_alpha_mle(data: np.ndarray, x_min: float) -> Tuple[float, np.ndarray]:
    """Continuous power-law MLE for the scaling exponent given x_min.

    For a continuous power law p(x) ~ x^{-alpha} on x >= x_min, the MLE is
        alpha = 1 + n / sum_i ln(x_i / x_min).
    Returns (alpha, tail_data).
    """
    tail = data[data >= x_min]
    n = len(tail)
    if n < 10:
        return np.nan, tail
    alpha = 1.0 + n / np.sum(np.log(tail / x_min))
    return alpha, tail


def _ks_distance(tail: np.ndarray, x_min: float, alpha: float) -> float:
    """Kolmogorov-Smirnov distance between empirical CDF of the tail and the
    fitted continuous power-law CDF F(x) = 1 - (x/x_min)^{-(alpha-1)}."""
    if len(tail) < 2 or not np.isfinite(alpha):
        return np.inf
    xs = np.sort(tail)
    emp = np.arange(1, len(xs) + 1) / len(xs)
    theo = 1.0 - (xs / x_min) ** (-(alpha - 1.0))
    return float(np.max(np.abs(emp - theo)))


def fit_powerlaw_csn(data: np.ndarray,
                     xmin_quantiles: Tuple[float, float] = (0.50, 0.99),
                     n_xmin: int = 40) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
    """Clauset-Shalizi-Newman fit: choose x_min minimizing the KS distance.

    Returns (alpha, x_min, ks, n_tail, xmin_grid, alpha_stability).
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    if len(data) < 50:
        return np.nan, np.nan, np.inf, 0, np.array([]), np.array([])

    lo, hi = np.quantile(data, xmin_quantiles[0]), np.quantile(data, xmin_quantiles[1])
    xmin_grid = np.linspace(lo, hi, n_xmin)
    best = (np.nan, np.nan, np.inf, 0)
    alpha_stability = np.full(len(xmin_grid), np.nan)

    for idx, xm in enumerate(xmin_grid):
        alpha, tail = _powerlaw_alpha_mle(data, xm)
        if not np.isfinite(alpha) or len(tail) < 10:
            continue
        ks = _ks_distance(tail, xm, alpha)
        alpha_stability[idx] = alpha
        if ks < best[2]:
            best = (alpha, xm, ks, len(tail))

    return (*best, xmin_grid, alpha_stability)


def csn_bootstrap_pvalue(data: np.ndarray, alpha: float, x_min: float,
                         n_boot: int = 500, seed: int = SEED) -> float:
    """Parametric-bootstrap goodness-of-fit p-value for the power-law hypothesis,
    following Clauset-Shalizi-Newman. Synthetic datasets are generated by mixing
    the empirical body (x < x_min) with power-law draws (x >= x_min); the fraction
    of synthetic KS distances exceeding the observed one is the p-value.
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    n = len(data)
    if n < 50 or not np.isfinite(alpha):
        return np.nan

    body = data[data < x_min]
    n_tail = int(np.sum(data >= x_min))
    p_tail = n_tail / n
    _, tail_obs = _powerlaw_alpha_mle(data, x_min)
    ks_obs = _ks_distance(tail_obs, x_min, alpha)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_boot):
        n_from_tail = int(rng.binomial(n, p_tail))
        # power-law draws via inverse CDF: x = x_min * (1 - u)^{-1/(alpha-1)}
        u = rng.random(n_from_tail)
        synth_tail = x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        n_from_body = n - n_from_tail
        if len(body) > 0 and n_from_body > 0:
            synth_body = rng.choice(body, size=n_from_body, replace=True)
            synth = np.concatenate([synth_body, synth_tail])
        else:
            synth = synth_tail
        a_s, xm_s, ks_s, _, _, _ = fit_powerlaw_csn(synth)
        if np.isfinite(ks_s) and ks_s >= ks_obs:
            count += 1
    return count / n_boot


def hill_estimator(data: np.ndarray, k: int) -> Tuple[float, float]:
    """Hill estimator of the tail index using the top-k order statistics."""
    if k >= len(data) or k < 2:
        return np.nan, np.nan
    sorted_data = np.sort(data)[::-1]
    log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
    alpha = k / np.sum(log_ratios)
    se = alpha / np.sqrt(k)
    return alpha, se


def hill_curve(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hill plot: alpha as a function of the number of order statistics k."""
    data = np.sort(data[data > 0])[::-1]
    n = len(data)
    k_grid = np.arange(10, max(11, n // 2), max(1, n // 200))
    curve = np.array([hill_estimator(data, int(k))[0] for k in k_grid])
    return k_grid, curve


def block_bootstrap_ci(data: np.ndarray, statistic, block_size: int = 50,
                       n_boot: int = 500, seed: int = SEED,
                       ci: float = 0.95) -> Tuple[float, float]:
    """Circular block-bootstrap confidence interval for a scalar statistic of a
    1-D series, preserving short-range dependence."""
    data = np.asarray(data)
    n = len(data)
    if n < block_size * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    estimates = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_size) % n) for s in starts])[:n]
        try:
            estimates.append(statistic(data[idx]))
        except Exception:
            continue
    if not estimates:
        return (np.nan, np.nan)
    lo = np.nanpercentile(estimates, 100 * (1 - ci) / 2)
    hi = np.nanpercentile(estimates, 100 * (1 + ci) / 2)
    return float(lo), float(hi)


def analyze_tail(returns: np.ndarray, tail: str = "positive",
                 n_boot: int = 300, seed: int = SEED) -> TailResult:
    """Full tail analysis: CSN x_min/alpha/p-value, Hill estimate at the CSN
    threshold, and bootstrap CIs, with arrays for Hill and stability plots."""
    if tail == "positive":
        data = returns[returns > 0]
    else:
        data = np.abs(returns[returns < 0])
    data = data[data > 0]

    alpha_csn, x_min, ks, n_tail, xmin_grid, alpha_stab = fit_powerlaw_csn(data)
    p_val = csn_bootstrap_pvalue(data, alpha_csn, x_min, n_boot=n_boot, seed=seed)

    # Hill at k equal to the number of CSN-tail observations.
    sorted_desc = np.sort(data)[::-1]
    k_hill = max(10, int(n_tail))
    k_hill = min(k_hill, len(data) - 2)
    hill_alpha, _ = hill_estimator(data, k_hill)

    # CIs via bootstrap.
    def _csn_stat(d):
        a, *_ = fit_powerlaw_csn(d)
        return a

    def _hill_stat(d):
        return hill_estimator(d, min(k_hill, len(d) - 2))[0]

    csn_ci = block_bootstrap_ci(data, _csn_stat, block_size=50, n_boot=n_boot, seed=seed + 1)
    hill_ci = block_bootstrap_ci(data, _hill_stat, block_size=50, n_boot=n_boot, seed=seed + 2)

    k_grid, hc = hill_curve(data)
    x_sorted = np.sort(data)[::-1]
    ccdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    return TailResult(
        tail=tail, x_min=x_min, n_tail=n_tail,
        alpha_csn=alpha_csn, alpha_csn_ci=csn_ci, ks_distance=ks, p_value=p_val,
        hill_alpha=hill_alpha, hill_ci=hill_ci, hill_k=k_hill,
        x_ccdf=x_sorted, ccdf=ccdf, k_grid=k_grid, hill_curve=hc,
        xmin_grid=xmin_grid, alpha_stability=alpha_stab,
    )


# =============================================================================
# DFA AND MF-DFA
# =============================================================================

def dfa(data: np.ndarray, n_min: int = 10, n_max: Optional[int] = None,
        n_points: int = 20, order: int = 1) -> DFAResult:
    """Monofractal DFA (q = 2)."""
    N = len(data)
    if n_max is None:
        n_max = N // 4
    scales = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int))
    scales = scales[scales >= order + 2]
    profile = np.cumsum(data - np.mean(data))
    fluctuations = np.full(len(scales), np.nan)

    for i, n in enumerate(scales):
        n_seg = N // n
        if n_seg < 2:
            continue
        # Stack all segments of length n into a matrix and detrend them jointly.
        seg = profile[:n_seg * n].reshape(n_seg, n)
        x = np.arange(n)
        # Vandermonde design matrix for a polynomial of given order.
        V = np.vander(x, order + 1)
        # Least-squares trend for every segment at once: coeffs (order+1, n_seg).
        coeffs, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        trend = (V @ coeffs).T
        resid = seg - trend
        fluctuations[i] = np.sqrt(np.mean(resid ** 2))

    valid = ~np.isnan(fluctuations)
    slope, intercept, r, _, se = linregress(np.log10(scales[valid]), np.log10(fluctuations[valid]))
    return DFAResult(scales=scales, fluctuations=fluctuations, H=slope,
                     H_se=se, R2=r ** 2, intercept=intercept)


def mfdfa(data: np.ndarray, q_values: np.ndarray = None,
          n_min: int = 16, n_max: Optional[int] = None,
          n_points: int = 20, order: int = 1) -> MFDFAResult:
    """Multifractal DFA: generalized Hurst spectrum H(q).

    A q-independent H(q) signals monofractality; a decreasing H(q) signals genuine
    multiscaling. The slope B in the linear fit H(q) = A + B q is reported as a
    multiscaling proxy.
    """
    if q_values is None:
        q_values = np.array([-5, -4, -3, -2, -1, 0.0001, 1, 2, 3, 4, 5], dtype=float)
    N = len(data)
    if n_max is None:
        n_max = N // 10
    scales = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int))
    scales = scales[scales >= order + 2]
    profile = np.cumsum(data - np.mean(data))

    # Fluctuation F_q(n) for each scale and q.
    Fq = np.full((len(q_values), len(scales)), np.nan)
    for si, n in enumerate(scales):
        n_seg = N // n
        if n_seg < 4:
            continue
        seg = profile[:n_seg * n].reshape(n_seg, n)
        x = np.arange(n)
        V = np.vander(x, order + 1)
        coeffs, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        trend = (V @ coeffs).T
        f2 = np.mean((seg - trend) ** 2, axis=1)
        for qi, q in enumerate(q_values):
            if abs(q) < 1e-6:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(f2 + 1e-30)))
            else:
                Fq[qi, si] = (np.mean(f2 ** (q / 2.0))) ** (1.0 / q)

    Hq = np.full(len(q_values), np.nan)
    Hq_se = np.full(len(q_values), np.nan)
    logs = np.log10(scales)
    for qi in range(len(q_values)):
        y = np.log10(Fq[qi])
        valid = np.isfinite(y)
        if np.sum(valid) >= 3:
            slope, _, _, _, se = linregress(logs[valid], y[valid])
            Hq[qi], Hq_se[qi] = slope, se

    valid = np.isfinite(Hq)
    B, A, _, _, B_se = linregress(q_values[valid], Hq[valid])
    return MFDFAResult(q_values=q_values, Hq=Hq, Hq_se=Hq_se,
                       slope_B=B, slope_B_se=B_se, intercept_A=A)


# =============================================================================
# SHUFFLE TEST AND ROLLING HURST
# =============================================================================

def shuffle_test(series: np.ndarray, n_shuffle: int = 200, seed: int = SEED,
                 n_min: int = 10, n_max: Optional[int] = None) -> Dict:
    """Compare the DFA Hurst exponent of a series with its shuffled surrogates.

    Shuffling destroys temporal ordering. A Hurst exponent that drops to ~0.5
    after shuffling is consistent with, but not proof of, genuine long memory:
    the test cannot by itself distinguish long memory from regime switching or
    structural breaks. This caveat is stated explicitly in the manuscript.
    """
    H_obs = dfa(series, n_min=n_min, n_max=n_max).H
    rng = np.random.default_rng(seed)
    H_shuf = np.empty(n_shuffle)
    for i in range(n_shuffle):
        s = series.copy()
        rng.shuffle(s)
        H_shuf[i] = dfa(s, n_min=n_min, n_max=n_max).H
    mean_s, std_s = float(np.mean(H_shuf)), float(np.std(H_shuf))
    z = (H_obs - mean_s) / std_s if std_s > 0 else np.nan
    return {"H_observed": float(H_obs), "H_shuffled_mean": mean_s,
            "H_shuffled_std": std_s, "z_score": float(z),
            "H_shuffled": H_shuf}


def rolling_hurst(series: np.ndarray, window: int = 756, step: int = 21,
                  n_min: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling-window DFA Hurst exponent. Window default ~3 trading years.

    Useful to detect whether apparent long memory is induced by regime changes
    (e.g. 2008, 2020) rather than stationary persistence.
    """
    centers, H_vals = [], []
    for start in range(0, len(series) - window + 1, step):
        seg = series[start:start + window]
        H_vals.append(dfa(seg, n_min=n_min, n_max=window // 4).H)
        centers.append(start + window // 2)
    return np.array(centers), np.array(H_vals)


# =============================================================================
# LAMBDA CALIBRATION (intermittency from the autocovariance of log|r|)
# =============================================================================

def estimate_lambda_sq(returns: np.ndarray, L: float = 252.0,
                       lag_min: int = 1, lag_max: int = 60) -> Tuple[float, float]:
    """Estimate the intermittency parameter lambda^2 from the slope of the
    autocovariance of ln|r| against ln(lag), following the MRW relation

        Cov[ln|r_t|, ln|r_{t+tau}|] ~ -lambda^2 ln(tau) + const,   1 << tau << L.

    Returns (lambda_sq_hat, se). The minus-slope of the log-lag regression is the
    estimate of lambda^2.
    """
    r = returns[np.abs(returns) > 0]
    x = np.log(np.abs(r))
    x = x - np.mean(x)
    lags = np.arange(lag_min, lag_max + 1)
    cov = np.array([np.mean(x[:-lag] * x[lag:]) for lag in lags])
    valid = cov > 0
    if np.sum(valid) < 5:
        # fall back to all lags; allow the regression to run on raw covariance
        valid = np.ones_like(cov, dtype=bool)
    slope, intercept, r_val, _, se = linregress(np.log(lags[valid]), cov[valid])
    lam_sq = max(-slope, 0.0)
    return float(lam_sq), float(se)


# =============================================================================
# PLOTTING (matplotlib configured here so the analysis core stays import-light)
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved, not shown, in batch runs
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "vol": "#d62728",
    "price": "#1f77b4",
    "returns": "#2ca02c",
    "pos": "#4a7298",
    "neg": "#c0392b",
    "fit_pos": "#1a365d",
    "fit_neg": "#8b0000",
    "ref": "#27ae60",
    "dfa_ret": "#4a7298",
    "dfa_vol": "#a65353",
    "rw": "#7f8c8d",
    "mf": "#6a3d9a",
}

FIGDIR = "figures"


def _ensure_figdir():
    import os
    os.makedirs(FIGDIR, exist_ok=True)


def save_fig(fig, name: str):
    _ensure_figdir()
    path = f"{FIGDIR}/{name}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_volatility(sigma_t: np.ndarray):
    fig, ax = plt.subplots(figsize=(15, 5))
    n = len(sigma_t)
    ax.plot(np.arange(n), sigma_t, color=COLORS["vol"], linewidth=0.8)
    ax.axhline(np.mean(sigma_t), color="black", ls="--", lw=1.0,
               label=fr"$\bar{{\sigma}} = {np.mean(sigma_t):.4f}$")
    ax.set_xlabel(r"$t$ (days)")
    ax.set_ylabel(r"$\sigma(t)$")
    ax.set_title("Multifractal stochastic volatility", fontweight="bold")
    ax.set_xlim([0, n - 1])
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_prices(prices: np.ndarray):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(np.arange(len(prices)), prices, color=COLORS["price"], linewidth=0.9)
    ax.set_xlabel(r"$t$ (days)")
    ax.set_ylabel(r"$S(t)$")
    ax.set_title("Simulated price series", fontweight="bold")
    ax.set_xlim([0, len(prices) - 1])
    plt.tight_layout()
    return fig


def plot_returns(returns: np.ndarray):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(np.arange(len(returns)), returns, color=COLORS["returns"], linewidth=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel(r"$t$ (days)")
    ax.set_ylabel(r"$r(t)$")
    ax.set_title("Simulated log-returns", fontweight="bold")
    ax.set_xlim([0, len(returns) - 1])
    plt.tight_layout()
    return fig


def plot_abs_returns(returns: np.ndarray):
    """|returns| as an observable proxy for the hidden volatility (Reviewer #2c)."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ar = np.abs(returns)
    ax.plot(np.arange(len(ar)), ar, color=COLORS["vol"], linewidth=0.7)
    ax.axhline(np.mean(ar), color="black", ls="--", lw=1.0,
               label=fr"mean $|r| = {np.mean(ar):.4f}$")
    ax.set_xlabel(r"$t$ (days)")
    ax.set_ylabel(r"$|r(t)|$")
    ax.set_title("Absolute returns as observable volatility proxy", fontweight="bold")
    ax.set_xlim([0, len(ar) - 1])
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_volatility_pdf(sigma_t: np.ndarray):
    """Histogram of log-volatility with normal overlay + QQ-plot (Reviewer #2d)."""
    log_sig = np.log(sigma_t)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(log_sig, bins=60, density=True, color=COLORS["vol"], alpha=0.55,
            edgecolor="white", linewidth=0.3)
    mu_, sd_ = np.mean(log_sig), np.std(log_sig)
    xs = np.linspace(log_sig.min(), log_sig.max(), 300)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu_) / sd_) ** 2) / (sd_ * np.sqrt(2 * np.pi)),
            "k--", lw=2, label="Gaussian fit")
    ax.set_xlabel(r"$\ln \sigma(t)$")
    ax.set_ylabel("density")
    ax.set_title("(a) Log-volatility distribution", fontweight="bold")
    ax.legend(framealpha=0.95)
    ax2 = axes[1]
    probplot(log_sig, dist="norm", plot=ax2)
    ax2.set_title("(b) Normal QQ-plot of $\\ln\\sigma$", fontweight="bold")
    ax2.get_lines()[0].set_color(COLORS["vol"])
    ax2.get_lines()[0].set_markersize(3)
    ax2.get_lines()[1].set_color("black")
    plt.tight_layout()
    return fig


def plot_ccdf(tp: TailResult, tn: TailResult, name: str, alpha_ref: float = 3.0):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, tr, col, fitcol, lab in [
        (axes[0], tp, COLORS["pos"], COLORS["fit_pos"], "Positive"),
        (axes[1], tn, COLORS["neg"], COLORS["fit_neg"], "Negative"),
    ]:
        ax.loglog(tr.x_ccdf, tr.ccdf, "o", color=col, ms=3, alpha=0.5)
        if np.isfinite(tr.alpha_csn) and tr.x_min > 0:
            xs = np.logspace(np.log10(tr.x_min), np.log10(tr.x_ccdf.max()), 100)
            p_at_xmin = np.mean(tr.x_ccdf >= tr.x_min)
            ys = p_at_xmin * (xs / tr.x_min) ** (-(tr.alpha_csn - 1.0))
            ax.loglog(xs, ys, "-", color=fitcol, lw=2,
                      label=fr"$\alpha = {tr.alpha_csn:.2f}$")
            ax.axvline(tr.x_min, color="gray", ls=":", lw=1.2,
                       label=fr"$x_{{\min}} = {tr.x_min:.3f}$")
        ax.set_xlabel(r"$|r|$")
        ax.set_ylabel(r"$P(|r| > x)$")
        ax.set_title(f"({'a' if lab=='Positive' else 'b'}) {lab} tail", fontweight="bold")
        ax.legend(framealpha=0.95)
    fig.suptitle(f"CCDF of log-returns - {name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_hill_stability(tp: TailResult, tn: TailResult, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for tr, col, lab in [(tp, COLORS["pos"], "positive"), (tn, COLORS["neg"], "negative")]:
        ax.plot(tr.k_grid, tr.hill_curve, color=col, lw=1.2, label=f"{lab} tail")
        if np.isfinite(tr.hill_alpha):
            ax.axhline(tr.hill_alpha, color=col, ls=":", lw=1.0)
    ax.axhline(3.0, color=COLORS["ref"], ls="--", lw=1.2, label=r"$\alpha = 3$")
    ax.set_xlabel(r"$k$ (order statistics)")
    ax.set_ylabel(r"Hill $\hat{\alpha}(k)$")
    ax.set_title("(a) Hill plot", fontweight="bold")
    ax.set_ylim(0, 8)
    ax.legend(framealpha=0.95)
    ax2 = axes[1]
    for tr, col, lab in [(tp, COLORS["pos"], "positive"), (tn, COLORS["neg"], "negative")]:
        valid = np.isfinite(tr.alpha_stability)
        ax2.plot(tr.xmin_grid[valid], tr.alpha_stability[valid], color=col, lw=1.2,
                 label=f"{lab} tail")
        if np.isfinite(tr.x_min):
            ax2.axvline(tr.x_min, color=col, ls=":", lw=1.0)
    ax2.axhline(3.0, color=COLORS["ref"], ls="--", lw=1.2, label=r"$\alpha = 3$")
    ax2.set_xlabel(r"$x_{\min}$")
    ax2.set_ylabel(r"CSN $\hat{\alpha}(x_{\min})$")
    ax2.set_title("(b) Threshold stability", fontweight="bold")
    ax2.set_ylim(0, 8)
    ax2.legend(framealpha=0.95)
    fig.suptitle(f"Tail-index diagnostics - {name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_dfa(dfa_ret: DFAResult, dfa_vol: DFAResult, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, d, col, lab in [
        (axes[0], dfa_ret, COLORS["dfa_ret"], "Returns"),
        (axes[1], dfa_vol, COLORS["dfa_vol"], "Absolute returns"),
    ]:
        v = ~np.isnan(d.fluctuations)
        ax.scatter(d.scales[v], d.fluctuations[v], color=col, s=55, alpha=0.8, zorder=3)
        nf = np.logspace(np.log10(d.scales[v][0]), np.log10(d.scales[v][-1]), 100)
        ax.plot(nf, (10 ** d.intercept) * nf ** d.H, "k--", lw=2,
                label=fr"$H_{{\mathrm{{DFA}}}} = {d.H:.3f}$")
        ref = d.fluctuations[v][0] * (nf / d.scales[v][0]) ** 0.5
        ax.plot(nf, ref, ":", color=COLORS["rw"], lw=1.5, label=r"$H = 0.5$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$F(n)$")
        ax.set_title(f"({'a' if lab=='Returns' else 'b'}) {lab}", fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.95)
    fig.suptitle(f"Detrended Fluctuation Analysis - {name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_mfdfa(mf_ret: MFDFAResult, mf_vol: MFDFAResult, name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(mf_ret.q_values, mf_ret.Hq, yerr=mf_ret.Hq_se, fmt="o-",
                color=COLORS["dfa_ret"], capsize=3,
                label=fr"returns ($B = {mf_ret.slope_B:.3f}$)")
    ax.errorbar(mf_vol.q_values, mf_vol.Hq, yerr=mf_vol.Hq_se, fmt="s-",
                color=COLORS["dfa_vol"], capsize=3,
                label=fr"$|$returns$|$ ($B = {mf_vol.slope_B:.3f}$)")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$H(q)$")
    ax.set_title(f"Generalized Hurst spectrum (MF-DFA) - {name}", fontweight="bold")
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_rolling_hurst(centers: np.ndarray, H_vals: np.ndarray, dates: np.ndarray,
                       name: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    if dates is not None and len(dates) > 0:
        x = dates[np.clip(centers, 0, len(dates) - 1)]
    else:
        x = centers
    ax.plot(x, H_vals, color=COLORS["dfa_vol"], lw=1.4)
    ax.axhline(0.5, color=COLORS["rw"], ls=":", lw=1.2, label=r"$H = 0.5$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"rolling $H_{\mathrm{DFA}}$ of $|r|$")
    ax.set_title(f"Rolling-window Hurst exponent of volatility - {name}", fontweight="bold")
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_sensitivity(lambda_vals, kurt_vals, vol_vals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lambda_vals, kurt_vals, "o-", color=COLORS["mf"])
    axes[0].set_xlabel(r"$\lambda^2$"); axes[0].set_ylabel("excess kurtosis")
    axes[0].set_title("(a) Excess kurtosis", fontweight="bold")
    axes[1].plot(lambda_vals, vol_vals, "s-", color=COLORS["price"])
    axes[1].axhline(20, color=COLORS["rw"], ls="--", lw=1.0, label="20%")
    axes[1].set_xlabel(r"$\lambda^2$"); axes[1].set_ylabel("annualized volatility (%)")
    axes[1].set_title("(b) Annualized volatility", fontweight="bold")
    axes[1].legend(framealpha=0.95)
    plt.tight_layout()
    return fig


# =============================================================================
# DATA FETCHING (Yahoo Finance via yfinance; falls back gracefully)
# =============================================================================

def fetch_market_data(ticker: str, name: str, start: str = "2000-01-01",
                      end: str = "2024-12-31") -> Optional[MarketData]:
    """Fetch daily data. Uses adjusted close when available, otherwise close;
    log-returns are differences of log closing prices. Requires network access
    to Yahoo Finance."""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) < 100:
            print(f"  [warn] insufficient data for {ticker}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if "Adj Close" in data.columns:
            prices = data["Adj Close"].values
        elif "Close" in data.columns:
            prices = data["Close"].values
        else:
            raise ValueError(f"no price column in {data.columns.tolist()}")
        returns = np.diff(np.log(prices))
        dates = data.index[1:].values
        return MarketData(ticker=ticker, name=name, prices=prices,
                          returns=returns, dates=dates)
    except Exception as e:
        print(f"  [warn] could not fetch {ticker}: {e}")
        return None


# =============================================================================
# PIPELINE
# =============================================================================

def pooled_returns(params: FSDEParameters, n_realizations: int, T: int,
                   x0: float = 100.0, seed: int = SEED) -> np.ndarray:
    model = FSDEModel(params)
    return np.concatenate([model.simulate(x0, T, seed=seed + i)[1]
                           for i in range(n_realizations)])


def run_sensitivity(params: FSDEParameters, lambda_values=None,
                    n_realizations: int = 5, T: int = 2520, seed: int = SEED) -> Dict:
    if lambda_values is None:
        lambda_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    kurt, vol = [], []
    for lsq in lambda_values:
        p = FSDEParameters(H_returns=params.H_returns, lambda_sq=lsq,
                           sigma0=params.sigma0, mu=params.mu, L=params.L)
        ks, vs = [], []
        m = FSDEModel(p)
        for i in range(n_realizations):
            _, r, _ = m.simulate(100.0, T, seed=seed + i * 100)
            st = compute_statistics(r)
            ks.append(st["kurtosis"]); vs.append(st["annualized_volatility"] * 100)
        kurt.append(float(np.mean(ks))); vol.append(float(np.mean(vs)))
    return {"lambda_values": lambda_values, "kurtosis": kurt, "volatility": vol}


def run_leverage_sweep(params: FSDEParameters, beta_values=None,
                       n_realizations: int = 60, T: int = 2520,
                       seed: int = SEED) -> Dict:
    """Sweep the Pochart-Bouchaud leverage coefficient beta to find the value
    that brings the simulated skewness toward the empirical range (about -0.34
    to -0.39), while checking that the tail and persistence properties are not
    materially disturbed.

    Skewness of heavy-tailed samples has very high sampling variance, so for
    each beta the skewness is reported both from a large pooled sample and as
    the median of the per-realization skewness, which is the more robust
    summary. Excess kurtosis and the DFA exponent of the absolute returns are
    reported alongside so that any distortion of the rest of the construction
    is visible.

    The symmetric baseline (beta = 0) is included as the first row for
    reference. Returns a dictionary with one entry per beta.
    """
    if beta_values is None:
        beta_values = [0.0, 0.6, 0.9, 1.2, 1.5, 2.0]

    rows = []
    for beta in beta_values:
        model = FSDEModel(params, leverage=beta)
        pooled, sk_real = [], []
        for i in range(n_realizations):
            _, r, _ = model.simulate(100.0, T, seed=seed + i)
            pooled.append(r)
            sk_real.append(float(skew(r)))
        pooled = np.concatenate(pooled)
        abs_pooled = np.abs(pooled)

        skew_pooled = float(skew(pooled))
        skew_median = float(np.median(sk_real))
        exc_kurt = float(kurtosis(pooled, fisher=True))
        ann_vol = float(np.std(pooled) * np.sqrt(252) * 100)
        # Persistence check on the volatility proxy.
        H_vol = dfa(abs_pooled, n_min=10, n_max=len(abs_pooled) // 10).H

        rows.append({
            "beta": float(beta),
            "skew_pooled": skew_pooled,
            "skew_median": skew_median,
            "excess_kurtosis": exc_kurt,
            "ann_volatility": ann_vol,
            "H_vol": float(H_vol),
            "n_pooled": int(len(pooled)),
        })
    return {"rows": rows, "n_realizations": n_realizations, "T": T}


def summarize_simulation(params: FSDEParameters, n_realizations: int = 20,
                         T: int = 2520, n_boot: int = 300, seed: int = SEED,
                         make_figures: bool = True) -> Dict:
    """Compute every simulated number reported in the manuscript, with CIs."""
    print(f"  pooling {n_realizations} x {T} steps ...")
    model = FSDEModel(params)
    prices, comp = model.simulate(100.0, T, seed=seed, return_components=True)
    pooled = pooled_returns(params, n_realizations, T, seed=seed)
    abs_pooled = np.abs(pooled)

    stats = compute_statistics(pooled)
    # robust skewness: median across realizations
    sk_real = [skew(model.simulate(100.0, T, seed=seed + i)[1]) for i in range(n_realizations)]
    stats["skewness_median"] = float(np.median(sk_real))

    print("  tails ...")
    tp = analyze_tail(pooled, "positive", n_boot=n_boot, seed=seed)
    tn = analyze_tail(pooled, "negative", n_boot=n_boot, seed=seed)

    print("  DFA / MF-DFA ...")
    dfa_ret = dfa(pooled, n_min=10, n_max=len(pooled) // 10)
    dfa_vol = dfa(abs_pooled, n_min=10, n_max=len(abs_pooled) // 10)
    H_ret_ci = block_bootstrap_ci(pooled, lambda d: dfa(d, n_min=10, n_max=len(d)//10).H,
                                  block_size=252, n_boot=200, seed=seed + 3)
    H_vol_ci = block_bootstrap_ci(abs_pooled, lambda d: dfa(d, n_min=10, n_max=len(d)//10).H,
                                  block_size=252, n_boot=200, seed=seed + 4)
    mf_ret = mfdfa(pooled)
    mf_vol = mfdfa(abs_pooled)

    print("  shuffle test ...")
    sh = shuffle_test(abs_pooled[:T], n_shuffle=100, seed=seed)

    print("  lambda calibration ...")
    lam_hat, lam_se = estimate_lambda_sq(pooled, L=params.L)

    figs = {}
    if make_figures:
        figs["vol"] = save_fig(plot_volatility(comp["volatility"]), "fsde_vol.png")
        figs["price"] = save_fig(plot_prices(prices), "fsde_price.png")
        figs["returns"] = save_fig(plot_returns(comp["returns"]), "fsde_returns.png")
        figs["absret"] = save_fig(plot_abs_returns(comp["returns"]), "fsde_absret.png")
        figs["volpdf"] = save_fig(plot_volatility_pdf(comp["volatility"]), "fsde_volpdf.png")
        figs["ccdf"] = save_fig(plot_ccdf(tp, tn, "FSDE simulation"), "fsde_ccdf.png")
        figs["hill"] = save_fig(plot_hill_stability(tp, tn, "FSDE simulation"), "fsde_hill.png")
        figs["dfa"] = save_fig(plot_dfa(dfa_ret, dfa_vol, "FSDE simulation"), "fsde_dfa.png")
        figs["mfdfa"] = save_fig(plot_mfdfa(mf_ret, mf_vol, "FSDE simulation"), "fsde_mfdfa.png")

    return {
        "stats": stats, "tp": tp, "tn": tn,
        "dfa_ret": dfa_ret, "dfa_vol": dfa_vol,
        "H_ret_ci": H_ret_ci, "H_vol_ci": H_vol_ci,
        "mf_ret": mf_ret, "mf_vol": mf_vol,
        "shuffle": sh, "lambda_hat": (lam_hat, lam_se),
        "alpha_pred": params.theoretical_alpha,
        "figs": figs, "n_pooled": len(pooled),
    }


def summarize_empirical(market: MarketData, n_boot: int = 300, seed: int = SEED,
                        make_figures: bool = True) -> Dict:
    r = market.returns
    stats = compute_statistics(r)
    tp = analyze_tail(r, "positive", n_boot=n_boot, seed=seed)
    tn = analyze_tail(r, "negative", n_boot=n_boot, seed=seed)
    dfa_ret = dfa(r, n_min=10, n_max=len(r) // 4)
    dfa_vol = dfa(np.abs(r), n_min=10, n_max=len(r) // 4)
    H_vol_ci = block_bootstrap_ci(np.abs(r), lambda d: dfa(d, n_min=10, n_max=len(d)//4).H,
                                  block_size=252, n_boot=200, seed=seed + 5)
    mf_ret = mfdfa(r)
    mf_vol = mfdfa(np.abs(r))
    sh = shuffle_test(np.abs(r), n_shuffle=100, seed=seed)
    lam_hat, lam_se = estimate_lambda_sq(r, L=252.0)
    centers, H_roll = rolling_hurst(np.abs(r), window=756, step=21)

    figs = {}
    if make_figures:
        key = market.ticker.replace("^", "")
        figs["ccdf"] = save_fig(plot_ccdf(tp, tn, market.name), f"emp_{key}_ccdf.png")
        figs["hill"] = save_fig(plot_hill_stability(tp, tn, market.name), f"emp_{key}_hill.png")
        figs["dfa"] = save_fig(plot_dfa(dfa_ret, dfa_vol, market.name), f"emp_{key}_dfa.png")
        figs["mfdfa"] = save_fig(plot_mfdfa(mf_ret, mf_vol, market.name), f"emp_{key}_mfdfa.png")
        figs["rolling"] = save_fig(
            plot_rolling_hurst(centers, H_roll, market.dates, market.name),
            f"emp_{key}_rolling.png")

    return {"stats": stats, "tp": tp, "tn": tn, "dfa_ret": dfa_ret, "dfa_vol": dfa_vol,
            "H_vol_ci": H_vol_ci, "mf_ret": mf_ret, "mf_vol": mf_vol, "shuffle": sh,
            "lambda_hat": (lam_hat, lam_se), "figs": figs,
            "rolling": (centers, H_roll)}


def _fmt_tail(t: TailResult) -> str:
    return (f"x_min={t.x_min:.4f} n_tail={t.n_tail} "
            f"alpha_csn={t.alpha_csn:.3f} CI[{t.alpha_csn_ci[0]:.2f},{t.alpha_csn_ci[1]:.2f}] "
            f"p={t.p_value:.3f} hill={t.hill_alpha:.3f} "
            f"CI[{t.hill_ci[0]:.2f},{t.hill_ci[1]:.2f}]")


def report(sim: Dict, sens: Dict, emp: Dict, lev: Dict = None):
    print("\n" + "=" * 70)
    print("SIMULATED RESULTS  (config: H_returns={}, lambda_sq={}, sigma0={}, L={})".format(
        CONFIG.H_returns, CONFIG.lambda_sq, CONFIG.sigma0, CONFIG.L))
    print("=" * 70)
    s = sim["stats"]
    print(f"n_pooled            = {sim['n_pooled']}")
    print(f"ann. volatility     = {s['annualized_volatility']*100:.2f}%")
    print(f"excess kurtosis     = {s['kurtosis']:.2f}")
    print(f"skewness (pooled)   = {s['skewness']:.4f}")
    print(f"skewness (median)   = {s['skewness_median']:.4f}")
    print(f"alpha predicted     = {sim['alpha_pred']:.3f}  [sqrt(2/lambda^2)]")
    print(f"POS tail: {_fmt_tail(sim['tp'])}")
    print(f"NEG tail: {_fmt_tail(sim['tn'])}")
    print(f"H_DFA returns       = {sim['dfa_ret'].H:.3f} (se {sim['dfa_ret'].H_se:.3f}) "
          f"CI[{sim['H_ret_ci'][0]:.3f},{sim['H_ret_ci'][1]:.3f}]")
    print(f"H_DFA |returns|     = {sim['dfa_vol'].H:.3f} (se {sim['dfa_vol'].H_se:.3f}) "
          f"CI[{sim['H_vol_ci'][0]:.3f},{sim['H_vol_ci'][1]:.3f}]")
    print(f"MF-DFA B (returns)  = {sim['mf_ret'].slope_B:.4f} (se {sim['mf_ret'].slope_B_se:.4f})")
    print(f"MF-DFA B (|returns|)= {sim['mf_vol'].slope_B:.4f} (se {sim['mf_vol'].slope_B_se:.4f})")
    sh = sim["shuffle"]
    print(f"shuffle |r|: H_obs={sh['H_observed']:.3f} H_shuf={sh['H_shuffled_mean']:.3f}"
          f"+-{sh['H_shuffled_std']:.3f} z={sh['z_score']:.1f}")
    print(f"lambda_sq_hat       = {sim['lambda_hat'][0]:.4f}")

    print("\n--- sensitivity (lambda^2 -> kurtosis, ann.vol%) ---")
    for l, k, v in zip(sens["lambda_values"], sens["kurtosis"], sens["volatility"]):
        print(f"  lambda^2={l:.2f}  kurt={k:6.2f}  vol={v:5.1f}%")

    if lev is not None:
        print(f"\n--- leverage sweep (Pochart-Bouchaud beta), "
              f"{lev['n_realizations']} realizations pooled ---")
        print(f"  {'beta':>5} {'skew_pool':>10} {'skew_med':>9} "
              f"{'exc_kurt':>9} {'vol%':>6} {'H_vol':>6}")
        for r in lev["rows"]:
            print(f"  {r['beta']:5.2f} {r['skew_pooled']:10.3f} {r['skew_median']:9.3f} "
                  f"{r['excess_kurtosis']:9.2f} {r['ann_volatility']:6.1f} {r['H_vol']:6.3f}")
        print("  target empirical skewness: S&P 500 = -0.39, Ibovespa = -0.34")
        print("  (choose the largest beta whose exc_kurt and H_vol stay close to")
        print("   the beta=0 row, then report its skew_pooled/skew_median in 3.5)")

    print("\n" + "=" * 70)
    print("EMPIRICAL RESULTS")
    print("=" * 70)
    if not emp:
        print("  [no network] empirical block not run; use [[RODAR]] markers in text.")
    for key, e in emp.items():
        st = e["stats"]
        print(f"\n[{key}]")
        print(f"  excess kurtosis = {st['kurtosis']:.2f}  skewness = {st['skewness']:.4f}")
        print(f"  POS {_fmt_tail(e['tp'])}")
        print(f"  NEG {_fmt_tail(e['tn'])}")
        print(f"  H_DFA returns   = {e['dfa_ret'].H:.3f} (se {e['dfa_ret'].H_se:.3f})")
        print(f"  H_DFA |returns| = {e['dfa_vol'].H:.3f} (se {e['dfa_vol'].H_se:.3f}) "
              f"CI[{e['H_vol_ci'][0]:.3f},{e['H_vol_ci'][1]:.3f}]")
        print(f"  MF-DFA B ret/vol= {e['mf_ret'].slope_B:.4f} / {e['mf_vol'].slope_B:.4f}")
        sh = e["shuffle"]
        print(f"  shuffle |r|: H_obs={sh['H_observed']:.3f} -> shuf {sh['H_shuffled_mean']:.3f} z={sh['z_score']:.1f}")
        print(f"  lambda_sq_hat   = {e['lambda_hat'][0]:.4f}")


def main(n_realizations: int = 20, T: int = 2520, n_boot: int = 300,
         seed: int = SEED, make_figures: bool = True, fetch_empirical: bool = True,
         run_leverage: bool = True) -> Dict:
    print("FSDE pipeline (revised)")
    print("-" * 40)
    print("Phase A-D: simulated summary")
    sim = summarize_simulation(CONFIG, n_realizations=n_realizations, T=T,
                               n_boot=n_boot, seed=seed, make_figures=make_figures)
    print("Sensitivity analysis")
    sens = run_sensitivity(CONFIG, n_realizations=5, T=T, seed=seed)
    if make_figures:
        save_fig(plot_sensitivity(sens["lambda_values"], sens["kurtosis"], sens["volatility"]),
                 "fsde_sensitivity.png")

    lev = None
    if run_leverage:
        print("Leverage sweep (skewness extension)")
        lev = run_leverage_sweep(CONFIG, n_realizations=max(60, n_realizations * 3),
                                 T=T, seed=seed)

    emp = {}
    if fetch_empirical:
        print("Phase E-F: empirical validation")
        for tic, nm in [("^GSPC", "S&P 500"), ("^BVSP", "Ibovespa")]:
            md = fetch_market_data(tic, nm, "2000-01-01", "2024-12-31")
            if md is not None:
                emp[nm] = summarize_empirical(md, n_boot=n_boot, seed=seed,
                                              make_figures=make_figures)

    report(sim, sens, emp, lev)
    return {"simulated": sim, "sensitivity": sens, "empirical": emp, "leverage": lev}


if __name__ == "__main__":
    results = main(n_realizations=20, T=2520, n_boot=300, seed=SEED,
                   make_figures=True, fetch_empirical=True, run_leverage=True)
