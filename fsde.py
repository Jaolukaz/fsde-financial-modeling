#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================================
Modeling Stock Time Evolution Using a Fractional Stochastic Differential Equation:
Reproducing Stylized Facts Through Multifractal Volatility
===================================================================================

Supplementary Material

Authors:
    João Lucas de Pinho Carvalho
    Department of Mathematics — Federal Center for Technological Education of Minas Gerais (CEFET-MG)
    Belo Horizonte, MG, Brazil
    
    Leonardo dos Santos Lima
    Department of Physics — Federal Center for Technological Education of Minas Gerais (CEFET-MG)
    Belo Horizonte, MG, Brazil
    

Reference:
    CARVALHO, J.L.P., LIMA, L.S. (2026) - Modeling Stock Time Evolution Using
    a Fractional Stochastic Differential Equation: Reproducing Stylized Facts
    Through Multifractal Volatility. Physica A.

Implementation Phases:
    Phase 1: Fractional Brownian Motion generation via Cholesky decomposition
    Phase 2: Multifractal stochastic volatility via superposed O-U processes
    Phase 3: Complete FSDE simulation
    Phase 4: Tail distribution analysis (CCDF, Hill estimator)
    Phase 5: Detrended Fluctuation Analysis for Hurst exponent estimation
    Phase 6: Empirical validation against S&P 500 and Ibovespa

================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import kurtosis, linregress, skew

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'fbm': '#9467bd',
    'volatility': '#d62728',
    'price': '#1f77b4',
    'returns': '#2ca02c',
    'positive_tail': '#4a7298',
    'negative_tail': '#c0392b',
    'fit_positive': '#1a365d',
    'fit_negative': '#8b0000',
    'alpha_3': '#27ae60',
    'dfa_returns': '#4a7298',
    'dfa_volatility': '#a65353',
    'random_walk': '#7f8c8d',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FSDEParameters:
    H: float = 0.70
    lambda_sq: float = 0.04
    sigma0: float = 0.012
    mu: float = 0.00005
    L: float = 252.0
    
    def __post_init__(self):
        assert 0 < self.H < 1
        assert self.lambda_sq >= 0
        assert self.sigma0 > 0
        assert self.L > 0
    
    @property
    def theoretical_alpha(self) -> float:
        return 1.0 / self.lambda_sq if self.lambda_sq > 0 else np.inf


@dataclass
class DFAResult:
    scales: np.ndarray
    fluctuations: np.ndarray
    H: float
    H_se: float
    R2: float
    intercept: float


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
# FRACTIONAL BROWNIAN MOTION
# =============================================================================

class FractionalBrownianMotion:
    def __init__(self, H: float = 0.70):
        if not 0 < H < 1:
            raise ValueError(f"H must be in (0,1), got {H}")
        self.H = H
        self._cholesky_cache: Dict[int, np.ndarray] = {}
    
    def covariance_matrix(self, n: int) -> np.ndarray:
        H2 = 2 * self.H
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        k = np.abs(i_idx - j_idx)
        return 0.5 * (np.abs(k + 1)**H2 - 2 * np.abs(k)**H2 + np.abs(k - 1)**H2)
    
    def _get_cholesky(self, n: int) -> np.ndarray:
        if n not in self._cholesky_cache:
            cov = self.covariance_matrix(n)
            cov += 1e-10 * np.eye(n)
            self._cholesky_cache[n] = cholesky(cov, lower=True)
        return self._cholesky_cache[n]
    
    def generate_increments(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        L = self._get_cholesky(n)
        return L @ z
    
    def generate_path(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        increments = self.generate_increments(n, seed)
        path = np.zeros(n + 1)
        path[1:] = np.cumsum(increments)
        return path


# =============================================================================
# MULTIFRACTAL STOCHASTIC VOLATILITY
# =============================================================================

class MultifractalVolatility:
    def __init__(self, sigma0: float = 0.012, lambda_sq: float = 0.04, L: float = 252.0):
        self.sigma0 = sigma0
        self.lambda_sq = lambda_sq
        self.L = L
    
    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        tau = np.abs(i_idx - j_idx).astype(float)
        tau[tau == 0] = 0.5
        
        cov_omega = self.lambda_sq * np.log(np.maximum(self.L / tau, 1.0))
        cov_omega += 1e-8 * np.eye(n)
        
        L_chol = cholesky(cov_omega, lower=True)
        z = rng.standard_normal(n)
        omega = L_chol @ z
        
        variance_omega = self.lambda_sq * np.log(self.L / 0.5)
        omega -= variance_omega / 2
        
        return self.sigma0 * np.exp(omega)


# =============================================================================
# FSDE MODEL
# =============================================================================

class FSDEModel:
    def __init__(self, params: Optional[FSDEParameters] = None):
        if params is None:
            params = FSDEParameters()
        self.params = params
        self.fbm = FractionalBrownianMotion(H=params.H)
        self.volatility = MultifractalVolatility(
            sigma0=params.sigma0,
            lambda_sq=params.lambda_sq,
            L=params.L
        )
    
    def simulate(self, x0: float, T: int, dt: float = 1.0,
                 seed: Optional[int] = None,
                 return_components: bool = False) -> Tuple:
        fbm_seed = seed
        vol_seed = seed + 1000 if seed is not None else None
        
        dBH = self.fbm.generate_increments(T, seed=fbm_seed)
        sigma_t = self.volatility.generate(T, seed=vol_seed)
        
        log_prices = np.zeros(T + 1)
        log_prices[0] = np.log(x0)
        
        for i in range(T):
            drift = (self.params.mu - 0.5 * sigma_t[i]**2) * dt
            diffusion = sigma_t[i] * dBH[i]
            log_prices[i + 1] = log_prices[i] + drift + diffusion
        
        prices = np.exp(log_prices)
        returns = np.diff(log_prices)
        
        fbm_path = np.zeros(T + 1)
        fbm_path[1:] = np.cumsum(dBH)
        
        if return_components:
            components = {
                'returns': returns,
                'volatility': sigma_t,
                'fbm_path': fbm_path,
                'fbm_increments': dBH
            }
            return prices, components
        else:
            return prices, returns, sigma_t


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(returns: np.ndarray) -> Dict:
    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'skewness': skew(returns),
        'kurtosis': kurtosis(returns, fisher=True),
        'min': np.min(returns),
        'max': np.max(returns),
        'annualized_return': np.mean(returns) * 252,
        'annualized_volatility': np.std(returns) * np.sqrt(252)
    }


def compute_ccdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_data = np.sort(data)[::-1]
    n = len(sorted_data)
    ccdf = np.arange(1, n + 1) / n
    return sorted_data, ccdf


def hill_estimator(data: np.ndarray, k: int) -> Tuple[float, float]:
    if k >= len(data) or k < 2:
        return np.nan, np.nan
    sorted_data = np.sort(data)[::-1]
    log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
    alpha = k / np.sum(log_ratios)
    se = alpha / np.sqrt(k)
    return alpha, se


def analyze_tail(returns: np.ndarray, tail: str = 'positive',
                 sigma_threshold: float = 2.0) -> Dict:
    std = np.std(returns)
    threshold = sigma_threshold * std
    
    if tail == 'positive':
        tail_data = returns[returns > threshold]
    else:
        tail_data = np.abs(returns[returns < -threshold])
    
    if len(tail_data) < 10:
        return {'error': 'Insufficient tail data'}
    
    x_ccdf, ccdf = compute_ccdf(tail_data)
    n_tail = len(tail_data)
    k_values = np.arange(10, min(n_tail - 5, n_tail // 2), 5)
    
    if len(k_values) < 3:
        k_values = np.arange(5, n_tail - 2, 2)
    
    alpha_values = []
    se_values = []
    for k in k_values:
        alpha, se = hill_estimator(tail_data, k)
        alpha_values.append(alpha)
        se_values.append(se)
    
    alpha_values = np.array(alpha_values)
    se_values = np.array(se_values)
    
    valid_mask = ~np.isnan(alpha_values)
    if np.sum(valid_mask) > 0:
        if len(alpha_values[valid_mask]) > 5:
            grad = np.abs(np.gradient(alpha_values[valid_mask]))
            smooth_grad = np.convolve(grad, np.ones(3)/3, mode='same')
            opt_idx = np.argmin(smooth_grad[2:-2]) + 2
            optimal_k = k_values[valid_mask][opt_idx]
        else:
            optimal_k = k_values[valid_mask][len(k_values[valid_mask])//2]
        hill_alpha, hill_se = hill_estimator(tail_data, optimal_k)
    else:
        hill_alpha, hill_se, optimal_k = np.nan, np.nan, 0
    
    mask = (ccdf > 1e-4) & (ccdf < 0.5)
    if np.sum(mask) > 5:
        log_x = np.log(x_ccdf[mask])
        log_ccdf = np.log(ccdf[mask])
        slope, intercept, r_value, _, std_err = linregress(log_x, log_ccdf)
        ols_alpha = -slope
        ols_R2 = r_value**2
    else:
        ols_alpha, ols_R2 = np.nan, np.nan
    
    return {
        'x': x_ccdf,
        'ccdf': ccdf,
        'threshold': threshold,
        'n_tail': n_tail,
        'hill_alpha': hill_alpha,
        'hill_se': hill_se,
        'optimal_k': optimal_k,
        'ols_alpha': ols_alpha,
        'ols_R2': ols_R2,
        'k_values': k_values,
        'alpha_values': alpha_values,
        'se_values': se_values
    }


def dfa(data: np.ndarray, n_min: int = 10, n_max: Optional[int] = None,
        n_points: int = 20, order: int = 1) -> DFAResult:
    N = len(data)
    if n_max is None:
        n_max = N // 4
    
    scales = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int))
    profile = np.cumsum(data - np.mean(data))
    fluctuations = np.zeros(len(scales))
    
    for i, n in enumerate(scales):
        n_segments = N // n
        if n_segments < 2:
            fluctuations[i] = np.nan
            continue
        
        F2_sum = 0.0
        count = 0
        
        for seg in range(n_segments):
            start = seg * n
            end = start + n
            segment = profile[start:end]
            
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            residuals = segment - trend
            
            F2_sum += np.mean(residuals**2)
            count += 1
        
        fluctuations[i] = np.sqrt(F2_sum / count) if count > 0 else np.nan
    
    valid = ~np.isnan(fluctuations)
    log_scales = np.log10(scales[valid])
    log_fluct = np.log10(fluctuations[valid])
    
    slope, intercept, r_value, _, std_err = linregress(log_scales, log_fluct)
    
    return DFAResult(
        scales=scales,
        fluctuations=fluctuations,
        H=slope,
        H_se=std_err,
        R2=r_value**2,
        intercept=intercept
    )


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_market_data(ticker: str, name: str, start: str = '2000-01-01',
                      end: str = '2024-12-31') -> Optional[MarketData]:
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) < 100:
            return None
        
        # Handle MultiIndex columns (yfinance >= 0.2.40)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Try 'Adj Close' first, fall back to 'Close'
        if 'Adj Close' in data.columns:
            prices = data['Adj Close'].values
        elif 'Close' in data.columns:
            prices = data['Close'].values
        else:
            raise ValueError(f"No price column found. Columns: {data.columns.tolist()}")
        
        returns = np.diff(np.log(prices))
        dates = data.index[1:].values
        return MarketData(ticker=ticker, name=name, prices=prices,
                          returns=returns, dates=dates)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_figure1_fbm(fbm_path: np.ndarray, figsize: Tuple = (15, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    n = len(fbm_path)
    ax.plot(np.arange(n), fbm_path, color=COLORS['fbm'], linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$B_H(t)$', fontsize=14)
    ax.set_title('Fractional Brownian Motion', fontsize=15, fontweight='bold')
    ax.set_xlim([0, n - 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_figure2_volatility(sigma_t: np.ndarray, figsize: Tuple = (15, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    n = len(sigma_t)
    mean_vol = np.mean(sigma_t)
    ax.plot(np.arange(n), sigma_t, color=COLORS['volatility'], linewidth=0.8)
    ax.axhline(y=mean_vol, color='black', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_vol:.4f}')
    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$\sigma_I(t)$', fontsize=14)
    ax.set_title('Multifractal Stochastic Volatility', fontsize=15, fontweight='bold')
    ax.set_xlim([0, n - 1])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_figure3_prices(prices: np.ndarray, figsize: Tuple = (15, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    n = len(prices)
    ax.plot(np.arange(n), prices, color=COLORS['price'], linewidth=0.8)
    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$x_I(t)$', fontsize=14)
    ax.set_title('Simulated Price Series', fontsize=15, fontweight='bold')
    ax.set_xlim([0, n - 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_figure4_returns(returns: np.ndarray, figsize: Tuple = (15, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    n = len(returns)
    ax.plot(np.arange(n), returns, color=COLORS['returns'], linewidth=0.5, alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$r(t)$', fontsize=14)
    ax.set_title('Simulated Log-Returns Series', fontsize=15, fontweight='bold')
    ax.set_xlim([0, n - 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_figure5_ccdf_simulated(returns: np.ndarray,
                                figsize: Tuple = (14, 5.5)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    tail_configs = [
        ('positive', '(a) Positive Returns', COLORS['positive_tail'], 
         COLORS['fit_positive'], r'$P(R > r)$'),
        ('negative', '(b) Negative Returns', COLORS['negative_tail'],
         COLORS['fit_negative'], r'$P(|R| > r)$')
    ]
    
    for col, (tail, title, scatter_color, fit_color, ylabel) in enumerate(tail_configs):
        ax = axes[col]
        result = analyze_tail(returns, tail=tail, sigma_threshold=2.0)
        x_data, ccdf_data = result['x'], result['ccdf']
        
        ax.scatter(x_data, ccdf_data, s=15, alpha=0.5, color=scatter_color,
                   edgecolors='none', label='Empirical CCDF', zorder=2)
        
        mask = (ccdf_data > 1e-4) & (ccdf_data < 0.5)
        if np.sum(mask) > 5:
            log_x, log_y = np.log(x_data[mask]), np.log(ccdf_data[mask])
            slope, intercept, _, _, _ = linregress(log_x, log_y)
            alpha_fit = -slope
            C_fit = np.exp(intercept)
            x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            ax.plot(x_fit, C_fit * x_fit**(-alpha_fit), '-', color=fit_color,
                    linewidth=2.5, label=f'Fit: α = {alpha_fit:.2f}', zorder=3)
        
        mid_idx = len(x_data) // 2
        C_ref = ccdf_data[mid_idx] * (x_data[mid_idx] ** 3)
        x_ref = np.logspace(np.log10(result['threshold']), np.log10(x_data.max()), 50)
        ax.plot(x_ref, C_ref * x_ref**(-3), '--', color=COLORS['alpha_3'],
                linewidth=2, label=r'$\alpha = 3$', zorder=1)
        
        ax.axvline(result['threshold'], color='gray', linestyle=':', linewidth=1.5, label='Threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|r|$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
    
    fig.suptitle('Tail Distribution of Log-Returns (CCDF)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_figure6_sensitivity(lambda_values: List[float], kurtosis_values: List[float],
                             volatility_values: List[float], figsize: Tuple = (14, 5.5)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    ax1.plot(lambda_values, kurtosis_values, 'bo-', linewidth=2, markersize=10)
    ax1.axhline(y=3, color='r', linestyle='--', linewidth=1.5, label=r'Target ($\kappa = 3$)')
    ax1.fill_between(lambda_values, 3, max(kurtosis_values) + 5, alpha=0.15, color='green')
    ax1.set_xlabel(r'$\lambda^2$', fontsize=14)
    ax1.set_ylabel('Excess Kurtosis', fontsize=14)
    ax1.set_title('Kurtosis vs Intermittency', fontsize=14, fontweight='bold')
    ax1.legend(loc='center right', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(lambda_values, volatility_values, 'go-', linewidth=2, markersize=10)
    ax2.axhline(y=20, color='r', linestyle='--', linewidth=1.5, label=r'Typical market ($\sim$20%)')
    ax2.set_xlabel(r'$\lambda^2$', fontsize=14)
    ax2.set_ylabel('Annualized Volatility (%)', fontsize=14)
    ax2.set_title('Volatility vs Intermittency', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_figure7_dfa_simulated(dfa_returns: DFAResult, dfa_volatility: DFAResult,
                               figsize: Tuple = (14, 5.5)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    valid = ~np.isnan(dfa_returns.fluctuations)
    ax1.scatter(dfa_returns.scales[valid], dfa_returns.fluctuations[valid],
                color=COLORS['dfa_returns'], s=60, alpha=0.8, zorder=3)
    
    n_fit = np.logspace(np.log10(dfa_returns.scales[valid][0]),
                        np.log10(dfa_returns.scales[valid][-1]), 100)
    F_fit = (10**dfa_returns.intercept) * n_fit**dfa_returns.H
    ax1.plot(n_fit, F_fit, 'k--', linewidth=2, label=f'$H = {dfa_returns.H:.3f}$')
    
    F_ref = dfa_returns.fluctuations[valid][0] * (n_fit / dfa_returns.scales[valid][0])**0.5
    ax1.plot(n_fit, F_ref, ':', color=COLORS['random_walk'], linewidth=1.5,
             label=r'$H = 0.5$ (random walk)')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$n$', fontsize=14)
    ax1.set_ylabel(r'$F(n)$', fontsize=14)
    ax1.set_title('(a) Returns', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    valid_v = ~np.isnan(dfa_volatility.fluctuations)
    ax2.scatter(dfa_volatility.scales[valid_v], dfa_volatility.fluctuations[valid_v],
                color=COLORS['dfa_volatility'], s=60, alpha=0.8, marker='s', zorder=3)
    
    n_fit_v = np.logspace(np.log10(dfa_volatility.scales[valid_v][0]),
                          np.log10(dfa_volatility.scales[valid_v][-1]), 100)
    F_fit_v = (10**dfa_volatility.intercept) * n_fit_v**dfa_volatility.H
    ax2.plot(n_fit_v, F_fit_v, 'k--', linewidth=2, label=f'$H = {dfa_volatility.H:.3f}$')
    
    F_ref_v = dfa_volatility.fluctuations[valid_v][0] * (n_fit_v / dfa_volatility.scales[valid_v][0])**0.5
    ax2.plot(n_fit_v, F_ref_v, ':', color=COLORS['random_walk'], linewidth=1.5,
             label=r'$H = 0.5$ (random walk)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$n$', fontsize=14)
    ax2.set_ylabel(r'$F(n)$', fontsize=14)
    ax2.set_title('(b) Volatility', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Detrended Fluctuation Analysis (DFA)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_figure_ccdf_empirical(market: MarketData, figsize: Tuple = (14, 5.5)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    returns = market.returns
    
    tail_configs = [
        ('positive', '(a) Positive Returns', '#6495ed', r'$P(R > r)$'),
        ('negative', '(b) Negative Returns', '#cd5c5c', r'$P(|R| > r)$')
    ]
    
    for col, (tail, title, scatter_color, ylabel) in enumerate(tail_configs):
        ax = axes[col]
        result = analyze_tail(returns, tail=tail, sigma_threshold=2.0)
        x_data, ccdf_data = result['x'], result['ccdf']
        
        ax.scatter(x_data, ccdf_data, s=50, alpha=0.3, color=scatter_color,
                   edgecolors='none', label='Empirical CCDF', zorder=2)
        
        mask = (ccdf_data > 1e-4) & (ccdf_data < 0.5)
        if np.sum(mask) > 5:
            log_x, log_y = np.log(x_data[mask]), np.log(ccdf_data[mask])
            slope, intercept, _, _, _ = linregress(log_x, log_y)
            alpha_fit = -slope
            C_fit = np.exp(intercept)
            x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
            ax.plot(x_fit, C_fit * x_fit**(-alpha_fit), '-', color='black',
                    linewidth=1.5, label=f'Fit: α = {alpha_fit:.2f}', zorder=3)
        
        mid_idx = len(x_data) // 2
        C_ref = ccdf_data[mid_idx] * (x_data[mid_idx] ** 3)
        x_ref = np.logspace(np.log10(result['threshold']), np.log10(x_data.max()), 50)
        ax.plot(x_ref, C_ref * x_ref**(-3), '--', color=COLORS['alpha_3'],
                linewidth=2, label=r'$\alpha = 3$', zorder=1)
        
        ax.axvline(result['threshold'], color='gray', linestyle=':', linewidth=1.5, label='Threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|r|$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='none', fontsize=10)
        ax.set_ylim(1e-4, 2)
    
    fig.suptitle(f'Tail Distribution of Log-Returns (CCDF) — {market.name}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_figure_dfa_empirical(market: MarketData, figsize: Tuple = (14, 5.5)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    returns = market.returns
    abs_returns = np.abs(returns)
    
    dfa_ret = dfa(returns, n_min=10, n_max=len(returns)//4, n_points=20)
    dfa_vol = dfa(abs_returns, n_min=10, n_max=len(abs_returns)//4, n_points=20)
    
    ax1 = axes[0]
    valid = ~np.isnan(dfa_ret.fluctuations)
    ax1.scatter(dfa_ret.scales[valid], dfa_ret.fluctuations[valid],
                color=COLORS['dfa_returns'], s=60, alpha=0.8, zorder=3)
    
    n_fit = np.logspace(np.log10(dfa_ret.scales[valid][0]),
                        np.log10(dfa_ret.scales[valid][-1]), 100)
    F_fit = (10**dfa_ret.intercept) * n_fit**dfa_ret.H
    ax1.plot(n_fit, F_fit, 'k--', linewidth=2, label=f'$H = {dfa_ret.H:.3f}$')
    
    F_ref = dfa_ret.fluctuations[valid][0] * (n_fit / dfa_ret.scales[valid][0])**0.5
    ax1.plot(n_fit, F_ref, ':', color=COLORS['random_walk'], linewidth=1.5,
             label=r'$H = 0.5$ (random walk)')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$n$', fontsize=14)
    ax1.set_ylabel(r'$F(n)$', fontsize=14)
    ax1.set_title('(a) Returns', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    valid_v = ~np.isnan(dfa_vol.fluctuations)
    ax2.scatter(dfa_vol.scales[valid_v], dfa_vol.fluctuations[valid_v],
                color=COLORS['dfa_volatility'], s=60, alpha=0.8, marker='s', zorder=3)
    
    n_fit_v = np.logspace(np.log10(dfa_vol.scales[valid_v][0]),
                          np.log10(dfa_vol.scales[valid_v][-1]), 100)
    F_fit_v = (10**dfa_vol.intercept) * n_fit_v**dfa_vol.H
    ax2.plot(n_fit_v, F_fit_v, 'k--', linewidth=2, label=f'$H = {dfa_vol.H:.3f}$')
    
    F_ref_v = dfa_vol.fluctuations[valid_v][0] * (n_fit_v / dfa_vol.scales[valid_v][0])**0.5
    ax2.plot(n_fit_v, F_ref_v, ':', color=COLORS['random_walk'], linewidth=1.5,
             label=r'$H = 0.5$ (random walk)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$n$', fontsize=14)
    ax2.set_ylabel(r'$F(n)$', fontsize=14)
    ax2.set_title('(b) Volatility', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Detrended Fluctuation Analysis (DFA) — {market.name}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# RUN FUNCTIONS
# =============================================================================

def run_simulation_figures(params: FSDEParameters, x0: float = 100.0,
                           n_steps: int = 2000, seed: int = SEED) -> Dict:
    model = FSDEModel(params)
    prices, components = model.simulate(x0, n_steps, seed=seed, return_components=True)
    
    return {
        'prices': prices,
        'returns': components['returns'],
        'volatility': components['volatility'],
        'fbm_path': components['fbm_path'],
        'params': params,
        'fig1': plot_figure1_fbm(components['fbm_path']),
        'fig2': plot_figure2_volatility(components['volatility']),
        'fig3': plot_figure3_prices(prices),
        'fig4': plot_figure4_returns(components['returns'])
    }


def run_sensitivity_analysis(base_params: FSDEParameters,
                             lambda_values: List[float] = None,
                             x0: float = 100.0, n_steps: int = 2000,
                             n_simulations: int = 5, seed: int = SEED) -> Dict:
    if lambda_values is None:
        lambda_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    
    kurtosis_vals = []
    volatility_vals = []
    
    for lsq in lambda_values:
        params = FSDEParameters(H=base_params.H, lambda_sq=lsq, sigma0=base_params.sigma0,
                                mu=base_params.mu, L=base_params.L)
        model = FSDEModel(params)
        
        kurt_list, vol_list = [], []
        for i in range(n_simulations):
            _, returns, _ = model.simulate(x0, n_steps, seed=seed + i * 100)
            stats = compute_statistics(returns)
            kurt_list.append(stats['kurtosis'])
            vol_list.append(stats['annualized_volatility'] * 100)
        
        kurtosis_vals.append(np.mean(kurt_list))
        volatility_vals.append(np.mean(vol_list))
    
    return {
        'lambda_values': lambda_values,
        'kurtosis': kurtosis_vals,
        'volatility': volatility_vals,
        'figure': plot_figure6_sensitivity(lambda_values, kurtosis_vals, volatility_vals)
    }


def run_ccdf_analysis(params: FSDEParameters, n_realizations: int = 20,
                      T: int = 2520, x0: float = 100.0, seed: int = SEED) -> Dict:
    model = FSDEModel(params)
    all_returns = []
    for i in range(n_realizations):
        _, returns, _ = model.simulate(x0, T, seed=seed + i)
        all_returns.append(returns)
    pooled_returns = np.concatenate(all_returns)
    
    return {
        'pooled_returns': pooled_returns,
        'positive_tail': analyze_tail(pooled_returns, 'positive'),
        'negative_tail': analyze_tail(pooled_returns, 'negative'),
        'n_samples': len(pooled_returns),
        'figure': plot_figure5_ccdf_simulated(pooled_returns)
    }


def run_dfa_analysis(params: FSDEParameters, n_realizations: int = 30,
                     T: int = 2520, x0: float = 100.0, seed: int = SEED) -> Dict:
    model = FSDEModel(params)
    all_returns = []
    for i in range(n_realizations):
        _, returns, _ = model.simulate(x0, T, seed=seed + i)
        all_returns.append(returns)
    
    pooled_returns = np.concatenate(all_returns)
    abs_returns = np.abs(pooled_returns)
    
    dfa_returns = dfa(pooled_returns, n_min=10, n_max=len(pooled_returns)//10, n_points=25)
    dfa_volatility = dfa(abs_returns, n_min=10, n_max=len(abs_returns)//10, n_points=25)
    
    return {
        'dfa_returns': dfa_returns,
        'dfa_volatility': dfa_volatility,
        'n_samples': len(pooled_returns),
        'figure': plot_figure7_dfa_simulated(dfa_returns, dfa_volatility)
    }


def run_empirical_validation(markets: Dict[str, MarketData]) -> Dict:
    results = {}
    figures = {}
    
    for key, market in markets.items():
        if market is None:
            continue
        
        pos_tail = analyze_tail(market.returns, 'positive')
        neg_tail = analyze_tail(market.returns, 'negative')
        dfa_ret = dfa(market.returns, n_min=10, n_max=len(market.returns)//4, n_points=20)
        dfa_vol = dfa(np.abs(market.returns), n_min=10, n_max=len(market.returns)//4, n_points=20)
        
        results[key] = {
            'market': market,
            'positive_tail': pos_tail,
            'negative_tail': neg_tail,
            'dfa_returns': dfa_ret,
            'dfa_volatility': dfa_vol
        }
        
        figures[f'{key}_ccdf'] = plot_figure_ccdf_empirical(market)
        figures[f'{key}_dfa'] = plot_figure_dfa_empirical(market)
    
    return {'results': results, 'figures': figures}


# =============================================================================
# MAIN
# =============================================================================

def main(run_phases: List[int] = None, seed: int = SEED, show_figures: bool = True) -> Dict:
    if run_phases is None:
        run_phases = [1, 2, 3, 4, 5, 6]
    
    all_results = {}
    
    if 1 in run_phases:
        print("Phase 1")
        params = FSDEParameters(H=0.65, lambda_sq=0.04, sigma0=0.012, mu=0.00005, L=252.0)
        all_results['simulation'] = run_simulation_figures(params, x0=100.0, n_steps=2000, seed=seed)
        if show_figures:
            plt.show()
    
    if 2 in run_phases:
        print("Phase 2")
        base_params = FSDEParameters(H=0.65, lambda_sq=0.04, sigma0=0.01, mu=0.00005, L=252.0)
        all_results['sensitivity'] = run_sensitivity_analysis(base_params, seed=seed)
        if show_figures:
            plt.show()
    
    if 3 in run_phases:
        print("Phase 3")
        params_ccdf = FSDEParameters(H=0.65, lambda_sq=0.10, sigma0=0.012, mu=0.00005, L=252.0)
        all_results['ccdf'] = run_ccdf_analysis(params_ccdf, n_realizations=20, T=2520, seed=seed)
        if show_figures:
            plt.show()
    
    if 4 in run_phases:
        print("Phase 4")
        params_dfa = FSDEParameters(H=0.65, lambda_sq=0.10, sigma0=0.012, mu=0.00005, L=252.0)
        all_results['dfa'] = run_dfa_analysis(params_dfa, n_realizations=30, T=2520, seed=seed)
        if show_figures:
            plt.show()
    
    if 5 in run_phases or 6 in run_phases:
        print("Phase 5-6")
        sp500 = fetch_market_data('^GSPC', 'S&P 500', '2000-01-01', '2024-12-31')
        ibovespa = fetch_market_data('^BVSP', 'Ibovespa', '2000-01-01', '2024-12-31')
        markets = {'sp500': sp500, 'ibovespa': ibovespa}
        all_results['empirical'] = run_empirical_validation(markets)
        if show_figures:
            plt.show()
    
    print("Finish")
    return all_results


if __name__ == "__main__":
    results = main(run_phases=[1, 2, 3, 4, 5, 6], seed=42, show_figures=True)
