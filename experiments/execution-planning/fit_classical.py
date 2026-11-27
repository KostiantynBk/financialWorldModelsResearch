"""Fit the hand-specified linear-Gaussian market model to data by maximum likelihood.

The model's form is fixed (trend with geometric fade, noisy signal, random-walk
price); its parameters are estimated from training episodes. The cost model
(spread, impact) is assumed and copied from `base`.
"""

import numpy as np
from scipy.optimize import minimize

from market import MarketParams, Paths


def _update(m, p, y, r):
    s = p + r
    innovation = y - m
    loglik = -0.5 * (y.size * np.log(2.0 * np.pi * s) + float((innovation**2).sum()) / s)
    gain = p / s
    return m + gain * innovation, (1.0 - gain) * p, loglik


def log_likelihood(params: MarketParams, paths: Paths) -> float:
    """Exact Gaussian log-likelihood of all observed signals and price changes.

    Uses the Kalman filter's prediction-error decomposition, processing each
    minute's price change and signal as sequential scalar observations.
    """
    n_episodes, n = paths.signal.shape
    m, p = np.zeros(n_episodes), params.alpha_std**2
    m, p, total = _update(m, p, paths.signal[:, 0], params.obs_std**2)
    for t in range(1, n):
        m, p, ll = _update(m, p, paths.price[:, t] - paths.price[:, t - 1], params.sigma**2)
        total += ll
        m, p = params.phi * m, params.phi**2 * p + params.alpha_innov_std**2
        m, p, ll = _update(m, p, paths.signal[:, t], params.obs_std**2)
        total += ll
    return total


def _to_params(theta, base: MarketParams) -> MarketParams:
    sigma, alpha_std, obs_std = np.exp(theta[:3])
    phi = 1.0 / (1.0 + np.exp(-theta[3]))
    return base.with_(
        sigma=float(sigma),
        r2=float(alpha_std**2 / (alpha_std**2 + sigma**2)),
        half_life=float(np.log(0.5) / np.log(phi)),
        obs_noise=float(obs_std / alpha_std),
    )


def _starting_values(paths: Paths) -> np.ndarray:
    """Moment-based starting point from the signal's autocovariances."""
    z = paths.signal - paths.signal.mean()
    c0 = float((z * z).mean())
    c1 = float((z[:, 1:] * z[:, :-1]).mean())
    c2 = float((z[:, 2:] * z[:, :-2]).mean())
    phi = float(np.clip(c2 / c1 if c1 > 0 else 0.5, 0.05, 0.99))
    alpha_var = float(np.clip(c1 / phi if c1 > 0 else 0.5 * c0, 0.05 * c0, 0.95 * c0))
    sigma = float(np.diff(paths.price, axis=1).std())
    obs_var = c0 - alpha_var
    return np.array([np.log(sigma), 0.5 * np.log(alpha_var), 0.5 * np.log(obs_var), np.log(phi / (1 - phi))])


def fit_linear_gaussian(paths: Paths, base: MarketParams) -> tuple[MarketParams, dict]:
    """Maximum-likelihood estimates of sigma, trend size, fade rate and signal noise."""
    n_obs = paths.signal.size

    def objective(theta):
        return -log_likelihood(_to_params(theta, base), paths) / n_obs

    result = minimize(objective, _starting_values(paths), method="L-BFGS-B")
    return _to_params(result.x, base), {"converged": bool(result.success), "neg_loglik_per_obs": float(result.fun)}
