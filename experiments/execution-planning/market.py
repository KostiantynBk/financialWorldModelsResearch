"""Synthetic single-asset market for executing one buy order.

The market path is exogenous: the agent's trades change its own inventory and
costs, never the price path. Units: prices are in basis points (bps) of the
arrival mid-price, so S_0 = 0; quantities are fractions of the order (the full
order is 1).

Dynamics for decision steps t = 0..N-1:

    S_{t+1}     = S_t + alpha_t + sigma * eps_{t+1}      mid-price
    alpha_{t+1} = phi * alpha_t + sigma_a * eta_{t+1}    hidden drift
    z_t         = alpha_t + sigma_z * xi_t               observed noisy signal

Buying v_t at step t costs v_t * (S_t + half_spread + impact * v_t) in bps.
All shocks are independent standard normals.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class MarketParams:
    n_steps: int = 60  # decision steps; the last step must complete the order
    sigma: float = 5.0  # mid-price noise per step, bps
    half_spread: float = 1.0  # bps paid on every unit traded
    impact: float = 30.0  # temporary impact: trading fraction v in one step costs impact * v bps per unit
    r2: float = 0.01  # share of one-step return variance explained by alpha
    half_life: float = 10.0  # half-life of alpha, in steps
    obs_noise: float = 1.0  # std of the signal's observation noise, relative to alpha's stationary std

    @property
    def phi(self) -> float:
        return 0.5 ** (1.0 / self.half_life)

    @property
    def alpha_std(self) -> float:
        """Stationary standard deviation of alpha (bps per step)."""
        return self.sigma * float(np.sqrt(self.r2 / (1.0 - self.r2)))

    @property
    def alpha_innov_std(self) -> float:
        return self.alpha_std * float(np.sqrt(1.0 - self.phi**2))

    @property
    def obs_std(self) -> float:
        """Observation-noise std of z. With no drift (r2 = 0), z is pure unit noise."""
        return self.obs_noise * self.alpha_std if self.alpha_std > 0 else 1.0

    def with_(self, **changes) -> "MarketParams":
        return replace(self, **changes)


@dataclass(frozen=True)
class Paths:
    alpha: np.ndarray  # (E, N) hidden drift alpha_t: expected price change from t to t+1
    price: np.ndarray  # (E, N) mid-price S_t at decision time t, bps of arrival (S_0 = 0)
    signal: np.ndarray  # (E, N) observed signal z_t


def simulate(params: MarketParams, n_episodes: int, seed: int) -> Paths:
    """Simulate independent episodes.

    The same seed gives the same standard-normal draws for any parameter values,
    so different settings share common random numbers.
    """
    rng = np.random.default_rng(seed)
    n = params.n_steps
    eta = rng.standard_normal((n_episodes, n))
    eps = rng.standard_normal((n_episodes, n))
    xi = rng.standard_normal((n_episodes, n))

    alpha = np.empty((n_episodes, n))
    alpha[:, 0] = params.alpha_std * eta[:, 0]
    for t in range(1, n):
        alpha[:, t] = params.phi * alpha[:, t - 1] + params.alpha_innov_std * eta[:, t]

    price = np.zeros((n_episodes, n))
    price[:, 1:] = np.cumsum(alpha[:, :-1] + params.sigma * eps[:, 1:], axis=1)
    signal = alpha + params.obs_std * xi
    return Paths(alpha=alpha, price=price, signal=signal)


def _kalman_update(m, p, y, r):
    gain = p / (p + r)
    return m + gain * (y - m), (1.0 - gain) * p


class AlphaFilter:
    """Kalman filter for alpha_t given prices S_0..S_t and signals z_0..z_t.

    It is updated one step at a time, so it can only use past and current data.
    The price change S_t - S_{t-1} = alpha_{t-1} + noise is informative about
    alpha_{t-1}; z_t is informative about alpha_t.
    """

    def __init__(self, params: MarketParams):
        self.params = params

    def reset(self, n_episodes: int) -> None:
        self.m = np.zeros(n_episodes)
        self.p = self.params.alpha_std**2
        self.prev_price = None

    def update(self, price_t: np.ndarray, signal_t: np.ndarray) -> np.ndarray:
        prm = self.params
        if prm.alpha_std == 0:
            return np.zeros_like(signal_t)
        m, p = self.m, self.p
        if self.prev_price is not None:
            m, p = _kalman_update(m, p, price_t - self.prev_price, prm.sigma**2)
            m, p = prm.phi * m, prm.phi**2 * p + prm.alpha_innov_std**2
        m, p = _kalman_update(m, p, signal_t, prm.obs_std**2)
        self.m, self.p, self.prev_price = m, p, price_t
        return m


def latest_signal_estimate(params: MarketParams, signal_t: np.ndarray) -> np.ndarray:
    """E[alpha_t | z_t] using only the current signal (no history)."""
    if params.alpha_std == 0:
        return np.zeros_like(signal_t)
    weight = params.alpha_std**2 / (params.alpha_std**2 + params.obs_std**2)
    return weight * signal_t
