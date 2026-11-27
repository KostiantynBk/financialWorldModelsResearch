"""Run policies on simulated paths and compare costs on the same paths."""

from dataclasses import dataclass

import numpy as np

from market import MarketParams, Paths


def run_policy(policy, paths: Paths, params: MarketParams) -> np.ndarray:
    """Return trades (E, N). Each step reveals only that step's price and signal."""
    n_episodes, n = paths.price.shape
    remaining = np.ones(n_episodes)
    trades = np.zeros((n_episodes, n))
    policy.reset(n_episodes)
    for t in range(n):
        v = np.asarray(policy.act(t, paths.price[:, t], paths.signal[:, t], remaining.copy()), dtype=float)
        v = remaining.copy() if t == n - 1 else np.clip(v, 0.0, remaining)
        trades[:, t] = v
        remaining = remaining - v
    return trades


def shortfall_bps(trades: np.ndarray, paths: Paths, params: MarketParams) -> np.ndarray:
    """Implementation shortfall per episode, in bps of arrival notional.

    Costs always use the environment's true parameters.
    """
    return (trades * (paths.price + params.half_spread + params.impact * trades)).sum(axis=1)


@dataclass(frozen=True)
class PairedDiff:
    mean: float
    se: float

    @property
    def low(self) -> float:
        return self.mean - 1.96 * self.se

    @property
    def high(self) -> float:
        return self.mean + 1.96 * self.se


def paired_diff(x: np.ndarray, ref: np.ndarray) -> PairedDiff:
    """Mean of x - ref over the same episodes, with a normal-approximation 95% CI.

    Episodes are independent by construction, so no blocking is needed.
    """
    d = x - ref
    return PairedDiff(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(d.size)))
