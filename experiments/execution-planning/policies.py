"""Execution policies for one buy order.

Each policy is stateful. At step t it sees only the current mid-price, the current
signal, and its remaining quantity (earlier values are available only if the policy
stored them itself). It returns the quantity to buy now. The runner clips this to
[0, remaining] and forces completion at the last step.
"""

import numpy as np

from market import AlphaFilter, MarketParams, latest_signal_estimate


class TWAP:
    """Trade the remaining quantity evenly over the remaining steps."""

    name = "TWAP"

    def __init__(self, params: MarketParams):
        self.n = params.n_steps

    def reset(self, n_episodes: int) -> None:
        pass

    def act(self, t, price_t, signal_t, remaining):
        return remaining / (self.n - t)


class UrgencyRule:
    """HRT's hand-crafted rule, adapted to a buy order.

    If remaining / steps_left > beta * sigma, buy everything left. Otherwise, if the
    signal is positive, buy remaining / steps_left; otherwise wait.
    """

    def __init__(self, params: MarketParams, beta: float):
        self.n = params.n_steps
        self.beta = beta
        self.threshold = beta * params.sigma
        self.name = f"Urgency rule (beta={beta:.3g})"

    def reset(self, n_episodes: int) -> None:
        pass

    def act(self, t, price_t, signal_t, remaining):
        rate = remaining / (self.n - t)
        return np.where(rate > self.threshold, remaining, np.where(signal_t > 0, rate, 0.0))


class LQGReference:
    """Exact optimum when the no-sell constraint does not bind.

    Kalman-filtered drift plus a Riccati recursion for the risk-neutral
    linear-quadratic problem. When the unconstrained action falls outside
    [0, remaining], the runner clips it; `clipped / decisions` reports how often.
    """

    name = "LQG reference"

    def __init__(self, params: MarketParams):
        self.params = params
        n, kappa, phi = params.n_steps, params.impact, params.phi
        self.gain_k = np.empty(n)
        self.gain_b = np.empty(n)
        a_next, b_next = kappa, 0.0  # value function V_{N-1}(q, m) = kappa * q^2
        for t in range(n - 2, -1, -1):
            k = kappa + a_next
            b = 1.0 + phi * b_next
            self.gain_k[t], self.gain_b[t] = k, b
            a_next, b_next = kappa * a_next / k, kappa * b / k
        self.filter = AlphaFilter(params)

    def reset(self, n_episodes: int) -> None:
        self.filter.reset(n_episodes)
        self.clipped = 0
        self.decisions = 0

    def act(self, t, price_t, signal_t, remaining):
        m = self.filter.update(price_t, signal_t)
        if t == self.params.n_steps - 1:
            return remaining
        keep = (self.params.impact * remaining - 0.5 * self.gain_b[t] * m) / self.gain_k[t]
        v = remaining - keep
        active = remaining > 1e-12
        self.clipped += int(np.count_nonzero(active & ((v < 0) | (v > remaining))))
        self.decisions += int(np.count_nonzero(active))
        return v


def project_to_simplex(u: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row of u onto {v >= 0, sum(v) = total_row}."""
    n_rows, h = u.shape
    s = -np.sort(-u, axis=1)
    shifted = np.cumsum(s, axis=1) - total[:, None]
    j = np.arange(1, h + 1)
    positive = s - shifted / j > 0
    rho = h - np.argmax(positive[:, ::-1], axis=1)
    theta = shifted[np.arange(n_rows), rho - 1] / rho
    v = np.maximum(u - theta[:, None], 0.0)
    return np.where(total[:, None] > 0, v, 0.0)


class Planner:
    """Receding-horizon planner that uses a market model to forecast drift.

    At each step it forecasts alpha over the remaining horizon, finds the schedule
    with the lowest expected cost, and executes only the first trade. With
    risk-neutral linear-impact costs, expected cost is
        impact * sum(v_j^2) - sum(v_j * G_j) + const,
    where G_j is the forecast drift from step t+j to the deadline. The optimal
    schedule is therefore the projection of G / (2 * impact) onto
    {v >= 0, sum v = remaining}. The model's parameters may differ from the
    environment's.
    """

    def __init__(self, model: MarketParams, use_history: bool = True, name: str | None = None):
        self.model = model
        self.use_history = use_history
        self.name = name or ("Planner" if use_history else "Planner (latest signal only)")
        self.filter = AlphaFilter(model)
        n, phi = model.n_steps, model.phi
        # drift_weights[t][j] = sum_{k=j}^{H-2} phi^k for horizon H = n - t; the last entry is 0.
        self.drift_weights = []
        for t in range(n):
            h = n - t
            powers = phi ** np.arange(h - 1)
            tail = np.cumsum(powers[::-1])[::-1]
            self.drift_weights.append(np.append(tail, 0.0))

    def reset(self, n_episodes: int) -> None:
        self.filter.reset(n_episodes)

    def act(self, t, price_t, signal_t, remaining):
        if self.use_history:
            m = self.filter.update(price_t, signal_t)
        else:
            m = latest_signal_estimate(self.model, signal_t)
        drift_to_go = m[:, None] * self.drift_weights[t][None, :]
        schedule = project_to_simplex(drift_to_go / (2.0 * self.model.impact), remaining)
        return schedule[:, 0]
