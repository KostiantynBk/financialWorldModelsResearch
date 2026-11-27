"""A small learned world model for the execution experiment.

It reads the market one minute at a time (price change and signal), keeps its own
learned internal state, and predicts future observations by rolling a learned
latent transition forward:

    h_t        = GRU(h_{t-1}, observation_t)          state built from data up to t
    s_0        = W h_t                                 latent state at decision time
    s_{j+1}    = s_j + MLP(s_j)                        learned latent dynamics
    prediction = D s_j  ->  (price change, signal) at t + 1 + j

It is trained only on observations (self-supervised); it is given none of the
simulator's equations or parameters. One structural assumption is imposed, which
the classical model shares: sign symmetry. Mirroring every price change and
signal mirrors the forecast, so the market has no drift on average. It is
enforced by averaging the network's forecast with the negated forecast for the
mirrored inputs.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from market import Paths


class LatentWorldModel(nn.Module):
    def __init__(self, hidden: int = 16, latent: int = 8, width: int = 32):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.GRU(2, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.transition = nn.Sequential(nn.Linear(latent, width), nn.Tanh(), nn.Linear(width, latent))
        self.decoder = nn.Linear(latent, 2)

    def rollout(self, h: torch.Tensor, steps: int) -> torch.Tensor:
        """h (B, hidden) -> predicted observations 1..steps ahead, shape (B, steps, 2)."""
        if steps == 0:
            return h.new_zeros(h.shape[0], 0, 2)
        s = self.to_latent(h)
        preds = []
        for _ in range(steps):
            preds.append(self.decoder(s))
            s = s + self.transition(s)
        return torch.stack(preds, dim=1)

    def predict(self, obs: torch.Tensor, times: torch.Tensor, steps: int) -> torch.Tensor:
        """Sign-symmetric forecasts from decision times `times`: (B, len(times), steps, 2).

        The state at time t is built from observations 0..t only (the GRU is causal).
        """
        b = obs.shape[0]
        states, _ = self.encoder(torch.cat([obs, -obs]))
        chosen = states[:, times].reshape(-1, self.hidden)
        preds = self.rollout(chosen, steps).reshape(2 * b, len(times), steps, 2)
        return 0.5 * (preds[:b] - preds[b:])


@dataclass(frozen=True)
class Scales:
    price_change: float
    signal: float


def _observations(paths: Paths, scales: Scales) -> np.ndarray:
    price_change = np.zeros_like(paths.price)
    price_change[:, 1:] = np.diff(paths.price, axis=1)
    return np.stack([price_change / scales.price_change, paths.signal / scales.signal], axis=-1)


def train_world_model(paths: Paths, seed: int, max_steps: int = 3000, batch_size: int = 64,
                      times_per_batch: int = 8, lr: float = 3e-3, holdout: float = 0.2,
                      eval_every: int = 25, patience: int = 12):
    """Train on the given episodes, holding out 20% for early stopping on the price forecast.

    Each step uses a random batch of episodes and a random sample of decision times.
    Returns (model, scales, info).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_episodes, n = paths.price.shape
    scales = Scales(price_change=float(np.diff(paths.price, axis=1).std()), signal=float(paths.signal.std()))
    obs = torch.tensor(_observations(paths, scales), dtype=torch.float32)
    order = rng.permutation(n_episodes)
    n_hold = max(1, int(round(holdout * n_episodes)))
    hold, fit = obs[order[:n_hold]], obs[order[n_hold:]]

    horizon = n - 1
    all_times = torch.arange(n - 1)
    target_time = all_times[:, None] + 1 + torch.arange(horizon)[None, :]
    mask = (target_time <= n - 1).float()
    target_time = target_time.clamp(max=n - 1)

    def errors(batch, times):
        preds = model.predict(batch, times, horizon)
        sq = (preds - batch[:, target_time[times]]) ** 2  # (B, T, H, 2)
        return sq, mask[times][None, :, :, None]

    model = LatentWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, since_best, step = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    while step < max_steps and since_best < patience:
        batch = fit[rng.choice(len(fit), size=min(batch_size, len(fit)), replace=False)]
        times = torch.tensor(np.sort(rng.choice(n - 1, size=min(times_per_batch, n - 1), replace=False)))
        sq, m = errors(batch, times)
        loss = (sq * m).sum() / (m.sum() * batch.shape[0] * 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % eval_every == 0:
            with torch.no_grad():
                sq, m = errors(hold, all_times)
                price_loss = float((sq[..., :1] * m).sum() / (m.sum() * hold.shape[0]))
            if price_loss < best - 1e-6:
                best, best_state, since_best = price_loss, copy.deepcopy(model.state_dict()), 0
            else:
                since_best += 1
    model.load_state_dict(best_state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, scales, {"steps": step, "holdout_price_loss": best, "parameters": n_params}


class LearnedForecaster:
    """Predictor interface for the planner: expected price change in each remaining step."""

    def __init__(self, model: LatentWorldModel, scales: Scales, n_steps: int):
        self.model, self.scales, self.n = model, scales, n_steps

    def reset(self, n_episodes: int) -> None:
        self.e = n_episodes
        self.h = torch.zeros(1, 2 * n_episodes, self.model.hidden)
        self.prev_price = None

    def update(self, t, price_t, signal_t):
        change = np.zeros_like(price_t) if self.prev_price is None else price_t - self.prev_price
        self.prev_price = price_t
        x = torch.tensor(np.stack([change / self.scales.price_change, signal_t / self.scales.signal], axis=-1),
                         dtype=torch.float32)
        with torch.no_grad():
            out, self.h = self.model.encoder(torch.cat([x, -x])[:, None, :], self.h)
            preds = self.model.rollout(out[:, 0], self.n - 1 - t)
            preds = 0.5 * (preds[: self.e] - preds[self.e:])
        return preds[..., 0].double().numpy() * self.scales.price_change
