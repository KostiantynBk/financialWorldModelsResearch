"""Stage 3: simulated markets that break the classical model's assumptions, and their best forecasters.

Scenarios (prespecified in PLAN.md, "Stage 3 prespecification"):
  A  signal on/off     each hour: informative signal (trend explains 2%) or pure noise, probability 1/2 each
  B  fast or slow      each hour: trend half-life 3 or 30 minutes, probability 1/2 each
  C  signal outliers   10% of signal readings carry 3x larger noise; total noise variance unchanged
  D  price outliers    the same, on price noise (negative control)

Regimes are drawn once per hour and fixed within it. The base shocks are the same standard normals as in
market.simulate (common random numbers); regime and outlier draws come from a separate random stream.

The best forecaster keeps, for each regime, a Gaussian estimate of the trend and that regime's predictive
likelihood, and weights the regimes by their posterior probability. With normal noise (A, B) this is exact
Bayesian filtering. With outlier noise (C, D), each update's two-component posterior is collapsed to one
Gaussian by moment matching (an approximation, checked against a particle filter).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from market import MarketParams, Paths

BASE = MarketParams()


@dataclass(frozen=True)
class Regime:
    alpha_std: float  # stationary standard deviation of the trend, bps per minute; 0 = no trend
    phi: float  # trend persistence per minute
    signal_noise: tuple  # mixture ((weight, std), ...) of the signal's observation noise

    @property
    def alpha_innov_std(self) -> float:
        return self.alpha_std * float(np.sqrt(1.0 - self.phi**2))


@dataclass(frozen=True)
class Scenario:
    key: str
    regimes: tuple  # (Regime, ...)
    probs: tuple  # probability of each regime in an hour
    price_noise: tuple  # mixture ((weight, std), ...) of the price noise per minute
    labels: tuple = ("all hours",)
    severe: bool = False


def phi_from_half_life(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life)


def trend_std(r2: float, sigma: float = BASE.sigma) -> float:
    """Trend standard deviation that explains a share r2 of one-minute price-change variance."""
    return sigma * float(np.sqrt(r2 / (1.0 - r2)))


def normal(std: float) -> tuple:
    return ((1.0, std),)


def outlier_mix(std: float, prob: float, multiple: float) -> tuple:
    """Noise that is usually small and sometimes `multiple` times larger, with total standard deviation `std`."""
    small = std / float(np.sqrt((1.0 - prob) + prob * multiple**2))
    return ((1.0 - prob, small), (prob, multiple * small))


def scenario(key: str, severe: bool = False) -> Scenario:
    a, sigma, phi10 = BASE.alpha_std, BASE.sigma, BASE.phi
    if key == "base":  # the Stage 1-2 market, for checks
        return Scenario("base", (Regime(a, phi10, normal(BASE.obs_std)),), (1.0,), normal(sigma))
    if key == "A":
        p_active, r2 = (0.3, 0.033) if severe else (0.5, 0.02)
        a_on = trend_std(r2)
        active = Regime(a_on, phi10, normal(a_on))
        quiet = Regime(0.0, phi10, normal(float(np.sqrt(2.0)) * a_on))  # same total signal variance as active
        return Scenario("A", (active, quiet), (p_active, 1.0 - p_active), normal(sigma), ("active", "quiet"), severe)
    if key == "B":
        fast, slow = (2.0, 60.0) if severe else (3.0, 30.0)
        regimes = (Regime(a, phi_from_half_life(fast), normal(a)), Regime(a, phi_from_half_life(slow), normal(a)))
        return Scenario("B", regimes, (0.5, 0.5), normal(sigma), (f"half-life {fast:g}", f"half-life {slow:g}"), severe)
    if key == "C":
        prob, multiple = (0.05, 5.0) if severe else (0.1, 3.0)
        return Scenario("C", (Regime(a, phi10, outlier_mix(a, prob, multiple)),), (1.0,), normal(sigma), severe=severe)
    if key == "D":
        return Scenario("D", (Regime(a, phi10, normal(a)),), (1.0,), outlier_mix(sigma, 0.1, 3.0))
    raise ValueError(key)


def _scales(u: np.ndarray, mixture: tuple) -> np.ndarray:
    """Noise standard deviation per element: mixture component chosen by the uniform draw u."""
    edges = np.cumsum([w for w, _ in mixture])
    stds = np.array([s for _, s in mixture])
    return stds[np.minimum(np.searchsorted(edges, u, side="right"), len(stds) - 1)]


def simulate_scenario(sc: Scenario, n_episodes: int, seed: int, n_steps: int = BASE.n_steps):
    """Simulate independent hours. Returns (Paths, regime index per hour)."""
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n_episodes, n_steps))
    eps = rng.standard_normal((n_episodes, n_steps))
    xi = rng.standard_normal((n_episodes, n_steps))
    extra = np.random.default_rng([seed, 1])
    regime = extra.choice(len(sc.regimes), size=n_episodes, p=np.array(sc.probs))
    u_signal = extra.random((n_episodes, n_steps))
    u_price = extra.random((n_episodes, n_steps))

    alpha_std = np.array([r.alpha_std for r in sc.regimes])[regime]
    phi = np.array([r.phi for r in sc.regimes])[regime]
    innov = alpha_std * np.sqrt(1.0 - phi**2)
    alpha = np.empty((n_episodes, n_steps))
    alpha[:, 0] = alpha_std * eta[:, 0]
    for t in range(1, n_steps):
        alpha[:, t] = phi * alpha[:, t - 1] + innov * eta[:, t]

    signal_scale = np.empty((n_episodes, n_steps))
    for k, r in enumerate(sc.regimes):
        mask = regime == k
        signal_scale[mask] = _scales(u_signal[mask], r.signal_noise)
    price_scale = _scales(u_price, sc.price_noise)
    price = np.zeros((n_episodes, n_steps))
    price[:, 1:] = np.cumsum(alpha[:, :-1] + price_scale[:, 1:] * eps[:, 1:], axis=1)
    return Paths(alpha=alpha, price=price, signal=alpha + signal_scale * xi), regime


def _gaussian_sum_update(m, p, y, mixture):
    """Observe y = trend + noise, noise a normal mixture; prior trend ~ N(m, p).

    Returns the moment-matched posterior (m, p) and the log predictive density of y.
    Exact when the noise has one component.
    """
    log_dens, means, variances = [], [], []
    for w, s in mixture:
        var = p + s**2
        log_dens.append(np.log(w) - 0.5 * (np.log(2.0 * np.pi * var) + (y - m) ** 2 / var))
        gain = p / var
        means.append(m + gain * (y - m))
        variances.append((1.0 - gain) * p)
    if len(mixture) == 1:
        return means[0], variances[0], log_dens[0]
    log_dens, means, variances = np.stack(log_dens), np.stack(means), np.stack(variances)
    top = log_dens.max(axis=0)
    log_pred = top + np.log(np.exp(log_dens - top).sum(axis=0))
    weights = np.exp(log_dens - log_pred)
    m_new = (weights * means).sum(axis=0)
    p_new = (weights * (variances + (means - m_new) ** 2)).sum(axis=0)
    return m_new, p_new, log_pred


class BayesForecaster:
    """Posterior-mean forecast of the price change in each remaining minute, under a scenario's structure.

    Same interface as the Stage 2 predictors: update(t, price_t, signal_t) -> (E, N-1-t).
    """

    def __init__(self, sc: Scenario, n_steps: int = BASE.n_steps):
        self.sc, self.n = sc, n_steps
        self.decay = np.array([[r.phi**j for j in range(n_steps)] for r in sc.regimes])
        self.log_prior = np.log(np.array(sc.probs))

    def reset(self, n_episodes: int) -> None:
        k = len(self.sc.regimes)
        self.m = np.zeros((k, n_episodes))
        self.p = np.array([np.full(n_episodes, r.alpha_std**2) for r in self.sc.regimes])
        self.loglik = np.repeat(self.log_prior[:, None], n_episodes, axis=1)
        self.prev_price = None

    @property
    def weights(self) -> np.ndarray:
        """Posterior regime probabilities, shape (K, E)."""
        top = self.loglik.max(axis=0)
        w = np.exp(self.loglik - top)
        return w / w.sum(axis=0)

    def trend(self) -> np.ndarray:
        """Posterior mean of the current trend, shape (E,)."""
        return (self.weights * self.m).sum(axis=0)

    def update(self, t, price_t, signal_t):
        for k, r in enumerate(self.sc.regimes):
            m, p = self.m[k], self.p[k]
            if self.prev_price is not None:
                m, p, ll = _gaussian_sum_update(m, p, price_t - self.prev_price, self.sc.price_noise)
                self.loglik[k] += ll
                m, p = r.phi * m, r.phi**2 * p + r.alpha_innov_std**2
            m, p, ll = _gaussian_sum_update(m, p, signal_t, r.signal_noise)
            self.loglik[k] += ll
            self.m[k], self.p[k] = m, p
        self.prev_price = price_t
        steps = self.n - 1 - t
        return (self.weights[:, :, None] * self.m[:, :, None] * self.decay[:, None, :steps]).sum(axis=0)


def episode_loglik(sc: Scenario, paths: Paths) -> np.ndarray:
    """Log-likelihood of each hour's observations under each regime, shape (K, E)."""
    f = BayesForecaster(sc, paths.price.shape[1])
    f.reset(paths.price.shape[0])
    f.loglik[:] = 0.0
    for t in range(paths.price.shape[1]):
        f.update(t, paths.price[:, t], paths.signal[:, t])
    return f.loglik


def mixture_loglik(sc: Scenario, paths: Paths) -> float:
    """Total log-likelihood of the hours when each hour's regime is unknown."""
    ll = episode_loglik(sc, paths) + np.log(np.array(sc.probs))[:, None]
    top = ll.max(axis=0)
    return float((top + np.log(np.exp(ll - top).sum(axis=0))).sum())


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _regime_aware(theta, key: str) -> Scenario:
    """Scenario with the true two-regime structure of A or B and parameters theta (unconstrained)."""
    if key == "A":
        p_active, a_on, phi, s_on, s_quiet, sigma = (_sigmoid(theta[0]), np.exp(theta[1]), _sigmoid(theta[2]),
                                                     np.exp(theta[3]), np.exp(theta[4]), np.exp(theta[5]))
        regimes = (Regime(float(a_on), float(phi), normal(float(s_on))), Regime(0.0, float(phi), normal(float(s_quiet))))
        return Scenario("A-fitted", regimes, (float(p_active), 1.0 - float(p_active)), normal(float(sigma)), ("active", "quiet"))
    p_first, phi1, phi2, a, s, sigma = (_sigmoid(theta[0]), _sigmoid(theta[1]), _sigmoid(theta[2]),
                                        np.exp(theta[3]), np.exp(theta[4]), np.exp(theta[5]))
    regimes = (Regime(float(a), float(phi1), normal(float(s))), Regime(float(a), float(phi2), normal(float(s))))
    return Scenario("B-fitted", regimes, (float(p_first), 1.0 - float(p_first)), normal(float(sigma)), ("first", "second"))


def fit_regime_aware(paths: Paths, key: str, single: MarketParams) -> tuple[Scenario, dict]:
    """Maximum-likelihood fit of the correct two-regime structure (A or B).

    Starts from the fitted single-regime model `single`; two starting points, the better fit is kept.
    """
    n_obs = paths.signal.size

    def objective(theta):
        return -mixture_loglik(_regime_aware(theta, key), paths) / n_obs

    a, s, sigma, phi = single.alpha_std, single.obs_std, single.sigma, single.phi
    if key == "A":
        starts = [[_logit(q), np.log(a * np.sqrt(2.0)), _logit(phi), np.log(s), np.log(np.sqrt(2.0 * a**2 + s**2)),
                   np.log(sigma)] for q in (0.3, 0.6)]
    else:
        fast, slow = phi_from_half_life(single.half_life / 3.0), phi_from_half_life(single.half_life * 3.0)
        starts = [[_logit(q), _logit(fast), _logit(slow), np.log(a), np.log(s), np.log(sigma)] for q in (0.4, 0.6)]
    best = min((minimize(objective, np.array(x0), method="L-BFGS-B") for x0 in starts), key=lambda res: res.fun)
    return _regime_aware(best.x, key), {"converged": bool(best.success), "neg_loglik_per_obs": float(best.fun)}


def _mixture_logpdf(x, mixture):
    terms = np.stack([np.log(w) - 0.5 * (np.log(2.0 * np.pi * s**2) + (x / s) ** 2) for w, s in mixture])
    top = terms.max(axis=0)
    return top + np.log(np.exp(terms - top).sum(axis=0))


def particle_trend(sc: Scenario, paths: Paths, n_particles: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap particle-filter estimate of the current trend (single-regime scenarios), shape (E, N).

    Used only to check the moment-matched filter in C and D.
    """
    assert len(sc.regimes) == 1
    r = sc.regimes[0]
    rng = np.random.default_rng(seed)
    n_episodes, n = paths.price.shape
    x = r.alpha_std * rng.standard_normal((n_episodes, n_particles))
    log_w = np.zeros((n_episodes, n_particles))
    out = np.zeros((n_episodes, n))
    offsets = np.arange(n_episodes)[:, None]
    for t in range(n):
        if t > 0:
            log_w += _mixture_logpdf((paths.price[:, t] - paths.price[:, t - 1])[:, None] - x, sc.price_noise)
            w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
            cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
            cum[:, -1] = 1.0
            positions = (rng.random((n_episodes, 1)) + np.arange(n_particles)) / n_particles
            idx = np.searchsorted((cum + offsets).ravel(), (positions + offsets).ravel())
            x = x.ravel()[np.minimum(idx, x.size - 1)].reshape(n_episodes, n_particles)
            log_w = np.zeros_like(log_w)
            x = r.phi * x + r.alpha_innov_std * rng.standard_normal(x.shape)
        log_w += _mixture_logpdf(paths.signal[:, t][:, None] - x, r.signal_noise)
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        out[:, t] = (w * x).sum(axis=1) / w.sum(axis=1)
    return out


def bayes_trend(sc: Scenario, paths: Paths) -> np.ndarray:
    """The best forecaster's posterior mean of the current trend at every minute, shape (E, N)."""
    f = BayesForecaster(sc, paths.price.shape[1])
    f.reset(paths.price.shape[0])
    out = np.zeros(paths.price.shape)
    for t in range(paths.price.shape[1]):
        f.update(t, paths.price[:, t], paths.signal[:, t])
        out[:, t] = f.trend()
    return out
