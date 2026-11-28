"""Correctness checks for Stage 3 (misspecified markets and their best forecasters). Run: python -m pytest -q

None of these checks uses the Stage 3 validation (212) or test (313) seeds.
"""

import numpy as np

from fit_classical import fit_linear_gaussian
from market import MarketParams, Paths, simulate
from policies import KalmanForecaster
from scenarios import (BayesForecaster, _gaussian_sum_update, bayes_trend, fit_regime_aware, normal, particle_trend,
                       scenario, simulate_scenario)

BASE = MarketParams()


def _r2(estimate, target):
    return 1.0 - float(((target - estimate) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def test_base_scenario_reproduces_the_stage1_simulator():
    paths, _ = simulate_scenario(scenario("base"), 50, seed=21)
    reference = simulate(BASE, 50, seed=21)
    for name in ("alpha", "price", "signal"):
        np.testing.assert_allclose(getattr(paths, name), getattr(reference, name), rtol=0, atol=1e-12)


def test_bayes_forecaster_equals_the_kalman_forecaster_on_the_base_market():
    paths = simulate(BASE, 40, seed=22)
    bayes, kalman = BayesForecaster(scenario("base")), KalmanForecaster(BASE)
    bayes.reset(40)
    kalman.reset(40)
    for t in range(BASE.n_steps - 1):
        np.testing.assert_allclose(bayes.update(t, paths.price[:, t], paths.signal[:, t]),
                                   kalman.update(t, paths.price[:, t], paths.signal[:, t]), rtol=0, atol=1e-12)


def test_weighting_identical_regimes_or_noise_components_changes_nothing():
    base = scenario("base")
    twin = base.__class__("twin", base.regimes * 2, (0.5, 0.5), base.price_noise)
    paths = simulate(BASE, 30, seed=23)
    a, b = BayesForecaster(base), BayesForecaster(twin)
    a.reset(30)
    b.reset(30)
    for t in range(BASE.n_steps - 1):
        np.testing.assert_allclose(a.update(t, paths.price[:, t], paths.signal[:, t]),
                                   b.update(t, paths.price[:, t], paths.signal[:, t]), rtol=0, atol=1e-12)
    m, p, y, s = np.array([0.3, -0.2]), np.array([0.5, 0.4]), np.array([1.0, 0.1]), 0.7
    single = _gaussian_sum_update(m, p, y, normal(s))
    split = _gaussian_sum_update(m, p, y, ((0.9, s), (0.1, s)))
    for x, z in zip(single, split):
        np.testing.assert_allclose(x, z, rtol=0, atol=1e-12)


def test_scenarios_match_their_design():
    a2 = BASE.alpha_std**2
    paths, regime = simulate_scenario(scenario("A"), 20_000, seed=24)
    assert abs(regime.mean() - 0.5) < 0.02
    signal_var = [paths.signal[regime == k].var() for k in (0, 1)]
    assert abs(signal_var[0] / signal_var[1] - 1) < 0.03  # signal size does not reveal the regime
    assert np.all(paths.alpha[regime == 1] == 0)
    assert abs(paths.alpha.var() / a2 - 1) < 0.05  # average trend strength unchanged

    paths, regime = simulate_scenario(scenario("B"), 20_000, seed=25)
    lag1 = [np.corrcoef(paths.alpha[regime == k][:, :-1].ravel(), paths.alpha[regime == k][:, 1:].ravel())[0, 1]
            for k in (0, 1)]
    np.testing.assert_allclose(lag1, [0.5 ** (1 / 3), 0.5 ** (1 / 30)], atol=0.01)
    assert abs(paths.alpha.var() / a2 - 1) < 0.05

    paths, _ = simulate_scenario(scenario("C"), 20_000, seed=26)
    noise = (paths.signal - paths.alpha).ravel()
    assert abs(noise.var() / BASE.obs_std**2 - 1) < 0.02
    kurtosis = float(np.mean(noise**4) / noise.var() ** 2)
    assert abs(kurtosis - 27 / 3.24) < 0.5  # 8.33 for the 10% / 3x mixture

    paths, _ = simulate_scenario(scenario("D"), 20_000, seed=27)
    price_noise = (np.diff(paths.price, axis=1) - paths.alpha[:, :-1]).ravel()
    assert abs(price_noise.var() / BASE.sigma**2 - 1) < 0.02


def test_bayes_forecaster_has_no_lookahead():
    sc = scenario("A")
    paths, _ = simulate_scenario(sc, 50, seed=28)
    cut = 30
    rng = np.random.default_rng(29)
    price, signal = paths.price.copy(), paths.signal.copy()
    price[:, cut:] += rng.normal(0, 50, size=price[:, cut:].shape)
    signal[:, cut:] += rng.normal(0, 5, size=signal[:, cut:].shape)
    before, after = BayesForecaster(sc), BayesForecaster(sc)
    before.reset(50)
    after.reset(50)
    for t in range(cut):
        np.testing.assert_array_equal(before.update(t, paths.price[:, t], paths.signal[:, t]),
                                      after.update(t, price[:, t], signal[:, t]))


def test_best_forecaster_beats_the_misspecified_kalman_filter():
    for key in ("A", "B", "C"):
        sc = scenario(key)
        train, _ = simulate_scenario(sc, 1000, seed=30)
        fitted, _ = fit_linear_gaussian(train, BASE)
        paths, _ = simulate_scenario(sc, 3000, seed=31)
        kalman = KalmanForecaster(fitted)
        kalman.reset(3000)
        single = np.stack([kalman.update(t, paths.price[:, t], paths.signal[:, t])[:, 0] for t in range(BASE.n_steps - 1)], 1)
        best = bayes_trend(sc, paths)[:, :-1]
        target = paths.alpha[:, :-1].ravel()
        assert _r2(best.ravel(), target) > _r2(single.ravel(), target) + 0.005, key


def test_moment_matched_filter_agrees_with_a_particle_filter():
    sc = scenario("C")
    paths, _ = simulate_scenario(sc, 300, seed=32)
    gaussian_sum, particles = bayes_trend(sc, paths), particle_trend(sc, paths, n_particles=1000, seed=33)
    target = paths.alpha.ravel()
    assert np.corrcoef(gaussian_sum.ravel(), particles.ravel())[0, 1] > 0.98
    assert abs(_r2(gaussian_sum.ravel(), target) - _r2(particles.ravel(), target)) < 0.02


def test_regime_aware_fit_recovers_scenario_A():
    sc = scenario("A")
    train, _ = simulate_scenario(sc, 2000, seed=34)
    single, _ = fit_linear_gaussian(train, BASE)
    fitted, info = fit_regime_aware(train, "A", single)
    active = fitted.regimes[0]
    assert abs(fitted.probs[0] - 0.5) < 0.1
    assert abs(active.alpha_std / sc.regimes[0].alpha_std - 1) < 0.15
    assert fitted.regimes[1].alpha_std == 0.0
