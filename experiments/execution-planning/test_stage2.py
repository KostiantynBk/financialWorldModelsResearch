"""Correctness checks for Stage 2 (fitted classical model and learned world model). Run: python -m pytest -q"""

import numpy as np
from scipy.stats import multivariate_normal

from evaluate import run_policy
from fit_classical import fit_linear_gaussian, log_likelihood
from market import MarketParams, Paths, simulate
from policies import ForecastPlanner, KalmanForecaster, Planner, forecast_cumulative, forecast_drift_to_go
from world_model import LearnedForecaster, train_world_model


def test_forecast_planner_with_true_model_matches_stage1_planner():
    params = MarketParams()
    paths = simulate(params, 300, seed=8)
    stage1 = run_policy(Planner(params), paths, params)
    stage2 = run_policy(ForecastPlanner(KalmanForecaster(params), params.impact, "true"), paths, params)
    np.testing.assert_allclose(stage2, stage1, atol=1e-10)


def test_log_likelihood_matches_multivariate_normal():
    params = MarketParams(n_steps=6, r2=0.05, half_life=3.0, obs_noise=0.7)
    paths = simulate(params, 4, seed=9)
    n = params.n_steps
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    trend_cov = params.alpha_std**2 * params.phi**lag
    select = np.eye(n)[: n - 1]  # price change at t = trend at t-1 + noise, t = 1..n-1
    cov = np.block([
        [trend_cov + params.obs_std**2 * np.eye(n), trend_cov @ select.T],
        [select @ trend_cov, select @ trend_cov @ select.T + params.sigma**2 * np.eye(n - 1)],
    ])
    y = np.hstack([paths.signal, np.diff(paths.price, axis=1)])
    expected = multivariate_normal(np.zeros(2 * n - 1), cov).logpdf(y).sum()
    assert np.isclose(log_likelihood(params, paths), expected, rtol=1e-9)


def test_maximum_likelihood_recovers_parameters():
    params = MarketParams()
    fitted, info = fit_linear_gaussian(simulate(params, 1000, seed=11), params)
    assert info["converged"]
    assert abs(fitted.half_life - params.half_life) < 1.5
    assert abs(fitted.alpha_std / params.alpha_std - 1) < 0.1
    assert abs(fitted.obs_std / params.obs_std - 1) < 0.1
    assert abs(fitted.sigma / params.sigma - 1) < 0.02
    assert fitted.impact == params.impact  # the cost model is assumed, not fitted


def _small_learned_model(params):
    model, scales, _ = train_world_model(simulate(params, 30, seed=12), seed=0, max_steps=10)
    return model, scales


def test_learned_planner_has_no_lookahead():
    params = MarketParams()
    model, scales = _small_learned_model(params)
    paths = simulate(params, 50, seed=13)
    cut = 25
    rng = np.random.default_rng(14)
    price, signal = paths.price.copy(), paths.signal.copy()
    price[:, cut:] += rng.normal(0, 50, size=price[:, cut:].shape)
    signal[:, cut:] += rng.normal(0, 5, size=signal[:, cut:].shape)
    perturbed = Paths(alpha=paths.alpha, price=price, signal=signal)

    def make():
        return ForecastPlanner(LearnedForecaster(model, scales, params.n_steps), params.impact, "learned")

    before = run_policy(make(), paths, params)
    after = run_policy(make(), perturbed, params)
    np.testing.assert_array_equal(before[:, :cut], after[:, :cut])


def test_learned_forecast_is_sign_symmetric():
    params = MarketParams()
    model, scales = _small_learned_model(params)
    paths = simulate(params, 40, seed=15)
    mirrored = Paths(alpha=-paths.alpha, price=-paths.price, signal=-paths.signal)
    forecast = forecast_drift_to_go(LearnedForecaster(model, scales, params.n_steps), paths)
    mirrored_forecast = forecast_drift_to_go(LearnedForecaster(model, scales, params.n_steps), mirrored)
    # The network runs in float32; allow rounding error that is tiny relative to the forecast's size.
    np.testing.assert_allclose(forecast, -mirrored_forecast, rtol=0, atol=1e-3 * np.abs(forecast).max())


def test_cumulative_forecasts_are_consistent():
    params = MarketParams()
    paths = simulate(params, 20, seed=16)
    out = forecast_cumulative(KalmanForecaster(params), paths, [1, 5, None])
    np.testing.assert_allclose(out[None], forecast_drift_to_go(KalmanForecaster(params), paths))
    assert np.isnan(out[5][:, -4:]).all() and not np.isnan(out[5][:, :-4]).any()  # t + 5 > N - 1 is undefined
    assert not np.isnan(out[1]).any()
    # The true model's one-step forecast is the filtered trend; five steps ahead adds its geometric fade.
    np.testing.assert_allclose(out[5][:, 0], out[1][:, 0] * (1 - params.phi**5) / (1 - params.phi))
