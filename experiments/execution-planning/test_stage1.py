"""Correctness checks for the Stage 1 environment and policies. Run: python -m pytest -q"""

import numpy as np
import pytest

from evaluate import run_policy, shortfall_bps
from market import AlphaFilter, MarketParams, Paths, latest_signal_estimate, simulate
from policies import TWAP, LQGReference, Planner, UrgencyRule, project_to_simplex


def test_projection_satisfies_kkt_conditions():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(500, 12))
    total = rng.uniform(0.0, 2.0, size=500)
    total[:5] = 0.0
    v = project_to_simplex(u, total)
    assert np.all(v >= 0)
    np.testing.assert_allclose(v.sum(axis=1), total, atol=1e-12)
    for row_u, row_v, tot in zip(u, v, total):
        if tot == 0:
            continue
        support = row_v > 0
        theta = (row_u - row_v)[support]
        assert np.ptp(theta) < 1e-12  # v = u - theta on the support
        assert np.all(row_u[~support] <= theta[0] + 1e-12)  # u - theta <= 0 off the support


def test_cost_identity_by_summation_by_parts():
    params = MarketParams()
    paths = simulate(params, 200, seed=1)
    rng = np.random.default_rng(2)
    trades = rng.dirichlet(np.ones(params.n_steps), size=200)
    remaining_before = 1.0 - np.cumsum(trades, axis=1) + trades  # q_t, before trading at t
    price_change = np.diff(paths.price, axis=1)  # S_t - S_{t-1}, t = 1..N-1
    expected = (
        params.half_spread
        + (remaining_before[:, 1:] * price_change).sum(axis=1)
        + params.impact * (trades**2).sum(axis=1)
    )
    np.testing.assert_allclose(shortfall_bps(trades, paths, params), expected, atol=1e-9)


def test_planner_matches_lqg_when_constraints_do_not_bind():
    params = MarketParams(impact=2000.0, r2=0.005)
    paths = simulate(params, 300, seed=3)
    lqg = LQGReference(params)
    lqg_trades = run_policy(lqg, paths, params)
    assert lqg.clipped == 0, "test premise: the no-sell constraint never binds"
    planner_trades = run_policy(Planner(params), paths, params)
    np.testing.assert_allclose(planner_trades, lqg_trades, atol=1e-10)


def test_without_signal_planner_and_lqg_are_twap():
    params = MarketParams(r2=0.0)
    paths = simulate(params, 100, seed=4)
    twap = run_policy(TWAP(params), paths, params)
    np.testing.assert_allclose(run_policy(Planner(params), paths, params), twap, atol=1e-12)
    np.testing.assert_allclose(run_policy(LQGReference(params), paths, params), twap, atol=1e-12)


@pytest.mark.parametrize(
    "make_policy",
    [
        lambda p: TWAP(p),
        lambda p: UrgencyRule(p, 0.01),
        lambda p: LQGReference(p),
        lambda p: Planner(p),
        lambda p: Planner(p, use_history=False),
    ],
)
def test_no_lookahead(make_policy):
    params = MarketParams()
    paths = simulate(params, 200, seed=5)
    cut = 25
    rng = np.random.default_rng(6)
    price = paths.price.copy()
    signal = paths.signal.copy()
    price[:, cut:] += rng.normal(0, 50, size=price[:, cut:].shape)
    signal[:, cut:] += rng.normal(0, 5, size=signal[:, cut:].shape)
    perturbed = Paths(alpha=paths.alpha, price=price, signal=signal)
    before = run_policy(make_policy(params), paths, params)
    after = run_policy(make_policy(params), perturbed, params)
    np.testing.assert_array_equal(before[:, :cut], after[:, :cut])


def test_filter_beats_latest_signal_estimate():
    params = MarketParams()
    paths = simulate(params, 2000, seed=7)
    kalman = AlphaFilter(params)
    kalman.reset(2000)
    filtered = np.column_stack(
        [kalman.update(paths.price[:, t], paths.signal[:, t]) for t in range(params.n_steps)]
    )
    latest = latest_signal_estimate(params, paths.signal)
    assert np.mean((filtered - paths.alpha) ** 2) < np.mean((latest - paths.alpha) ** 2)
