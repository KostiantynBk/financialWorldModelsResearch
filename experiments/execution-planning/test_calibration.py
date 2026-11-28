"""Correctness checks for the real-data calibration (no network access needed). Run: python -m pytest -q"""

import numpy as np

from calibrate_crypto import (analyse, forward_return, half_life, imbalance, lagged, parse_klines, r2_interval, r2_zero,
                              simulate_continuous, slopes_ahead, to_bars)
from market import MarketParams


def _row(open_time, close, volume, taker):
    return f"{open_time},1,1,1,{close},{volume},0,0,0,{taker},0,0"


def test_parse_handles_microseconds_milliseconds_and_headers():
    us = parse_klines("\n".join([_row(1_790_812_800_000_000, 100.0, 2.0, 1.5), _row(1_790_812_860_000_000, 101.0, 1.0, 0.0)]))
    ms = parse_klines("open_time,o,h,l,c,v,ct,qv,n,tb,tq,i\n" + _row(1_790_812_800_000, 100.0, 2.0, 1.5))
    assert us[0, 0] == ms[0, 0] == 1_790_812_800 // 60
    assert us[1, 0] - us[0, 0] == 1
    np.testing.assert_allclose(us[0, 1:], [100.0, 2.0, 1.5])


def test_bars_are_sorted_deduplicated_and_gap_filled():
    rows = np.array([[12, 103.0, 1.0, 1.0], [10, 100.0, 2.0, 1.0], [10, 100.0, 2.0, 1.0], [13, 104.0, 3.0, 0.0]])
    bars = to_bars(rows)
    np.testing.assert_array_equal(bars.minute, [10, 11, 12, 13])
    np.testing.assert_array_equal(bars.observed, [True, False, True, True])
    np.testing.assert_allclose(bars.close, [100.0, 100.0, 103.0, 104.0])  # the gap carries the last price forward
    np.testing.assert_allclose(bars.volume, [2.0, 0.0, 1.0, 3.0])


def test_imbalance():
    np.testing.assert_allclose(imbalance(np.array([2.0, 4.0, 0.0, 1.0]), np.array([2.0, 1.0, 0.0, 0.5])),
                               [1.0, -0.5, 0.0, 0.0])


def test_features_use_only_the_past_and_targets_only_the_future():
    x = np.arange(10.0)
    lags = lagged(x, 3)
    np.testing.assert_array_equal(lags[5], [5.0, 4.0, 3.0])
    assert np.isnan(lags[1, 2])
    f = forward_return(x, 2)
    assert f[3] == 2.0 and np.isnan(f[8])


def test_r2_and_bootstrap_interval():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    assert r2_zero(y, y) == 1.0 and r2_zero(y, np.zeros_like(y)) == 0.0
    f = 0.3 * y + rng.standard_normal(5000)
    point, low, high = r2_interval(y, f, np.arange(5000) // 100)
    assert low <= point <= high


def test_continuous_simulator_and_persistence():
    params = MarketParams(r2=0.05, half_life=10.0)
    lp, z = simulate_continuous(params, 400_000, seed=1)
    ret = np.diff(lp, prepend=0.0)
    assert abs(ret[1:].std() / np.sqrt(params.sigma**2 + params.alpha_std**2) - 1) < 0.01
    slopes = slopes_ahead(z, ret, np.arange(400_000 - 40), 30)
    assert abs(half_life(slopes) - 10.0) < 1.5  # the predictive slope fades with the trend's half-life


def test_analyse_recovers_the_theoretical_next_minute_r2():
    params = MarketParams()  # trend explains 1%; signal noise equal to the trend's size
    lp, z = simulate_continuous(params, 300_000, seed=2)
    rows = np.arange(300_000)
    scores, extras = analyse(lp, z, rows[30:200_000], rows[200_000:], rows // 1440)
    signal_only = next(s for s in scores if s["horizon"] == 1 and s["predictor"] == "imbalance")
    # R^2 of the next return on z_t = r2 * a^2 / (a^2 + noise^2) = 0.5%
    assert abs(signal_only["r2"] - 0.005) < 0.0015
    assert abs(extras["vol_fit"] - np.sqrt(params.sigma**2 + params.alpha_std**2)) < 0.05
