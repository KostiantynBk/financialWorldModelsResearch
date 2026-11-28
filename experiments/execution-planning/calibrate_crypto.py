"""Calibration on real data: how predictive are short-term signals in a real market?

Run from this folder:  python calibrate_crypto.py
Downloads Binance spot one-minute bars from the public archive (data.binance.vision) into data/binance/,
verifies each file's SHA-256 checksum, and writes results/calibration.md, results/calibration.csv and
results/calibration_sim.csv. The raw data is git-ignored and not redistributed here. About 10 minutes.

Design (fixed before any result was seen; descriptive measurement, not a hypothesis test):
- Markets: BTCUSDT and ETHUSDT on Binance spot, one-minute bars.
- Chronological split: fit period November 2025 to June 2026, evaluation period July to October 2026.
- At the close of minute t we know the close price and the minute's trade imbalance
  TI_t = (aggressive buy volume - aggressive sell volume) / total volume, in [-1, 1].
- Targets: log return from the close of minute t to the close of minute t + h, in bps, for h = 1, 5, 15, 30, 60.
- Predictors, fitted by least squares on the fit period and scored on the evaluation period:
  "imbalance" (TI_t), "last return" (r_t), and "history" (TI and r over the last 30 minutes, 60 regressors).
- Score: out-of-sample R^2 relative to forecasting zero, as in Stages 2-3, with a 95% interval from a
  block bootstrap over days.
- Same-minute R^2 of r_t on TI_t, to compare with predictive power.
- Persistence: slope of the return k minutes ahead on TI_t, k = 1..30, on the fit period.
- Volatility: standard deviation of one-minute returns, in bps.
- Reference: the same statistics on a continuous version of the Stage 1 simulator for each Stage 1 grid cell,
  with the simulator's signal in the role of TI, to place the real markets in the Stage 1 headroom table.

Diagnostic added after the first results were seen: the same predictors refitted without an intercept, because
the out-of-sample R^2 became more negative with the horizon, as expected if the fit period's average drift does
not carry over to the evaluation period.
"""

import csv
import hashlib
import io
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from market import MarketParams

HERE = Path(__file__).parent
CACHE = HERE / "data" / "binance"
RESULTS = HERE / "results"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIT_MONTHS = ["2025-11", "2025-12"] + [f"2026-{m:02d}" for m in range(1, 7)]
EVAL_MONTHS = [f"2026-{m:02d}" for m in range(7, 11)]
HORIZONS = [1, 5, 15, 30, 60]
LAGS = 30
PERSISTENCE_LAGS = 30
N_BOOT = 1000
GRID_R2 = [0.0025, 0.005, 0.01, 0.02, 0.05]
GRID_HALF_LIFE = [3.0, 10.0, 30.0]
SIM_SEED = 5050


# ---------------------------------------------------------------- data

def download(symbol: str, month: str) -> Path:
    """Fetch one monthly file (once) and verify its published SHA-256 checksum."""
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}-1m-{month}.zip"
    path = CACHE / name
    if not path.exists():
        url = f"{BASE_URL}/{symbol}/1m/{name}"
        data = urllib.request.urlopen(url, timeout=120).read()
        expected = urllib.request.urlopen(url + ".CHECKSUM", timeout=60).read().decode().split()[0]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"checksum mismatch for {name}")
        path.write_bytes(data)
    return path


def parse_klines(text: str) -> np.ndarray:
    """Rows of (minute index since the epoch, close, volume, aggressive buy volume).

    Handles files with or without a header, and open times in milliseconds or microseconds.
    """
    lines = text.strip().splitlines()
    if lines and not lines[0][:1].isdigit():
        lines = lines[1:]
    raw = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", usecols=(0, 4, 5, 9), ndmin=2)
    per_minute = np.where(raw[:, 0] > 1e14, 60_000_000, 60_000)
    minute = np.floor_divide(raw[:, 0], per_minute)
    return np.column_stack([minute, raw[:, 1:]])


@dataclass(frozen=True)
class Bars:
    minute: np.ndarray  # minutes since the epoch, consecutive
    close: np.ndarray
    volume: np.ndarray
    taker_buy: np.ndarray
    observed: np.ndarray  # False where the archive has no bar (price carried forward, no volume)


def to_bars(rows: np.ndarray) -> Bars:
    """Sort, drop duplicate minutes and fill gaps so that every minute has a row."""
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    rows = rows[np.concatenate([[True], np.diff(rows[:, 0]) > 0])]
    first, last = int(rows[0, 0]), int(rows[-1, 0])
    n = last - first + 1
    pos = (rows[:, 0] - first).astype(int)
    observed = np.zeros(n, dtype=bool)
    observed[pos] = True
    close = np.full(n, np.nan)
    close[pos] = rows[:, 1]
    idx = np.where(observed, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    close = close[idx]
    volume, taker = np.zeros(n), np.zeros(n)
    volume[pos], taker[pos] = rows[:, 2], rows[:, 3]
    return Bars(np.arange(first, last + 1), close, volume, taker, observed)


def load(symbol: str, months: list) -> Bars:
    rows = []
    for month in months:
        with zipfile.ZipFile(download(symbol, month)) as z:
            rows.append(parse_klines(z.read(z.namelist()[0]).decode()))
    return to_bars(np.vstack(rows))


# ---------------------------------------------------------------- features and scoring

def imbalance(volume: np.ndarray, taker_buy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(volume)
    np.divide(2.0 * taker_buy - volume, volume, out=out, where=volume > 0)
    return out


def log_price_bps(close: np.ndarray) -> np.ndarray:
    return 1e4 * np.log(close)


def last_return(lp: np.ndarray) -> np.ndarray:
    r = np.zeros_like(lp)
    r[1:] = np.diff(lp)
    return r


def forward_return(lp: np.ndarray, h: int) -> np.ndarray:
    out = np.full_like(lp, np.nan)
    out[: len(lp) - h] = lp[h:] - lp[:-h]
    return out


def lagged(x: np.ndarray, lags: int) -> np.ndarray:
    """Column j holds x[t - j], j = 0..lags-1 (NaN before the start): only past and current values."""
    out = np.full((len(x), lags), np.nan)
    for j in range(lags):
        out[j:, j] = x[: len(x) - j]
    return out


def features(signal: np.ndarray, ret: np.ndarray) -> dict:
    return {"imbalance": signal[:, None], "last return": ret[:, None],
            "history": np.hstack([lagged(signal, LAGS), lagged(ret, LAGS)])}


def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ coef[:-1] + coef[-1]


def r2_zero(y: np.ndarray, f: np.ndarray) -> float:
    """Out-of-sample R^2 relative to forecasting zero."""
    return 1.0 - float(((y - f) ** 2).sum() / (y**2).sum())


def r2_interval(y, f, day, n_boot=N_BOOT, seed=0):
    """Point estimate and 95% block-bootstrap interval (blocks = days)."""
    _, day_idx = np.unique(day, return_inverse=True)
    sse = np.bincount(day_idx, (y - f) ** 2)
    sst = np.bincount(day_idx, y**2)
    draws = np.random.default_rng(seed).integers(0, len(sse), size=(n_boot, len(sse)))
    boot = 1.0 - sse[draws].sum(axis=1) / sst[draws].sum(axis=1)
    return r2_zero(y, f), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def slopes_ahead(signal: np.ndarray, ret: np.ndarray, rows: np.ndarray, k_max: int) -> np.ndarray:
    """Least-squares slope of the one-minute return k minutes ahead on the signal now, k = 1..k_max."""
    out = []
    for k in range(1, k_max + 1):
        ok = rows[rows + k < len(ret)]
        x, y = signal[ok], ret[ok + k]
        x = x - x.mean()
        out.append(float(x @ (y - y.mean()) / (x @ x)))
    return np.array(out)


def half_life(slopes: np.ndarray) -> float:
    """First horizon (in minutes, interpolated) at which the slope falls below half its 1-minute value."""
    if slopes[0] <= 0:
        return float("nan")
    below = np.nonzero(slopes < 0.5 * slopes[0])[0]
    if len(below) == 0:
        return float("inf")
    k = below[0]
    s0, s1 = slopes[k - 1], slopes[k]
    return float(k + (s0 - 0.5 * slopes[0]) / (s0 - s1))  # slopes[j] is the slope j+1 minutes ahead


def analyse(lp, signal, fit_rows, eval_rows, day):
    """All predictive statistics for one price series and signal. Returns (score rows, extras)."""
    ret = last_return(lp)
    feats = features(signal, ret)
    scores = []
    for h in HORIZONS:
        y = forward_return(lp, h)
        fit = fit_rows[fit_rows + h <= fit_rows.max()]  # fit targets end inside the fit period
        for name, x in feats.items():
            ok_fit = fit[np.isfinite(x[fit]).all(axis=1)]
            coef = ols(x[ok_fit], y[ok_fit])
            ev = eval_rows[np.isfinite(y[eval_rows])]
            r2, low, high = r2_interval(y[ev], predict(coef, x[ev]), day[ev], seed=h)
            coef0, *_ = np.linalg.lstsq(x[ok_fit], y[ok_fit], rcond=None)  # diagnostic: no intercept
            r2_0, low_0, high_0 = r2_interval(y[ev], x[ev] @ coef0, day[ev], seed=h)
            scores.append({"horizon": h, "predictor": name, "r2": r2, "low": low, "high": high,
                           "r2_no_intercept": r2_0, "low_no_intercept": low_0, "high_no_intercept": high_0})
    same = ols(signal[fit_rows, None], ret[fit_rows])
    contemporaneous = r2_interval(ret[eval_rows], predict(same, signal[eval_rows, None]), day[eval_rows], seed=99)
    slopes = slopes_ahead(signal, ret, fit_rows, PERSISTENCE_LAGS)
    extras = {"contemporaneous": contemporaneous, "slopes": slopes, "slope_half_life": half_life(slopes),
              "vol_fit": float(ret[fit_rows].std()), "vol_eval": float(ret[eval_rows].std()),
              "mean_fit": float(ret[fit_rows].mean()), "mean_eval": float(ret[eval_rows].mean())}
    return scores, extras


# ---------------------------------------------------------------- simulator reference

def simulate_continuous(params: MarketParams, n_minutes: int, seed: int):
    """One long path of the Stage 1 market without hourly resets: (log price in bps, signal)."""
    rng = np.random.default_rng(seed)
    eta, eps, xi = rng.standard_normal((3, n_minutes))
    alpha = np.empty(n_minutes)
    alpha[0] = params.alpha_std * eta[0]
    for t in range(1, n_minutes):
        alpha[t] = params.phi * alpha[t - 1] + params.alpha_innov_std * eta[t]
    lp = np.zeros(n_minutes)
    lp[1:] = np.cumsum(alpha[:-1] + params.sigma * eps[1:])
    return lp, alpha + params.obs_std * xi


def stage1_headroom() -> dict:
    rows = csv.DictReader(open(RESULTS / "gate_b.csv"))
    return {(float(r["r2"]), float(r["half_life"])): float(r["headroom_bps"]) for r in rows if r["group"] == "grid"}


# ---------------------------------------------------------------- report

def _cell(score, suffix=""):
    return (f"{100 * score['r2' + suffix]:+.3f}% [{100 * score['low' + suffix]:+.3f}, "
            f"{100 * score['high' + suffix]:+.3f}]")


def main():
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    real, provenance = {}, []
    for symbol in SYMBOLS:
        bars = load(symbol, FIT_MONTHS + EVAL_MONTHS)
        boundary = int(load(symbol, FIT_MONTHS).minute[-1])
        rows_all = np.arange(len(bars.minute))
        fit_rows = rows_all[(bars.minute <= boundary) & (rows_all >= LAGS)]
        eval_rows = rows_all[bars.minute > boundary]
        scores, extras = analyse(log_price_bps(bars.close), imbalance(bars.volume, bars.taker_buy),
                                 fit_rows, eval_rows, bars.minute // 1440)
        real[symbol] = (scores, extras)
        provenance.append({"symbol": symbol, "minutes": len(bars.minute), "missing": int((~bars.observed).sum()),
                           "fit_minutes": len(fit_rows), "eval_minutes": len(eval_rows)})
        print(f"{symbol}: done ({time.time() - start:.0f}s)", flush=True)

    n_fit, n_eval = provenance[0]["fit_minutes"] + LAGS, provenance[0]["eval_minutes"]
    headroom = stage1_headroom()
    sim = []
    for i, (r2, hl) in enumerate((r2, hl) for r2 in GRID_R2 for hl in GRID_HALF_LIFE):
        params = MarketParams(r2=r2, half_life=hl)
        lp, z = simulate_continuous(params, n_fit + n_eval, SIM_SEED + i)
        rows_all = np.arange(n_fit + n_eval)
        scores, extras = analyse(lp, z, rows_all[LAGS:n_fit], rows_all[n_fit:], rows_all // 1440)
        sim.append({"r2": r2, "half_life": hl, "scores": scores, "extras": extras,
                    "headroom": headroom.get((r2, hl), float("nan"))})
    print(f"simulator grid: done ({time.time() - start:.0f}s)", flush=True)

    with (RESULTS / "calibration.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["market", "horizon", "predictor", "r2", "low", "high", "r2_no_intercept",
                                               "low_no_intercept", "high_no_intercept"])
        writer.writeheader()
        for symbol, (scores, _) in real.items():
            writer.writerows({"market": symbol, **s} for s in scores)
    with (RESULTS / "calibration_sim.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sim_r2", "sim_half_life", "stage1_headroom_bps", "horizon", "predictor",
                                               "r2", "low", "high", "r2_no_intercept", "low_no_intercept",
                                               "high_no_intercept"])
        writer.writeheader()
        for cell in sim:
            writer.writerows({"sim_r2": cell["r2"], "sim_half_life": cell["half_life"],
                              "stage1_headroom_bps": cell["headroom"], **s} for s in cell["scores"])
    write_markdown(real, sim, provenance, time.time() - start)
    print(f"Done in {time.time() - start:.0f} s. See {RESULTS / 'calibration.md'}")


def _pick(scores, h, name):
    return next(s for s in scores if s["horizon"] == h and s["predictor"] == name)


def write_markdown(real, sim, provenance, elapsed):
    lines = [
        "# Calibration on real data (generated by calibrate_crypto.py)", "",
        f"Binance spot one-minute bars from the public archive (data.binance.vision); every file's SHA-256 checksum was "
        f"verified. Fit period {FIT_MONTHS[0]} to {FIT_MONTHS[-1]}, evaluation period {EVAL_MONTHS[0]} to {EVAL_MONTHS[-1]}. "
        f"R² is out of sample (evaluation period) and relative to forecasting zero; intervals are 95% block-bootstrap "
        f"intervals over days. Runtime {elapsed:.0f} s.", "",
        "| Market | Minutes | Minutes missing in the archive | Fit minutes | Evaluation minutes |", "|---|---:|---:|---:|---:|",
    ]
    for p in provenance:
        lines.append(f"| {p['symbol']} | {p['minutes']:,} | {p['missing']:,} | {p['fit_minutes']:,} | {p['eval_minutes']:,} |")
    lines += ["", "## Volatility and same-minute vs predictive power", "",
              "| Market | One-minute volatility, fit / evaluation (bps) | Same-minute R² of the return on the imbalance | "
              "Next-minute R² from the imbalance |", "|---|---|---|---|"]
    for symbol, (scores, ex) in real.items():
        c = ex["contemporaneous"]
        lines.append(f"| {symbol} | {ex['vol_fit']:.2f} / {ex['vol_eval']:.2f} | {100 * c[0]:.1f}% "
                     f"[{100 * c[1]:.1f}, {100 * c[2]:.1f}] | {_cell(_pick(scores, 1, 'imbalance'))} |")
    lines.append(f"| Stage 1 simulator (base case) | {MarketParams().sigma:.2f} | — | — |")
    for symbol, (scores, ex) in real.items():
        lines += ["", f"## Out-of-sample predictive R²: {symbol}", "",
                  "| Horizon (minutes) | Imbalance | Last return | History (30 minutes of both) |", "|---:|---|---|---|"]
        for h in HORIZONS:
            lines.append(f"| {h} | {_cell(_pick(scores, h, 'imbalance'))} | {_cell(_pick(scores, h, 'last return'))} | "
                         f"{_cell(_pick(scores, h, 'history'))} |")
        s = ex["slopes"]
        lines += ["", f"Persistence (fit period): slope of the return k minutes ahead on the imbalance, in bps per unit "
                  f"of imbalance: k = 1: {s[0]:+.3f}, k = 2: {s[1]:+.3f}, k = 5: {s[4]:+.3f}, k = 10: {s[9]:+.3f}, "
                  f"k = 30: {s[29]:+.3f}. Slope half-life: {ex['slope_half_life']:.1f} minutes."]
    lines += ["", "## Diagnostic: refitted without an intercept (added after the first results)", "",
              "Mean one-minute return (bps), fit / evaluation period: "
              + "; ".join(f"{sym} {ex['mean_fit']:+.4f} / {ex['mean_eval']:+.4f}" for sym, (_, ex) in real.items())
              + ". A fitted intercept carries the fit period's average drift into the evaluation period.", "",
              "| Market | Horizon (minutes) | Imbalance, no intercept | History, no intercept |", "|---|---:|---|---|"]
    for symbol, (scores, _) in real.items():
        for h in HORIZONS:
            lines.append(f"| {symbol} | {h} | {_cell(_pick(scores, h, 'imbalance'), '_no_intercept')} | "
                         f"{_cell(_pick(scores, h, 'history'), '_no_intercept')} |")
    lines += ["", "## Reference: the same statistics on the Stage 1 simulator", "",
              "A continuous version of the Stage 1 market (signal noise equal to the trend's size), with the same numbers of "
              "fit and evaluation minutes. Headroom is the Stage 1 true-model planner's saving over the best simple rule.", "",
              "| Simulator R² | Half-life | Next-minute R², signal only | History R², 1 / 5 / 15 / 30 min | Slope half-life | "
              "Stage 1 headroom (bps) |", "|---:|---:|---|---|---:|---:|"]
    for cell in sim:
        sc = cell["scores"]
        hist = " / ".join(f"{100 * _pick(sc, h, 'history')['r2']:+.3f}%" for h in (1, 5, 15, 30))
        lines.append(f"| {100 * cell['r2']:.2f}% | {cell['half_life']:g} | {100 * _pick(sc, 1, 'imbalance')['r2']:+.3f}% | "
                     f"{hist} | {cell['extras']['slope_half_life']:.1f} | {cell['headroom']:+.2f} |")
    lines += ["", "| Real market | Next-minute R², imbalance only | History R², 1 / 5 / 15 / 30 min | Slope half-life |",
              "|---|---|---|---:|"]
    for symbol, (scores, ex) in real.items():
        hist = " / ".join(f"{100 * _pick(scores, h, 'history')['r2']:+.3f}%" for h in (1, 5, 15, 30))
        lines.append(f"| {symbol} | {100 * _pick(scores, 1, 'imbalance')['r2']:+.3f}% | {hist} | {ex['slope_half_life']:.1f} |")
    (RESULTS / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
