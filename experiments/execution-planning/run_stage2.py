"""Stage 2: a learned world model vs a fitted classical model vs simple rules.

Run from this folder:  python run_stage2.py
Writes results/stage2_runs.csv, results/stage2_summary.csv and results/stage2.md.

Protocol (prespecified in PLAN.md):
- True market: the Stage 1 base-case simulator.
- Training data: run r = 0..4 draws 1,000 hours with seed 101 + r (run 0 uses the plan's
  training seed 101). The first D hours (D = 50, 200, 1,000) are used, so smaller training
  sets are nested in larger ones. The run index also seeds the network's initialization.
- Everything that learns or tunes sees only those D hours: the learned world model,
  the fitted classical model, the urgency rule's threshold, and the choice of best
  simple rule.
- Test: 20,000 hours from the reserved test seed 303, used here for the first time.
- The cost model (spread, impact) is assumed known by every model-based strategy;
  only the market forecast differs between them.
- Decision metrics: mean implementation shortfall (paired 95% intervals) and CVaR95, the
  mean cost of the worst 5% of test hours. Forecast metric: out-of-sample R² of the
  forecast price change 1, 5 and 15 minutes ahead and to the deadline. The learned model
  gives point forecasts only, so likelihood-based scores are not reported.
"""

import csv
import time
from pathlib import Path

import numpy as np

from evaluate import paired_diff
from fit_classical import fit_linear_gaussian
from market import MarketParams, Paths, simulate
from policies import TWAP, ForecastPlanner, KalmanForecaster, UrgencyRule, forecast_cumulative
from run_stage1 import cost, tune_rules
from world_model import LearnedForecaster, train_world_model

BASE = MarketParams()
D_VALUES = [50, 200, 1000]
N_RUNS = 5
TRAIN_SEED0, POOL_SIZE = 101, 1000
TEST_SEED, N_TEST = 303, 20_000
HORIZONS = {"1": 1, "5": 5, "15": 15, "deadline": None}  # minutes ahead; None = to the deadline
RESULTS = Path(__file__).parent / "results"

TRUE = "True model + planner"
FITTED = "Fitted classical model + planner"
LEARNED = "Learned model + planner"
BEST_RULE = "Best simple rule"


def subset(paths: Paths, d: int) -> Paths:
    return Paths(alpha=paths.alpha[:d], price=paths.price[:d], signal=paths.signal[:d])


def planner_cost(forecaster, test):
    return cost(ForecastPlanner(forecaster, BASE.impact, "planner"), test, BASE)


def cvar95(costs: np.ndarray) -> float:
    """Mean cost of the worst 5% of hours."""
    worst = np.sort(costs)[-max(1, int(round(0.05 * costs.size))):]
    return float(worst.mean())


def realized_changes(paths: Paths) -> dict:
    """Realized S_{t+h} - S_t for each horizon, indexed by decision time t; NaN past the deadline."""
    n_episodes, n = paths.price.shape
    out = {}
    for key, h in HORIZONS.items():
        changes = np.full((n_episodes, n - 1), np.nan)
        for t in range(n - 1):
            end = n - 1 if h is None else t + h
            if end <= n - 1:
                changes[:, t] = paths.price[:, end] - paths.price[:, t]
        out[key] = changes
    return out


def forecasts(forecaster, test) -> dict:
    raw = forecast_cumulative(forecaster, test, list(HORIZONS.values()))
    return {key: raw[h] for key, h in HORIZONS.items()}


def r2(forecast, realized) -> float:
    """Out-of-sample R² relative to forecasting no change, over all valid (hour, minute) pairs."""
    valid = ~np.isnan(realized)
    return 1.0 - float(((realized - forecast)[valid] ** 2).sum() / (realized[valid] ** 2).sum())


def forecast_scores(forecaster, test, realized, true_forecasts) -> dict:
    f = forecasts(forecaster, test)
    scores = {f"r2_{key}": r2(f[key], realized[key]) for key in HORIZONS}
    scores["rmse_to_true"] = float(np.sqrt(((f["deadline"] - true_forecasts["deadline"]) ** 2).mean()))
    return scores


def main():
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    test = simulate(BASE, N_TEST, TEST_SEED)
    realized = realized_changes(test)
    true_forecasts = forecasts(KalmanForecaster(BASE), test)
    true_r2 = {key: r2(true_forecasts[key], realized[key]) for key in HORIZONS}
    twap = cost(TWAP(BASE), test, BASE)
    true = planner_cost(KalmanForecaster(BASE), test)
    print(f"Test set ready: TWAP {twap.mean():+.3f}, true model + planner {true.mean():+.3f} bps", flush=True)

    rows, arrays = [], {d: {BEST_RULE: [], FITTED: [], LEARNED: []} for d in D_VALUES}
    for r in range(N_RUNS):
        pool = simulate(BASE, POOL_SIZE, TRAIN_SEED0 + r)
        for d in D_VALUES:
            train = subset(pool, d)
            beta, best_rule = tune_rules(BASE, train)
            urgency = cost(UrgencyRule(BASE, beta), test, BASE)
            best = twap if best_rule == "TWAP" else urgency

            fitted, fit_info = fit_linear_gaussian(train, BASE)
            fitted_cost = planner_cost(KalmanForecaster(fitted), test)
            fitted_scores = forecast_scores(KalmanForecaster(fitted), test, realized, true_forecasts)

            t0 = time.time()
            model, scales, info = train_world_model(train, seed=r)
            train_seconds = time.time() - t0
            learned_cost = planner_cost(LearnedForecaster(model, scales, BASE.n_steps), test)
            learned_scores = forecast_scores(
                LearnedForecaster(model, scales, BASE.n_steps), test, realized, true_forecasts)

            for name, c in [(BEST_RULE, best), (FITTED, fitted_cost), (LEARNED, learned_cost)]:
                arrays[d][name].append(c)
            row = {
                "run": r, "train_hours": d, "best_rule": best_rule, "beta": beta,
                "best_rule_bps": best.mean(), "fitted_bps": fitted_cost.mean(), "learned_bps": learned_cost.mean(),
                "best_rule_cvar95": cvar95(best), "fitted_cvar95": cvar95(fitted_cost),
                "learned_cvar95": cvar95(learned_cost),
                "fitted_half_life": fitted.half_life, "fitted_alpha_std": fitted.alpha_std,
                "fitted_obs_std": fitted.obs_std, "fitted_sigma": fitted.sigma, "fitted_converged": fit_info["converged"],
            }
            row.update({f"fitted_{key}": value for key, value in fitted_scores.items()})
            row.update({f"learned_{key}": value for key, value in learned_scores.items()})
            row.update({"learned_steps": info["steps"], "learned_train_seconds": train_seconds,
                        "learned_parameters": info["parameters"]})
            rows.append(row)
            print(f"  run {r} D={d:4d}: best rule ({best_rule}) {best.mean():+.3f} | fitted {fitted_cost.mean():+.3f} "
                  f"| learned {learned_cost.mean():+.3f} (R2 to deadline {learned_scores['r2_deadline']:.4f}, "
                  f"{train_seconds:.0f}s) bps", flush=True)

    summary = summarize(rows, arrays, twap, true, true_r2)
    write_csv(RESULTS / "stage2_runs.csv", rows)
    write_csv(RESULTS / "stage2_summary.csv", summary)
    write_markdown(RESULTS / "stage2.md", rows, summary, time.time() - start)
    print(f"Done in {time.time() - start:.0f} s. See {RESULTS / 'stage2.md'}")


def summarize(rows, arrays, twap, true, true_r2):
    per_run_keys = (["best_rule_cvar95", "fitted_cvar95", "learned_cvar95",
                     "fitted_half_life", "fitted_alpha_std", "fitted_obs_std",
                     "fitted_rmse_to_true", "learned_rmse_to_true", "learned_steps", "learned_train_seconds"]
                    + [f"{model}_r2_{key}" for model in ("fitted", "learned") for key in HORIZONS])
    out = []
    for d in D_VALUES:
        avg = {name: np.mean(arrays[d][name], axis=0) for name in arrays[d]}  # per test hour, averaged over runs
        runs = [row for row in rows if row["train_hours"] == d]
        record = {"train_hours": d, "twap_bps": twap.mean(), "true_bps": true.mean(),
                  "twap_cvar95": cvar95(twap), "true_cvar95": cvar95(true)}
        record.update({f"true_r2_{key}": value for key, value in true_r2.items()})
        for key, name in [("best_rule", BEST_RULE), ("fitted", FITTED), ("learned", LEARNED)]:
            record[f"{key}_bps"] = avg[name].mean()
        record["best_rule_is_twap_runs"] = sum(row["best_rule"] == "TWAP" for row in runs)
        comparisons = {
            "learned_saving_vs_best_rule": (avg[BEST_RULE], avg[LEARNED]),  # H1, positive = learned cheaper
            "fitted_saving_vs_best_rule": (avg[BEST_RULE], avg[FITTED]),
            "learned_cost_of_model_error": (avg[LEARNED], true),  # H3, positive = worse than the true model
            "fitted_cost_of_model_error": (avg[FITTED], true),
            "learned_minus_fitted": (avg[LEARNED], avg[FITTED]),  # H4, positive = learned costlier
        }
        for key, (a, b) in comparisons.items():
            diff = paired_diff(a, b)
            record[key] = diff.mean
            record[f"{key}_low"] = diff.low
            record[f"{key}_high"] = diff.high
        per_run = [row["best_rule_bps"] - row["learned_bps"] for row in runs]
        record["learned_saving_vs_best_rule_run_min"] = min(per_run)
        record["learned_saving_vs_best_rule_run_max"] = max(per_run)
        for key in per_run_keys:
            values = [row[key] for row in runs]
            record[key] = float(np.mean(values))
            record[f"{key}_min"] = float(np.min(values))
            record[f"{key}_max"] = float(np.max(values))
        out.append(record)
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ci(rec, key):
    return f"{rec[key]:+.2f} [{rec[key + '_low']:+.2f}, {rec[key + '_high']:+.2f}]"


def _mm(rec, key, fmt):
    return f"{rec[key]:{fmt}} [{rec[key + '_min']:{fmt}}, {rec[key + '_max']:{fmt}}]"


def write_markdown(path, rows, summary, elapsed):
    n_params = rows[0]["learned_parameters"]
    first = summary[0]
    lines = [
        "# Stage 2 results (generated by run_stage2.py)",
        "",
        f"True market: Stage 1 base case. Training: {N_RUNS} runs (seeds {TRAIN_SEED0}–{TRAIN_SEED0 + N_RUNS - 1}), "
        f"using the first D of 1,000 simulated hours. Test: {N_TEST:,} hours from the reserved seed {TEST_SEED}. "
        "Costs are mean implementation shortfall in bps of arrival notional on the test hours, averaged over runs. "
        "Intervals are paired 95% intervals over test hours for the run-averaged costs; they do not include "
        f"run-to-run variation, which is shown separately. Learned model: {n_params:,} parameters. Runtime {elapsed:.0f} s.",
        "",
        "## Execution cost on test hours (bps, lower is better)",
        "",
        "| Training hours | TWAP | Best simple rule | Fitted classical model + planner | Learned model + planner | True model + planner (ceiling) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(f"| {s['train_hours']:,} | {s['twap_bps']:+.2f} | {s['best_rule_bps']:+.2f} "
                     f"(TWAP in {s['best_rule_is_twap_runs']}/{N_RUNS} runs) | {s['fitted_bps']:+.2f} | "
                     f"{s['learned_bps']:+.2f} | {s['true_bps']:+.2f} |")
    lines += [
        "",
        "## Hypothesis comparisons (bps, 95% CI)",
        "",
        "Saving = best simple rule cost − strategy cost (positive: the strategy is cheaper). "
        "Cost of model error = strategy cost − true model + planner cost (positive: worse than the ceiling). "
        "Learned − fitted: positive means the learned model is costlier.",
        "",
        "| Training hours | H1: learned saving vs best rule | Range over runs | Fitted saving vs best rule | "
        "H3: learned cost of model error | Fitted cost of model error | H4: learned − fitted |",
        "|---:|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['train_hours']:,} | {_ci(s, 'learned_saving_vs_best_rule')} | "
            f"{s['learned_saving_vs_best_rule_run_min']:+.2f} to {s['learned_saving_vs_best_rule_run_max']:+.2f} | "
            f"{_ci(s, 'fitted_saving_vs_best_rule')} | {_ci(s, 'learned_cost_of_model_error')} | "
            f"{_ci(s, 'fitted_cost_of_model_error')} | {_ci(s, 'learned_minus_fitted')} |")
    lines += [
        "",
        "## Tail cost on test hours: CVaR95 (bps, lower is better)",
        "",
        "Mean cost of the worst 5% of test hours, computed per run and averaged over runs, with [min, max].",
        "",
        "| Training hours | TWAP | Best simple rule | Fitted classical model + planner | Learned model + planner | True model + planner |",
        "|---:|---:|---|---|---|---:|",
    ]
    for s in summary:
        lines.append(f"| {s['train_hours']:,} | {s['twap_cvar95']:.2f} | {_mm(s, 'best_rule_cvar95', '.2f')} | "
                     f"{_mm(s, 'fitted_cvar95', '.2f')} | {_mm(s, 'learned_cvar95', '.2f')} | {s['true_cvar95']:.2f} |")
    lines += [
        "",
        "## Forecast quality on test hours (reported separately from cost)",
        "",
        "Out-of-sample R² of the forecast price change over the next 1, 5 and 15 minutes and to the deadline, "
        "relative to forecasting no change, pooled over all test hours and decision minutes. The true model's R² "
        "is the best achievable. RMSE to true forecast = distance from the true model's forecast to the deadline, "
        "in bps. Mean over runs, with [min, max].",
        "",
        "| Training hours | Model | 1 min | 5 min | 15 min | To deadline | RMSE to true forecast |",
        "|---:|---|---|---|---|---|---|",
        f"| — | True model | {first['true_r2_1']:.4f} | {first['true_r2_5']:.4f} | {first['true_r2_15']:.4f} | "
        f"{first['true_r2_deadline']:.4f} | 0 |",
    ]
    for s in summary:
        for model, label in [("fitted", "Fitted classical model"), ("learned", "Learned model")]:
            cells = " | ".join(_mm(s, f"{model}_r2_{key}", ".4f") for key in HORIZONS)
            lines.append(f"| {s['train_hours']:,} | {label} | {cells} | {_mm(s, f'{model}_rmse_to_true', '.2f')} |")
    lines += [
        "",
        "## Training budget of the learned model (mean over runs, [min, max])",
        "",
        "| Training hours | Gradient steps | Wall-clock time (s) |",
        "|---:|---|---|",
    ]
    for s in summary:
        lines.append(f"| {s['train_hours']:,} | {_mm(s, 'learned_steps', '.0f')} | "
                     f"{_mm(s, 'learned_train_seconds', '.0f')} |")
    lines += [
        "",
        "## Fitted classical model: parameter recovery (mean over runs, [min, max])",
        "",
        f"True values: half-life {BASE.half_life:g} steps, trend size {BASE.alpha_std:.3f} bps, "
        f"signal noise {BASE.obs_std:.3f} bps.",
        "",
        "| Training hours | Half-life | Trend size | Signal noise |",
        "|---:|---|---|---|",
    ]
    for s in summary:
        lines.append(f"| {s['train_hours']:,} | {_mm(s, 'fitted_half_life', '.2f')} | "
                     f"{_mm(s, 'fitted_alpha_std', '.3f')} | {_mm(s, 'fitted_obs_std', '.3f')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
