"""Stage 2 supplement: what has the learned world model learned about the hidden trend?

Run from this folder:  python probe_stage2.py
Writes results/stage2_probe.csv and results/stage2_probe.md. About 30 minutes on a laptop CPU:
it retrains the 15 Stage 2 learned models (same data and seeds) and checks that each one
reproduces its published test cost.

Measured on fresh simulated hours, where the hidden trend is known (probe fitting: seed 404,
scoring: seed 505):
1. Is the trend recoverable from the model's memory? A linear readout ("probe") from the memory
   to the true trend. Controls: the same probe on untrained networks of the same architecture,
   and on the raw latest signals.
2. Does the model use the trend correctly? Its forecast of the price change over the next
   30 minutes is compared with the true model's: correlation, size and relative error.
3. Which of these explains decision cost across the 15 models? Rank correlations with the cost
   above the true model + planner on the Stage 2 test hours. Exploratory.

Optional parallel use:  python probe_stage2.py --part DIR 50   (one process per training size)
                        python probe_stage2.py --combine DIR
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from fit_classical import fit_linear_gaussian
from market import simulate
from policies import TWAP, KalmanForecaster
from run_stage1 import cost
from run_stage2 import BASE, D_VALUES, N_RUNS, N_TEST, POOL_SIZE, RESULTS, TEST_SEED, TRAIN_SEED0, planner_cost, subset
from world_model import LatentWorldModel, LearnedForecaster, Scales, _observations, train_world_model

PROBE_FIT_SEED, PROBE_EVAL_SEED, N_PROBE = 404, 505, 2000
HORIZON = 30  # minutes ahead compared with the true model
N = BASE.n_steps
T = N - 1  # decision minutes 0..N-2
UNTRAINED_SEEDS = range(5)
SIGNAL_LAGS = (1, 3, 10)


def r2(estimate, target):
    return 1.0 - float(((target - estimate) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def kalman_trend(params, paths):
    """Kalman estimate of the current trend from data up to each decision minute, shape (E, T)."""
    forecaster = KalmanForecaster(params)
    forecaster.reset(paths.price.shape[0])
    out = np.zeros((paths.price.shape[0], T))
    for t in range(T):
        out[:, t] = forecaster.update(t, paths.price[:, t], paths.signal[:, t])[:, 0]
    return out


def learned_outputs(model, scales, paths):
    """Memory (E, T, 2 x hidden; both mirror branches) and price-change forecasts 1..HORIZON ahead (E, T, HORIZON)."""
    obs = torch.tensor(_observations(paths, scales), dtype=torch.float32)
    e = obs.shape[0]
    with torch.no_grad():
        states, _ = model.encoder(torch.cat([obs, -obs]))
        memory = torch.cat([states[:e], states[e:]], dim=-1)[:, :T]
        forecasts = model.predict(obs, torch.arange(T), HORIZON)[..., 0] * scales.price_change
    return memory.double().numpy(), forecasts.double().numpy()


def linear_probe(features_fit, target_fit):
    design = np.column_stack([features_fit, np.ones(len(features_fit))])
    coef, *_ = np.linalg.lstsq(design, target_fit, rcond=None)
    return lambda features: np.column_stack([features, np.ones(len(features))]) @ coef


def memory_probe_r2(model, scales, probe_paths):
    fit, scored = probe_paths
    memory_fit, _ = learned_outputs(model, scales, fit)
    memory_scored, _ = learned_outputs(model, scales, scored)
    width = memory_fit.shape[-1]
    probe = linear_probe(memory_fit.reshape(-1, width), fit.alpha[:, :T].ravel())
    return r2(probe(memory_scored.reshape(-1, width)), scored.alpha[:, :T].ravel())


def compare_forecasts(forecasts, true_trend, phi):
    """Compare price-change forecasts 1..HORIZON minutes ahead with the true model's.

    The true model forecasts phi**k times its trend estimate for the change k+1 minutes ahead.
    Only decision times whose whole horizon ends before the deadline are used. Size at a horizon
    is the regression slope of the forecast on the true forecast (1 = right size).
    """
    window = slice(0, T - HORIZON + 1)
    f = forecasts[:, window].reshape(-1, HORIZON)
    true_f = true_trend[:, window].reshape(-1, 1) * phi ** np.arange(HORIZON)[None, :]
    size = (f * true_f).sum(0) / (true_f**2).sum(0)
    total, true_total = f.sum(1), true_f.sum(1)
    return {
        "size_1min": float(size[0]), "size_10min": float(size[9]),
        "size_20min": float(size[19]), "size_30min": float(size[29]),
        "total_30min_size": float(total @ true_total / (true_total @ true_total)),
        "total_30min_corr": float(np.corrcoef(total, true_total)[0, 1]),
        "total_30min_rel_error": float(((total - true_total) ** 2).sum() / (true_total**2).sum()),
    }


def setup():
    published = {(int(row["run"]), int(row["train_hours"])): float(row["learned_bps"])
                 for row in csv.DictReader(open(RESULTS / "stage2_runs.csv"))}
    test = simulate(BASE, N_TEST, TEST_SEED)
    probe_paths = (simulate(BASE, N_PROBE, PROBE_FIT_SEED), simulate(BASE, N_PROBE, PROBE_EVAL_SEED))
    return published, test, probe_paths, kalman_trend(BASE, probe_paths[1])


def probe_model(d, r, published, test, probe_paths, true_trend):
    train = subset(simulate(BASE, POOL_SIZE, TRAIN_SEED0 + r), d)
    fitted, _ = fit_linear_gaussian(train, BASE)
    model, scales, _ = train_world_model(train, seed=r)  # identical to Stage 2: same data, same seed
    _, forecasts = learned_outputs(model, scales, probe_paths[1])
    alpha = probe_paths[1].alpha[:, :T].ravel()
    record = {
        "train_hours": d, "run": r, "published_cost_bps": published[(r, d)],
        "reproduced_cost_bps": float(planner_cost(LearnedForecaster(model, scales, N), test).mean()),
        "memory_probe_r2": memory_probe_r2(model, scales, probe_paths),
        "own_trend_estimate_r2": r2(forecasts[..., 0].ravel(), alpha),  # its 1-minute forecast estimates the trend
        "fitted_classical_trend_r2": r2(kalman_trend(fitted, probe_paths[1]).ravel(), alpha),
    }
    record.update(compare_forecasts(forecasts, true_trend, BASE.phi))
    return record


def run_sizes(sizes):
    published, test, probe_paths, true_trend = setup()
    records = []
    for d in sizes:
        for r in range(N_RUNS):
            t0 = time.time()
            records.append(probe_model(d, r, published, test, probe_paths, true_trend))
            rec = records[-1]
            print(f"D={d:4d} run {r}: cost {rec['reproduced_cost_bps']:+.3f} (published {rec['published_cost_bps']:+.3f}) "
                  f"| memory probe R2 {rec['memory_probe_r2']:.3f} | 30-min forecast corr {rec['total_30min_corr']:.3f} "
                  f"size {rec['total_30min_size']:.2f} | {time.time() - t0:.0f}s", flush=True)
    return records


def controls(test, probe_paths, true_trend):
    fit, scored = probe_paths
    alpha_fit, alpha_scored = fit.alpha[:, :T].ravel(), scored.alpha[:, :T].ravel()
    out = {
        "twap_cost_bps": float(cost(TWAP(BASE), test, BASE).mean()),
        "true_cost_bps": float(planner_cost(KalmanForecaster(BASE), test).mean()),
        "true_model_trend_r2": r2(true_trend.ravel(), alpha_scored),
        "untrained_memory_probe_r2": [],
    }
    for seed in UNTRAINED_SEEDS:
        train = subset(simulate(BASE, POOL_SIZE, TRAIN_SEED0 + seed), 200)  # input scaling as in training
        scales = Scales(price_change=float(np.diff(train.price, axis=1).std()), signal=float(train.signal.std()))
        torch.manual_seed(seed)
        out["untrained_memory_probe_r2"].append(memory_probe_r2(LatentWorldModel().eval(), scales, probe_paths))
    for k in SIGNAL_LAGS:
        def lagged(paths, k=k):
            return np.stack([np.pad(paths.signal, ((0, 0), (j, 0)))[:, :T] for j in range(k)], axis=-1).reshape(-1, k)
        out[f"latest_{k}_signals_r2"] = r2(linear_probe(lagged(fit), alpha_fit)(lagged(scored)), alpha_scored)
    return out


def _mm(values, fmt):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():{fmt}} [{values.min():{fmt}}, {values.max():{fmt}}]"


def write_outputs(records, ctrl, elapsed):
    records = sorted(records, key=lambda rec: (rec["train_hours"], rec["run"]))
    headroom = ctrl["twap_cost_bps"] - ctrl["true_cost_bps"]
    for rec in records:
        rec["share_of_headroom"] = (ctrl["twap_cost_bps"] - rec["reproduced_cost_bps"]) / headroom
        rec["cost_above_true_bps"] = rec["reproduced_cost_bps"] - ctrl["true_cost_bps"]
    with (RESULTS / "stage2_probe.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    by_d = {d: [rec for rec in records if rec["train_hours"] == d] for d in D_VALUES}
    max_diff = max(abs(rec["reproduced_cost_bps"] - rec["published_cost_bps"]) for rec in records)
    lines = [
        "# Stage 2 supplement: what the learned model learned about the trend (generated by probe_stage2.py)",
        "",
        f"The {len(records)} Stage 2 learned models were retrained with the same data and seeds. "
        + (f"Each reproduces its published test cost (largest difference {max_diff:.1e} bps). " if max_diff < 1e-6 else
           f"**Warning: reproduced test costs differ from the published ones by up to {max_diff:.3f} bps.** ")
        + f"Probes use {N_PROBE:,} fresh simulated hours for "
        f"fitting (seed {PROBE_FIT_SEED}) and {N_PROBE:,} for scoring (seed {PROBE_EVAL_SEED}); in simulation the hidden "
        f"trend is known. Runtime {elapsed:.0f} s.",
        "",
        "## 1. Is the trend recoverable from the model's memory?",
        "",
        "R² of the true hidden trend (1 = perfect). Memory rows use a linear readout fitted on the fitting hours. "
        "The true model's Kalman estimate is the best achievable from the observations.",
        "",
        "| Source | Trend R² |",
        "|---|---|",
        f"| True model (best achievable) | {ctrl['true_model_trend_r2']:.3f} |",
    ]
    for d in D_VALUES:
        lines.append(f"| Fitted classical model, {d:,} training hours | "
                     f"{_mm([rec['fitted_classical_trend_r2'] for rec in by_d[d]], '.3f')} |")
    for d in D_VALUES:
        lines.append(f"| Learned model's memory, {d:,} training hours | "
                     f"{_mm([rec['memory_probe_r2'] for rec in by_d[d]], '.3f')} |")
    lines.append(f"| **Untrained network's memory (control, {len(ctrl['untrained_memory_probe_r2'])} networks)** | "
                 f"**{_mm(ctrl['untrained_memory_probe_r2'], '.3f')}** |")
    for k in sorted(SIGNAL_LAGS, reverse=True):
        label = "Latest signal only" if k == 1 else f"Latest {k} signals"
        lines.append(f"| {label} | {ctrl[f'latest_{k}_signals_r2']:.3f} |")
    lines += [
        "",
        f"Mean [min, max] over the {N_RUNS} runs. If an untrained network's memory scores as high as a trained one, "
        "the probe cannot show what training added.",
        "",
        "## 2. Does the model use the trend correctly?",
        "",
        f"The learned model's forecasts compared with the true model's, at decision minutes whose next {HORIZON} minutes "
        "end before the deadline. Size: regression slope on the true forecast (1 = right size, below 1 = too timid, "
        "above 1 = too bold). Relative error: mean squared distance from the true 30-minute forecast divided by the "
        "true forecast's mean square (0 = identical). Own trend estimate: R² of the model's 1-minute price forecast "
        "as an estimate of the current trend. Mean [min, max] over runs.",
        "",
        "| Training hours | 30-min forecast: correlation | Size | Relative error | Size at 1 / 10 / 20 / 30 min ahead | Own trend estimate R² |",
        "|---:|---|---|---|---|---|",
    ]
    for d in D_VALUES:
        recs = by_d[d]
        sizes = " / ".join(f"{np.mean([rec[f'size_{h}min'] for rec in recs]):.2f}" for h in (1, 10, 20, 30))
        lines.append(f"| {d:,} | {_mm([rec['total_30min_corr'] for rec in recs], '.3f')} | "
                     f"{_mm([rec['total_30min_size'] for rec in recs], '.2f')} | "
                     f"{_mm([rec['total_30min_rel_error'] for rec in recs], '.2f')} | {sizes} | "
                     f"{_mm([rec['own_trend_estimate_r2'] for rec in recs], '.2f')} |")
    gap = np.array([rec["cost_above_true_bps"] for rec in records])
    measures = [
        ("Relative error of the 30-min forecast", [rec["total_30min_rel_error"] for rec in records]),
        ("Correlation of the 30-min forecast", [rec["total_30min_corr"] for rec in records]),
        ("Size error of the 30-min forecast, abs(size - 1)", [abs(rec["total_30min_size"] - 1) for rec in records]),
        ("Own trend estimate R²", [rec["own_trend_estimate_r2"] for rec in records]),
        ("Memory probe R²", [rec["memory_probe_r2"] for rec in records]),
    ]
    lines += [
        "",
        "## 3. What explains the decision cost? (exploratory)",
        "",
        f"Spearman rank correlation across the {len(records)} models between each measure and the model's cost above "
        f"the true model + planner on the Stage 2 test hours (seed {TEST_SEED}). The training sets are nested within "
        "runs, so the models are not fully independent.",
        "",
        "| Measure | Spearman | p-value |",
        "|---|---:|---:|",
    ]
    for name, values in measures:
        rho, p = spearmanr(values, gap)
        lines.append(f"| {name} | {rho:+.2f} | {p:.3f} |")
    lines += [
        "",
        "## Per-model values",
        "",
        "| Training hours | Run | Test cost (bps) | Share of headroom | 30-min forecast correlation | Size | Relative error | Memory probe R² |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        lines.append(f"| {rec['train_hours']:,} | {rec['run']} | {rec['reproduced_cost_bps']:+.3f} | "
                     f"{rec['share_of_headroom']:.0%} | {rec['total_30min_corr']:.3f} | {rec['total_30min_size']:.2f} | "
                     f"{rec['total_30min_rel_error']:.2f} | {rec['memory_probe_r2']:.3f} |")
    (RESULTS / "stage2_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start = time.time()
    args = sys.argv[1:]
    if args[:1] == ["--part"]:
        part_dir = Path(args[1])
        part_dir.mkdir(parents=True, exist_ok=True)
        sizes = [int(x) for x in args[2:]]
        records = run_sizes(sizes)
        (part_dir / f"part_{'_'.join(map(str, sizes))}.json").write_text(
            json.dumps({"records": records, "seconds": time.time() - start}))
        return
    if args[:1] == ["--combine"]:
        parts = [json.loads(p.read_text()) for p in sorted(Path(args[1]).glob("part_*.json"))]
        records = [rec for part in parts for rec in part["records"]]
        elapsed = sum(part["seconds"] for part in parts)
    else:
        records = run_sizes(D_VALUES)
        elapsed = None
    _, test, probe_paths, true_trend = setup()
    ctrl = controls(test, probe_paths, true_trend)
    write_outputs(records, ctrl, (elapsed or 0.0) + time.time() - start)
    print(f"Done. See {RESULTS / 'stage2_probe.md'}")


if __name__ == "__main__":
    main()
