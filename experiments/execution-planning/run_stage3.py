"""Stage 3: learned vs classical models when the market breaks the classical model's assumptions.

Prespecified in PLAN.md ("Stage 3 prespecification"). Run from this folder, in order:
    python run_stage3.py gate       Gate C1 on validation hours (seed 212) -> results/stage3_gate.*
    python run_stage3.py select     larger-variant selection on validation hours -> results/stage3_selection.*
    python run_stage3.py main A     final run on test hours (seed 313); one scenario per call (A, B, C, D);
                                    intermediate files go to results/stage3_work/ (not committed)
    python run_stage3.py report     -> results/stage3.md, stage3_runs.csv, stage3_summary.csv
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from evaluate import paired_diff
from fit_classical import fit_linear_gaussian
from market import MarketParams
from policies import TWAP, KalmanForecaster, UrgencyRule
from run_stage1 import cost, tune_rules
from run_stage2 import cvar95, planner_cost, subset
from scenarios import BayesForecaster, bayes_trend, fit_regime_aware, particle_trend, scenario, simulate_scenario
from world_model import LearnedForecaster, train_world_model

BASE = MarketParams()
MAIN, CONTROL, REGIME_AWARE = ["A", "B", "C"], "D", ("A", "B")
D_VALUES, PRIMARY_D = [200, 1000, 5000], 1000
N_RUNS, TRAIN_SEED0, POOL_SIZE = 5, 111, 5000
VALIDATION_SEED, N_VALIDATION = 212, 20_000
TEST_SEED, N_TEST = 313, 20_000
METRIC_HOURS = 5000  # forecast metrics use the first 5,000 test hours
GATE_MIN_GAP = 0.15
BONFERRONI_Z = 2.394  # two-sided 98.33% interval (0.05 / 3 scenarios)
CANDIDATES = {"GRU 32": {"hidden": 32, "latent": 16, "width": 64}, "GRU 64": {"hidden": 64, "latent": 16, "width": 64}}
PF_HOURS, PF_PARTICLES = 1000, 2000
HORIZONS = (1, 5, 15)
WINDOW = 30
RESULTS = Path(__file__).parent / "results"
WORK = RESULTS / "stage3_work"
SCENARIO_NAMES = {"A": "Signal on/off", "B": "Fast or slow trends", "C": "Signal outliers", "D": "Price outliers (negative control)"}
STRATEGIES = {
    "best_rule": "Best simple rule", "fitted": "Fitted classical model + planner",
    "regime_aware": "Fitted regime-aware model + planner", "learned": "Learned model + planner",
    "larger": "Larger learned model + planner", "true": "True model + planner",
}


def r2(estimate, target):
    return 1.0 - float(((target - estimate) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def training_pool(sc, r):
    return simulate_scenario(sc, POOL_SIZE, TRAIN_SEED0 + r)[0]


def best_rule_cost(train, paths, twap_cost=None):
    beta, best_rule = tune_rules(BASE, train)
    if best_rule == "TWAP":
        return best_rule, beta, (cost(TWAP(BASE), paths, BASE) if twap_cost is None else twap_cost)
    return best_rule, beta, cost(UrgencyRule(BASE, beta), paths, BASE)


def forecast_metrics(forecaster, reference, paths):
    """Forecast quality against realized price changes, and against the reference (true model) forecast."""
    e, n = paths.price.shape
    forecaster.reset(e)
    reference.reset(e)
    keys = list(HORIZONS) + ["deadline"]
    sse, sst = dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0.0)
    acc = dict.fromkeys(["fg", "ff", "gg", "f", "g", "d2", "n"], 0.0)
    size_num, size_den = np.zeros(WINDOW), np.zeros(WINDOW)
    for t in range(n - 1):
        f = forecaster.update(t, paths.price[:, t], paths.signal[:, t])
        g = reference.update(t, paths.price[:, t], paths.signal[:, t])
        cumulative = np.cumsum(f, axis=1)
        for h in keys:
            end = n - 1 if h == "deadline" else t + h
            if end > n - 1:
                continue
            realized = paths.price[:, end] - paths.price[:, t]
            sse[h] += float(((realized - cumulative[:, end - t - 1]) ** 2).sum())
            sst[h] += float((realized**2).sum())
        if t <= n - 1 - WINDOW:
            size_num += (f[:, :WINDOW] * g[:, :WINDOW]).sum(axis=0)
            size_den += (g[:, :WINDOW] ** 2).sum(axis=0)
            ft, gt = f[:, :WINDOW].sum(axis=1), g[:, :WINDOW].sum(axis=1)
            for k, v in [("fg", ft @ gt), ("ff", ft @ ft), ("gg", gt @ gt), ("f", ft.sum()), ("g", gt.sum()),
                         ("d2", ((ft - gt) ** 2).sum()), ("n", ft.size)]:
                acc[k] += float(v)
    out = {f"r2_{h}": 1.0 - sse[h] / sst[h] for h in keys}
    mf, mg = acc["f"] / acc["n"], acc["g"] / acc["n"]
    var_f, var_g = acc["ff"] / acc["n"] - mf**2, acc["gg"] / acc["n"] - mg**2
    out["total30_corr"] = (acc["fg"] / acc["n"] - mf * mg) / float(np.sqrt(var_f * var_g)) if var_f > 0 else 0.0
    out["total30_size"] = acc["fg"] / acc["gg"]
    out["total30_rel_error"] = acc["d2"] / acc["gg"]
    sizes = size_num / size_den
    out.update(size_1=float(sizes[0]), size_10=float(sizes[9]), size_20=float(sizes[19]), size_30=float(sizes[29]))
    return out


def _ci(d, z=1.96):
    return d.mean, d.mean - z * d.se, d.mean + z * d.se


def _write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- Gate C1 (validation hours)

def gate():
    rows, decisions = [], {}
    for key in MAIN + [CONTROL]:
        for severe in ((False, True) if key in MAIN else (False,)):
            t0 = time.time()
            sc = scenario(key, severe)
            train = subset(training_pool(sc, 0), PRIMARY_D)
            val, _ = simulate_scenario(sc, N_VALIDATION, VALIDATION_SEED)
            fitted, _ = fit_linear_gaussian(train, BASE)
            true = planner_cost(BayesForecaster(sc), val)
            fitted_cost = planner_cost(KalmanForecaster(fitted), val)
            best_rule, beta, best = best_rule_cost(train, val)
            gap, headroom = paired_diff(fitted_cost, true), paired_diff(best, true)
            passed = bool(gap.mean >= GATE_MIN_GAP and gap.low > 0)
            row = {"scenario": key, "setting": "severe" if severe else "prespecified", "true_bps": true.mean(),
                   "fitted_bps": fitted_cost.mean(), "best_rule": best_rule, "best_rule_bps": best.mean(),
                   "gap_bps": gap.mean, "gap_low": gap.low, "gap_high": gap.high,
                   "headroom_bps": headroom.mean, "headroom_low": headroom.low, "headroom_high": headroom.high,
                   "passed": passed, "fitted_half_life": fitted.half_life, "fitted_alpha_std": fitted.alpha_std,
                   "fitted_obs_std": fitted.obs_std}
            if key in REGIME_AWARE:
                aware, info = fit_regime_aware(train, key, fitted)
                d = paired_diff(planner_cost(BayesForecaster(aware), val), true)
                row.update(regime_aware_gap_bps=d.mean, regime_aware_gap_low=d.low, regime_aware_gap_high=d.high,
                           regime_aware_converged=info["converged"])
            if len(sc.regimes) == 1:
                sub = subset(val, PF_HOURS)
                gs, pf = bayes_trend(sc, sub), particle_trend(sc, sub, PF_PARTICLES, seed=0)
                target = sub.alpha.ravel()
                row.update(moment_matched_trend_r2=r2(gs.ravel(), target), particle_trend_r2=r2(pf.ravel(), target),
                           filters_corr=float(np.corrcoef(gs.ravel(), pf.ravel())[0, 1]))
            rows.append(row)
            print(f"{key} ({row['setting']}): gap {gap.mean:+.3f} [{gap.low:+.3f}, {gap.high:+.3f}] -> "
                  f"{'PASS' if passed else 'fail'} | headroom {headroom.mean:+.3f} | {time.time() - t0:.0f}s", flush=True)
            if passed or key not in MAIN:
                break
        decisions[key] = {"severe": severe, "passed": passed}
    (RESULTS / "stage3_gate.json").write_text(json.dumps({"decisions": decisions, "rows": rows}, indent=1))
    _write_csv(RESULTS / "stage3_gate.csv", rows)
    print("Gate decisions:", decisions)


# ---------------------------------------------------------------- larger-variant selection (validation hours)

def select():
    decisions = json.loads((RESULTS / "stage3_gate.json").read_text())["decisions"]
    eligible = [k for k in MAIN if decisions[k]["passed"]]
    rows = []
    for name, arch in CANDIDATES.items():
        for key in eligible:
            t0 = time.time()
            sc = scenario(key, decisions[key]["severe"])
            train = subset(training_pool(sc, 0), PRIMARY_D)
            val, _ = simulate_scenario(sc, N_VALIDATION, VALIDATION_SEED)
            model, scales, info = train_world_model(train, seed=0, architecture=arch)
            c = float(planner_cost(LearnedForecaster(model, scales, BASE.n_steps), val).mean())
            rows.append({"candidate": name, "scenario": key, "validation_bps": c, "parameters": info["parameters"],
                         "steps": info["steps"], "seconds": time.time() - t0})
            print(f"{name} on {key}: validation cost {c:+.3f} bps ({info['parameters']} parameters, {info['steps']} steps)",
                  flush=True)
    means = {name: float(np.mean([row["validation_bps"] for row in rows if row["candidate"] == name]))
             for name in CANDIDATES} if eligible else {}
    chosen = min(CANDIDATES, key=lambda n: (means[n], list(CANDIDATES).index(n))) if eligible else None
    out = {"eligible": eligible, "means": means, "chosen": chosen,
           "architecture": CANDIDATES.get(chosen), "rows": rows}
    (RESULTS / "stage3_selection.json").write_text(json.dumps(out, indent=1))
    print("Selected:", chosen, means)


# ---------------------------------------------------------------- final run (test hours), one scenario per call

def main_runs(key):
    WORK.mkdir(parents=True, exist_ok=True)
    decisions = json.loads((RESULTS / "stage3_gate.json").read_text())["decisions"]
    selection = json.loads((RESULTS / "stage3_selection.json").read_text())
    sc = scenario(key, decisions[key]["severe"])
    if key in MAIN and not decisions[key]["passed"]:
        (WORK / f"{key}.json").write_text(json.dumps({"scenario": key, "skipped": True}))
        print(f"{key}: failed Gate C1; no learned models trained")
        return
    sizes = D_VALUES if key in MAIN else [PRIMARY_D]
    learners = {"learned": None}
    if key in MAIN and selection["chosen"]:
        learners["larger"] = selection["architecture"]
    test, regime = simulate_scenario(sc, N_TEST, TEST_SEED)
    metric_paths = subset(test, METRIC_HOURS)
    true = planner_cost(BayesForecaster(sc), test)
    twap = cost(TWAP(BASE), test, BASE)
    summary = {"scenario": key, "skipped": False, "severe": sc.severe, "labels": list(sc.labels),
               "regime_share": [float((regime == k).mean()) for k in range(len(sc.regimes))],
               "true_bps": float(true.mean()), "twap_bps": float(twap.mean()),
               "true_cvar95": cvar95(true), "twap_cvar95": cvar95(twap),
               "true_by_regime": [float(true[regime == k].mean()) for k in range(len(sc.regimes))],
               "twap_by_regime": [float(twap[regime == k].mean()) for k in range(len(sc.regimes))],
               "true_metrics": forecast_metrics(BayesForecaster(sc), BayesForecaster(sc), metric_paths)}
    sums, rows = {}, []
    for r in range(N_RUNS):
        pool = training_pool(sc, r)
        for d in sizes:
            t_cell = time.time()
            train = subset(pool, d)
            best_rule, beta, best = best_rule_cost(train, test, twap)
            costs = {"best_rule": best}
            fitted, _ = fit_linear_gaussian(train, BASE)
            makers = {"fitted": lambda f=fitted: KalmanForecaster(f)}
            row = {"scenario": key, "run": r, "train_hours": d, "best_rule": best_rule, "beta": beta,
                   "fitted_half_life": fitted.half_life, "fitted_alpha_std": fitted.alpha_std, "fitted_obs_std": fitted.obs_std}
            if key in REGIME_AWARE:
                aware, info = fit_regime_aware(train, key, fitted)
                makers["regime_aware"] = lambda a=aware: BayesForecaster(a)
                row.update(regime_aware_converged=info["converged"], regime_aware_p0=aware.probs[0])
            for name, arch in learners.items():
                t0 = time.time()
                model, scales, info = train_world_model(train, seed=r, architecture=arch)
                row.update({f"{name}_steps": info["steps"], f"{name}_train_seconds": time.time() - t0,
                            f"{name}_parameters": info["parameters"]})
                makers[name] = lambda m=model, s=scales: LearnedForecaster(m, s, BASE.n_steps)
            for name, make in makers.items():
                costs[name] = planner_cost(make(), test)
                row.update({f"{name}_{k}": v for k, v in forecast_metrics(make(), BayesForecaster(sc), metric_paths).items()})
            for name, c in costs.items():
                row[f"{name}_bps"], row[f"{name}_cvar95"] = float(c.mean()), cvar95(c)
                for k in range(len(sc.regimes)):
                    row[f"{name}_bps_regime{k}"] = float(c[regime == k].mean())
                sums[(name, d)] = sums.get((name, d), 0.0) + c
            rows.append(row)
            print(f"{key} run {r} D={d:5d}: " + " | ".join(f"{name} {costs[name].mean():+.3f}" for name in costs)
                  + f" | true {true.mean():+.3f} | {time.time() - t_cell:.0f}s", flush=True)
    summary["rows"] = rows
    (WORK / f"{key}.json").write_text(json.dumps(summary, indent=1))
    np.savez(WORK / f"{key}.npz", true=true, twap=twap, regime=regime,
             **{f"{name}_{d}": s / N_RUNS for (name, d), s in sums.items()})


# ---------------------------------------------------------------- report

def _fmt(mean, low, high):
    return f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"


def _mm(values, fmt):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():{fmt}} [{v.min():{fmt}}, {v.max():{fmt}}]"


def report():
    gate_data = json.loads((RESULTS / "stage3_gate.json").read_text())
    selection = json.loads((RESULTS / "stage3_selection.json").read_text())
    lines = ["# Stage 3 results (generated by run_stage3.py)", "",
             "Prespecified in [PLAN.md](../PLAN.md#stage-3-prespecification). Costs are mean implementation shortfall in bps "
             f"of arrival notional on {N_TEST:,} test hours (seed {TEST_SEED}), averaged over {N_RUNS} runs (training seeds "
             f"{TRAIN_SEED0}–{TRAIN_SEED0 + N_RUNS - 1}). Intervals are paired over test hours for run-averaged costs; "
             "run-to-run ranges are shown separately. Forecast metrics use the first "
             f"{METRIC_HOURS:,} test hours.", "",
             f"## Gate C1: misspecification headroom (validation hours, seed {VALIDATION_SEED})", "",
             f"Gap = fitted classical + planner cost − true model + planner cost, with the classical model fitted on run 0's "
             f"{PRIMARY_D:,} training hours. Pass: gap ≥ {GATE_MIN_GAP} bps and interval above zero. Headroom = best simple "
             "rule − true model + planner.", "",
             "| Scenario | Setting | Gap (bps) [95% CI] | Gate | Headroom over best rule | Regime-aware model's gap | Filter check (trend R², moment-matched / particle) |",
             "|---|---|---|---|---|---|---|"]
    for row in gate_data["rows"]:
        aware = _fmt(row["regime_aware_gap_bps"], row["regime_aware_gap_low"], row["regime_aware_gap_high"]) \
            if "regime_aware_gap_bps" in row else "—"
        check = f"{row['moment_matched_trend_r2']:.3f} / {row['particle_trend_r2']:.3f}" if "particle_trend_r2" in row else "—"
        lines.append(f"| {row['scenario']} {SCENARIO_NAMES[row['scenario']]} | {row['setting']} | "
                     f"{_fmt(row['gap_bps'], row['gap_low'], row['gap_high'])} | {'**pass**' if row['passed'] else 'fail'} | "
                     f"{_fmt(row['headroom_bps'], row['headroom_low'], row['headroom_high'])} | {aware} | {check} |")
    lines += ["", "## Larger learned model: selection (validation hours)", ""]
    if selection["chosen"]:
        lines += ["| Candidate | " + " | ".join(selection["eligible"]) + " | Mean |",
                  "|---|" + "---:|" * (len(selection["eligible"]) + 1)]
        for name in CANDIDATES:
            vals = [row["validation_bps"] for key in selection["eligible"] for row in selection["rows"]
                    if row["candidate"] == name and row["scenario"] == key]
            params = next(row["parameters"] for row in selection["rows"] if row["candidate"] == name)
            lines.append(f"| {name} ({params:,} parameters) | " + " | ".join(f"{v:+.3f}" for v in vals)
                         + f" | {selection['means'][name]:+.3f} |")
        lines += ["", f"Selected: **{selection['chosen']}**."]
    else:
        lines.append("No main scenario passed Gate C1, so no larger model was trained.")

    runs_all, summary_rows, verdicts = [], [], []
    for key in MAIN + [CONTROL]:
        part = WORK / f"{key}.json"
        if not part.exists():
            continue
        data = json.loads(part.read_text())
        lines += ["", f"## Scenario {key}: {SCENARIO_NAMES[key]}", ""]
        if data["skipped"]:
            lines.append("Failed Gate C1 (also at the severe setting); no learned models were trained.")
            continue
        arrays = np.load(WORK / f"{key}.npz")
        rows = data["rows"]
        runs_all += rows
        sizes = sorted({row["train_hours"] for row in rows})
        names = [n for n in ["best_rule", "fitted", "regime_aware", "learned", "larger"] if f"{n}_{sizes[0]}" in arrays]
        true = arrays["true"]
        lines += [f"Setting: {'severe' if data['severe'] else 'prespecified'}. TWAP {data['twap_bps']:+.2f} bps; "
                  f"true model + planner {data['true_bps']:+.2f} bps.", "",
                  "| Training hours | " + " | ".join(STRATEGIES[n] for n in names) + " | True model + planner |",
                  "|---:|" + "---:|" * (len(names) + 1)]
        for d in sizes:
            lines.append(f"| {d:,} | " + " | ".join(f"{arrays[f'{n}_{d}'].mean():+.2f}" for n in names) + f" | {true.mean():+.2f} |")
        lines += ["", "**Comparisons** (bps, paired 95% CI; negative = the first strategy is cheaper):", "",
                  "| Training hours | Learned − fitted classical | Learned − best rule | Learned − true model |"
                  + (" Larger − fitted classical | Larger − learned |" if "larger" in names else "")
                  + (" Regime-aware − learned |" if "regime_aware" in names else ""),
                  "|---:|---|---|---|" + ("---|---|" if "larger" in names else "") + ("---|" if "regime_aware" in names else "")]
        for d in sizes:
            a = {n: arrays[f"{n}_{d}"] for n in names}
            cells = [_fmt(*_ci(paired_diff(a["learned"], a["fitted"]))), _fmt(*_ci(paired_diff(a["learned"], a["best_rule"]))),
                     _fmt(*_ci(paired_diff(a["learned"], true)))]
            if "larger" in names:
                cells += [_fmt(*_ci(paired_diff(a["larger"], a["fitted"]))), _fmt(*_ci(paired_diff(a["larger"], a["learned"])))]
            if "regime_aware" in names:
                cells.append(_fmt(*_ci(paired_diff(a["regime_aware"], a["learned"]))))
            lines.append(f"| {d:,} | " + " | ".join(cells) + " |")
            record = {"scenario": key, "train_hours": d, "true_bps": float(true.mean()), "twap_bps": data["twap_bps"]}
            record.update({f"{n}_bps": float(a[n].mean()) for n in names})
            for label, (x, y) in {"learned_minus_fitted": ("learned", "fitted"), "learned_minus_best_rule": ("learned", "best_rule"),
                                  "larger_minus_fitted": ("larger", "fitted"), "regime_aware_minus_learned": ("regime_aware", "learned")}.items():
                if x in a and y in a:
                    m, lo, hi = _ci(paired_diff(a[x], a[y]))
                    record.update({label: m, f"{label}_low": lo, f"{label}_high": hi})
            summary_rows.append(record)
        if key in MAIN and PRIMARY_D in sizes:
            diff = paired_diff(arrays[f"learned_{PRIMARY_D}"], arrays[f"fitted_{PRIMARY_D}"])
            m, lo, hi = _ci(diff)
            _, blo, bhi = _ci(diff, BONFERRONI_Z)
            verdict = "supported" if hi < 0 else "not supported"
            verdicts.append((key, m, lo, hi, blo, bhi, verdict, bhi < 0))
            lines += ["", f"**H4b at {PRIMARY_D:,} training hours:** learned − fitted classical = {_fmt(m, lo, hi)} bps "
                      f"(Bonferroni 98.3%: [{blo:+.2f}, {bhi:+.2f}]): **{verdict}**"
                      + (" and survives the correction." if bhi < 0 else ".")]
        if len(data["labels"]) > 1:
            lines += ["", f"**Costs by regime at {PRIMARY_D:,} training hours** (bps):", "",
                      "| Regime (share of hours) | " + " | ".join(STRATEGIES[n] for n in names) + " | True model + planner |",
                      "|---|" + "---:|" * (len(names) + 1)]
            regime = arrays["regime"]
            for k, label in enumerate(data["labels"]):
                cells = [f"{arrays[f'{n}_{PRIMARY_D}'][regime == k].mean():+.2f}" for n in names]
                lines.append(f"| {label} ({data['regime_share'][k]:.0%}) | " + " | ".join(cells) + f" | {true[regime == k].mean():+.2f} |")
        model_names = [n for n in ["fitted", "regime_aware", "learned", "larger"] if n in names]
        lines += ["", "**Forecast quality** (mean over runs): R² of the price change 1, 5 and 15 minutes ahead and to the deadline; "
                  "the 30-minute forecast's correlation, size (1 = right) and relative error against the true model's forecast.", "",
                  "| Training hours | Model | R² 1 / 5 / 15 min / deadline | 30-min correlation | Size | Relative error |",
                  "|---:|---|---|---|---|---|"]
        tm = data["true_metrics"]
        lines.append(f"| — | True model | {tm['r2_1']:.4f} / {tm['r2_5']:.4f} / {tm['r2_15']:.4f} / {tm['r2_deadline']:.4f} | 1 | 1 | 0 |")
        for d in sizes:
            sel = [row for row in rows if row["train_hours"] == d]
            for n in model_names:
                r2s = " / ".join(f"{np.mean([row[f'{n}_r2_{h}'] for row in sel]):.4f}" for h in list(HORIZONS) + ["deadline"])
                lines.append(f"| {d:,} | {STRATEGIES[n].replace(' + planner', '')} | {r2s} | "
                             f"{_mm([row[f'{n}_total30_corr'] for row in sel], '.3f')} | {_mm([row[f'{n}_total30_size'] for row in sel], '.2f')} | "
                             f"{_mm([row[f'{n}_total30_rel_error'] for row in sel], '.2f')} |")
        lines += ["", "**Tail cost and training budget** (mean over runs [min, max]):", "",
                  "| Training hours | CVaR95: " + " | ".join(STRATEGIES[n] for n in names) + " | Learned: steps, seconds |"
                  + (" Larger: steps, seconds |" if "larger" in names else ""),
                  "|---:|" + "---|" * len(names) + "---|" + ("---|" if "larger" in names else "")]
        for d in sizes:
            sel = [row for row in rows if row["train_hours"] == d]
            cells = [_mm([row[f"{n}_cvar95"] for row in sel], ".1f") for n in names]
            budget = [f"{np.mean([row[f'{x}_steps'] for row in sel]):.0f}, {np.mean([row[f'{x}_train_seconds'] for row in sel]):.0f}"
                      for x in (["learned", "larger"] if "larger" in names else ["learned"])]
            lines.append(f"| {d:,} | " + " | ".join(cells + budget) + " |")
        lines.append(f"\nTWAP CVaR95 {data['twap_cvar95']:.1f}; true model + planner CVaR95 {data['true_cvar95']:.1f}.")
    if verdicts:
        lines += ["", "## Summary of the primary hypothesis (H4b)", "",
                  "| Scenario | Learned − fitted classical at 1,000 hours (bps) [95% CI] | Bonferroni 98.3% | Verdict |", "|---|---|---|---|"]
        for key, m, lo, hi, blo, bhi, verdict, survives in verdicts:
            lines.append(f"| {key} {SCENARIO_NAMES[key]} | {_fmt(m, lo, hi)} | [{blo:+.2f}, {bhi:+.2f}] | {verdict}"
                         + (", survives correction" if survives else "") + " |")
    (RESULTS / "stage3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if runs_all:
        _write_csv(RESULTS / "stage3_runs.csv", runs_all)
    if summary_rows:
        _write_csv(RESULTS / "stage3_summary.csv", summary_rows)
    print(f"Wrote {RESULTS / 'stage3.md'}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "gate":
        gate()
    elif command == "select":
        select()
    elif command == "main":
        main_runs(sys.argv[2])
    elif command == "report":
        report()
    else:
        raise SystemExit(__doc__)
