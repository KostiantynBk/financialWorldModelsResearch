"""Stage 1: environment sanity (Gate A) and planning headroom (Gate B). No training.

Run from this folder:  python run_stage1.py
Writes results/gate_a.csv, results/gate_b.csv and results/stage1.md.

Splits: the urgency rule's beta and the "best simple rule" are chosen on training
episodes and evaluated on validation episodes. Test episodes (seed 303) are
reserved for the learned-model comparison (H1) and are not used here.
"""

import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluate import paired_diff, run_policy, shortfall_bps
from market import MarketParams, Paths, simulate
from policies import TWAP, LQGReference, Planner, UrgencyRule

BASE = MarketParams()
SEEDS = {"train": 101, "validation": 202, "test": 303}
N_TRAIN, N_VALIDATION = 10_000, 20_000
BETAS = np.append(np.geomspace(0.001, 0.2, 24), np.inf)
RESULTS = Path(__file__).parent / "results"

PLANNER = "True model + planner"
LATEST = "True model, latest signal only + planner"
PRIVILEGED = "True trend (not achievable) + planner"


def cost(policy, paths, params):
    return shortfall_bps(run_policy(policy, paths, params), paths, params)


def tune_rules(params, train):
    """Choose beta for the urgency rule, then the better of TWAP and that rule, on training episodes."""
    twap_cost = cost(TWAP(params), train, params).mean()
    rule_costs = [cost(UrgencyRule(params, b), train, params).mean() for b in BETAS]
    beta = float(BETAS[int(np.argmin(rule_costs))])
    best_rule = "TWAP" if twap_cost <= min(rule_costs) else "Urgency rule"
    return beta, best_rule


def evaluate_setting(params):
    train = simulate(params, N_TRAIN, SEEDS["train"])
    val = simulate(params, N_VALIDATION, SEEDS["validation"])
    beta, best_rule = tune_rules(params, train)
    lqg = LQGReference(params)
    costs = {
        "TWAP": cost(TWAP(params), val, params),
        "Urgency rule": cost(UrgencyRule(params, beta), val, params),
        "LQG reference": cost(lqg, val, params),
        PLANNER: cost(Planner(params), val, params),
        LATEST: cost(Planner(params, use_history=False), val, params),
    }
    privileged_paths = Paths(alpha=val.alpha, price=val.price, signal=val.alpha)
    costs[PRIVILEGED] = cost(Planner(params.with_(obs_noise=0.0)), privileged_paths, params)
    return {
        "costs": costs,
        "beta": beta,
        "best_rule": best_rule,
        "lqg_clip_rate": lqg.clipped / max(lqg.decisions, 1),
    }


def gate_a():
    rows, checks = [], []
    settings = {"No-signal control": BASE.with_(r2=0.0), "Base case": BASE}
    for label, params in settings.items():
        out = evaluate_setting(params)
        costs = out["costs"]
        ref_name = "TWAP" if params.r2 == 0 else PLANNER
        for name, c in costs.items():
            d = paired_diff(c, costs[ref_name])
            rows.append({
                "setting": label, "policy": name, "mean_bps": c.mean(), "reference": ref_name,
                "diff_vs_reference_bps": d.mean, "ci_low": d.low, "ci_high": d.high,
            })
        if params.r2 == 0:
            exact = float(np.max(np.abs(costs[PLANNER] - costs["TWAP"])))
            checks.append((f"{label}: planner equals TWAP exactly (max abs difference {exact:.1e} bps)", exact < 1e-9))
            beaten = [n for n, c in costs.items() if paired_diff(c, costs["TWAP"]).high < 0]
            checks.append((f"{label}: no policy beats TWAP beyond noise (beaten by: {beaten or 'none'})", not beaten))
        else:
            beaten = [n for n, c in costs.items()
                      if n != PRIVILEGED and paired_diff(c, costs[PLANNER]).high < 0]
            checks.append((f"{label}: no achievable policy beats the planner beyond noise (beaten by: {beaten or 'none'})", not beaten))
            ceiling = paired_diff(costs[PRIVILEGED], costs[PLANNER])
            checks.append((f"{label}: privileged ceiling beats the planner ({ceiling.mean:+.3f} bps)", ceiling.high < 0))
            checks.append((f"{label}: LQG reference action clipped in {out['lqg_clip_rate']:.1%} of decisions (information only)", True))
        checks.append((f"{label}: urgency-rule beta tuned on training = {out['beta']:.3g}; best simple rule = {out['best_rule']}", True))
    return rows, checks


def gate_b():
    settings = []
    for r2 in [0.0025, 0.005, 0.01, 0.02, 0.05]:
        for half_life in [3.0, 10.0, 30.0]:
            settings.append(("grid", BASE.with_(r2=r2, half_life=half_life)))
    for obs_noise in [0.0, 0.5, 2.0, 4.0]:
        settings.append(("obs_noise sweep", BASE.with_(obs_noise=obs_noise)))
    for impact in [10.0, 100.0]:
        settings.append(("impact sweep", BASE.with_(impact=impact)))

    rows = []
    for group, params in settings:
        out = evaluate_setting(params)
        costs = out["costs"]
        best = costs[out["best_rule"]]
        headroom = paired_diff(best, costs[PLANNER])
        vs_twap = paired_diff(costs["TWAP"], costs[PLANNER])
        memory = paired_diff(costs[LATEST], costs[PLANNER])
        ceiling = paired_diff(costs[PLANNER], costs[PRIVILEGED])
        rows.append({
            "group": group, "r2": params.r2, "half_life": params.half_life,
            "obs_noise": params.obs_noise, "impact": params.impact,
            "alpha_std_bps": params.alpha_std,
            "twap_bps": costs["TWAP"].mean(), "best_rule": out["best_rule"], "beta": out["beta"],
            "best_rule_bps": best.mean(), "planner_bps": costs[PLANNER].mean(),
            "headroom_bps": headroom.mean, "headroom_low": headroom.low, "headroom_high": headroom.high,
            "headroom_vs_twap_bps": vs_twap.mean,
            "memory_gain_bps": memory.mean, "memory_low": memory.low, "memory_high": memory.high,
            "gap_to_privileged_bps": ceiling.mean, "lqg_clip_rate": out["lqg_clip_rate"],
            "passes_gate_b": headroom.mean > 2 * headroom.se,
        })
        print(f"  {group:16s} r2={params.r2:<7g} half_life={params.half_life:<5g} obs_noise={params.obs_noise:<4g} "
              f"impact={params.impact:<5g} headroom={headroom.mean:+.3f} bps", flush=True)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, a_rows, a_checks, b_rows, elapsed):
    lines = [
        "# Stage 1 results (generated by run_stage1.py)",
        "",
        f"Base parameters: `{asdict(BASE)}`. Seeds: train {SEEDS['train']} ({N_TRAIN:,} episodes), "
        f"validation {SEEDS['validation']} ({N_VALIDATION:,} episodes); test seed {SEEDS['test']} not used in Stage 1. "
        f"Costs are mean implementation shortfall in bps of arrival notional, including the half-spread. "
        f"Intervals are paired, normal-approximation 95%. Runtime {elapsed:.0f} s.",
        "",
        "## Gate A — environment sanity",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {text}" for text, ok in a_checks]
    lines += ["", "| Setting | Policy | Mean cost (bps) | Difference vs reference (bps) | 95% CI |", "|---|---|---:|---:|---|"]
    for r in a_rows:
        lines.append(f"| {r['setting']} | {r['policy']} | {r['mean_bps']:.3f} | {r['diff_vs_reference_bps']:+.3f} vs {r['reference']} "
                     f"| [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
    lines += [
        "", "## Gate B — headroom of the true-model planner over the best simple rule", "",
        "Headroom = best simple rule cost − planner cost (positive means the planner is cheaper). "
        "Memory gain = latest-signal-only planner cost − planner cost. "
        "Gap to privileged = planner cost − privileged planner cost.", "",
        "| Group | R² | Half-life | Obs. noise | Impact | Best rule | Headroom (bps) [95% CI] | vs TWAP | Memory gain | Gap to privileged | LQG clip rate | Gate B |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for r in b_rows:
        lines.append(
            f"| {r['group']} | {r['r2']:g} | {r['half_life']:g} | {r['obs_noise']:g} | {r['impact']:g} | {r['best_rule']} "
            f"| {r['headroom_bps']:+.3f} [{r['headroom_low']:+.3f}, {r['headroom_high']:+.3f}] | {r['headroom_vs_twap_bps']:+.3f} "
            f"| {r['memory_gain_bps']:+.3f} | {r['gap_to_privileged_bps']:+.3f} | {r['lqg_clip_rate']:.1%} "
            f"| {'pass' if r['passes_gate_b'] else 'fail'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    print("Gate A ...", flush=True)
    a_rows, a_checks = gate_a()
    for text, ok in a_checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {text}")
    print("Gate B ...", flush=True)
    b_rows = gate_b()
    write_csv(RESULTS / "gate_a.csv", a_rows)
    write_csv(RESULTS / "gate_b.csv", b_rows)
    write_markdown(RESULTS / "stage1.md", a_rows, a_checks, b_rows, time.time() - start)
    print(f"Done in {time.time() - start:.0f} s. See {RESULTS / 'stage1.md'}")


if __name__ == "__main__":
    main()
