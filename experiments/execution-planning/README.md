# Execution planning with a market model

**Status:** Stage 1 (simulator checks and planning headroom) completed on 25 November 2026. No learned model has been tested yet. All results come from a synthetic market with illustrative parameters. They say nothing about real markets or live trading. The staged [research plan](PLAN.md) is updated as results arrive.

**Terminology:** Stage 1's planner uses a *hand-specified model*: our own equations with the simulator's true parameters. In this repository, *world model* means a model learned from data; learned models start in Stage 2.

## Question

This experiment applies the repository's central question to one decision: **buying a fixed quantity before a deadline.** Does a planner that uses a market model execute the order more cheaply than the best simple rule? Under which conditions?

The test problem is adapted from Hudson River Trading's write-up on [signal-driven optimal execution](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/) (Erginbas, 2026). That write-up compares model-free reinforcement-learning policies with a hand-crafted rule. Here, HRT's rule is one of the baselines; the research question is not a replication of HRT's study.

Stage 1 gives the planner the **true** market model. This establishes a reference before any model is learned. Later stages will compare a planner using a learned model against this reference, which separates model error from planner error.

## What is observed, predicted, and decided

| | In this experiment |
|---|---|
| **Observed** at each step | The mid-price and a noisy short-term signal, up to the current step |
| **Predicted** | The hidden drift (expected price change) over the remaining steps |
| **Action** | How much to buy this step: between zero and the quantity still to buy. The last step buys whatever remains. |
| **Decision improved** | The execution schedule. Cost is implementation shortfall against the arrival mid-price. |
| **Market impact** | Not modeled. The price path is exogenous; our trades pay the spread and a temporary impact cost but never move future prices. |

## Simulator

Prices are in basis points (bps) of the arrival mid-price, so the starting price is 0. Quantities are fractions of the order. For steps $$t = 0, \dots, N-1$$:

$$
S_{t+1} = S_t + \alpha_t + \sigma \varepsilon_{t+1}, \qquad
\alpha_{t+1} = \phi\, \alpha_t + \sigma_\alpha \eta_{t+1}, \qquad
z_t = \alpha_t + \sigma_z \xi_t
$$

Buying $$v_t$$ at step $$t$$ costs $$v_t (S_t + c + \kappa v_t)$$. The policy sees $$S_t$$ and $$z_t$$, never $$\alpha_t$$. Every policy runs on the same simulated paths.

| Parameter | Base value | Meaning |
|---|---|---|
| `n_steps` (N) | 60 | Decision steps, for example one hour in one-minute steps |
| `sigma` (σ) | 5 bps | Mid-price noise per step |
| `half_spread` (c) | 1 bp | Paid on every unit bought |
| `impact` (κ) | 30 bps | Buying the whole order in one step would cost an extra 30 bps |
| `r2` | 1% | Share of one-step return variance explained by the drift |
| `half_life` | 10 steps | How quickly the drift decays; this sets φ |
| `obs_noise` | 1 | Signal noise relative to the drift's typical size |

These values are **illustrative, not calibrated.** Calibrating them to real data, for example public crypto order-flow data, is a planned next step.

## Policies

| Policy | What it does | Trained? |
|---|---|---|
| TWAP | Buys the remaining quantity evenly over the remaining steps | No |
| Urgency rule | HRT's hand-crafted rule: buy everything left if the required pace exceeds β·σ; otherwise buy at the even pace when the signal is positive, and wait when it isn't | β is tuned on training episodes |
| LQG reference | Closed-form optimum (Kalman filter plus Riccati recursion) if trades could be negative. Its actions are clipped to the allowed range. | No |
| **Planner** | At each step, forecasts the drift with the true model and finds the cheapest remaining schedule with no selling, then executes the first trade | No |
| Planner (latest signal only) | Same, but estimates the drift from the current signal alone, without history | No |
| Privileged planner | Sees the true drift, which no real policy can. It shows how much better drift estimates could help. | No |

For risk-neutral costs with linear impact, the planner's schedule is the Euclidean projection of the forecast drift to the deadline, scaled by 1/(2κ), onto the set of non-negative schedules that complete the order. It therefore needs no iterative optimization.

## How to run

```bash
cd experiments/execution-planning
python -m pytest -q        # 10 correctness tests
python run_stage1.py       # about 3 minutes on a laptop CPU; writes results/
```

Tested with Python 3.14, NumPy 2.4 and pytest 9.1. Seeds: training 101 (10,000 episodes), validation 202 (20,000 episodes). The test seed 303 is reserved for the learned-model comparison and has not been used.

**Tests:**
- The cost formula matches a summation-by-parts identity.
- The planner equals the closed-form optimum exactly when the constraint does not bind.
- Without a signal, the planner and the closed-form optimum both reduce to TWAP.
- Changing future prices or signals never changes earlier actions (no look-ahead).
- The Kalman filter estimates the drift more accurately than the latest signal alone.

## Stage 1 results

Full tables are in [results/stage1.md](results/stage1.md), [results/gate_a.csv](results/gate_a.csv) and [results/gate_b.csv](results/gate_b.csv). The urgency rule's β and the best simple rule were selected on training episodes. Costs below are measured on validation episodes, with paired 95% intervals.

**Gate A (simulator checks): passed.**
- **No signal:** every model-based policy reduces exactly to TWAP. The tuned urgency rule costs 1.29 bps more [1.18, 1.39], because it waits on noise and then has to catch up.
- **Base case:** no achievable policy beats the true-model planner.

| Base case | Mean cost (bps) | vs Planner (bps) [95% CI] |
|---|---:|---|
| TWAP (best simple rule on training) | 1.46 | +2.87 [+2.69, +3.05] |
| Urgency rule (β = 0.1) | 1.96 | +3.37 [+3.16, +3.58] |
| LQG reference, clipped | −1.16 | +0.25 [+0.23, +0.27] |
| **Planner (true model)** | **−1.41** | — |
| Planner (latest signal only) | −0.79 | +0.62 [+0.56, +0.68] |
| Privileged planner (not achievable) | −2.05 | −0.64 [−0.71, −0.58] |

**Gate B (headroom): passed in all 21 settings tested.** The true-model planner beats the best simple rule by more than twice the standard error in every setting.
- **Signal strength and persistence:** headroom grows from +0.28 bps (R² 0.25%, half-life 3 steps) to +10.2 bps (R² 5%, half-life 30 steps).
- **Best simple rule:** TWAP in 10 of the 15 grid settings. The tuned urgency rule won only with stronger signals, and in the low-impact sweep.
- **Memory:** using the signal's history beats using only the latest signal by +0.62 bps in the base case. The gain is zero when the signal is observed without noise, and largest for strong, persistent signals (+4.2 bps at R² 5%, half-life 30).

**What to take from this:**
1. Planning can only help if the signal is informative. Here, the gain over TWAP depends mainly on how predictive the signal is and how long it lasts. Real values for both are unknown until we measure them.
2. The no-sell constraint matters: the closed-form action had to be clipped in 44% of base-case decisions. The constrained planner is therefore the best reference available, **not a proven optimum**. A scenario-based planner could do better.
3. Stage 1 says nothing yet about learned models. The question for Stage 2 is how much of this headroom survives when the planner uses a model fitted to limited data.

## Limitations

- **Parameters:** illustrative and uncalibrated. Headroom depends strongly on signal strength and persistence.
- **Costs:** stylized. Constant half-spread, linear temporary impact, no permanent impact, order book, fees or latency; single asset, buying only.
- **Objective:** risk-neutral (mean cost only, no risk aversion).
- **Planner:** plans on the expected drift path (certainty equivalence). Under a binding constraint this is not proven optimal.
- **Statistics:** confidence intervals use a normal approximation over independent episodes, from one validation seed.
- **Scope:** synthetic results only. They are not evidence about real markets or live trading.

## Next step (Stage 2)

Stage 2 replaces the hand-specified model with a learned world model: a small recurrent neural network trained on 50, 200 or 1,000 simulated price-and-signal histories. The planner, referee and cost model stay the same. On the reserved test episodes, the learned model is compared with three references:
- a fitted hand-specified model
- the true-model ceiling
- the best simple rule

Forecast accuracy is reported separately from execution cost. See the [research plan](PLAN.md) for the hypotheses and later stages.
