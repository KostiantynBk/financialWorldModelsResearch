# Execution planning with a market model

**Status:** Stage 1 (simulator checks and planning headroom) completed on 25 November 2026. Stage 2 (a learned world model and a fitted classical model) completed on 27 November 2026. All results come from a synthetic market with illustrative parameters. They say nothing about real markets or live trading. The staged [research plan](PLAN.md) is updated as results arrive.

**Terminology:** Stage 1's planner uses a *hand-specified model*: our own equations with the simulator's true parameters. In this repository, *world model* means a model learned from data; the first learned model is tested in Stage 2.

## Question

This experiment applies the repository's central question to one decision: **buying a fixed quantity before a deadline.** Does a planner that uses a market model execute the order more cheaply than the best simple rule? Under which conditions?

The test problem is adapted from Hudson River Trading's write-up on [signal-driven optimal execution](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-hrt-ai-labs-project-2026/) (Erginbas, 2026). That write-up compares model-free reinforcement-learning policies with a hand-crafted rule. Here, HRT's rule is one of the baselines; the research question is not a replication of HRT's study.

Stage 1 gives the planner the **true** market model. This establishes a reference before any model is learned. Stage 2 gives the same planner a model learned from data and a classical model fitted to the same data, and compares both against this reference, which separates model error from planner error.

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

Model-based strategies are named *predictor + planner*. Every one uses the same planner and the same assumed cost model (spread and impact); only the market forecast differs.

| Strategy | What it does | Trained? |
|---|---|---|
| TWAP | Buys the remaining quantity evenly over the remaining steps | No |
| Urgency rule | HRT's hand-crafted rule: buy everything left if the required pace exceeds β·σ; otherwise buy at the even pace when the signal is positive, and wait when it isn't | β is tuned on training episodes |
| LQG reference | Closed-form optimum (Kalman filter plus Riccati recursion) if trades could be negative. Its actions are clipped to the allowed range. | No |
| **True model + planner** | At each step, forecasts the drift with the simulator's true equations and parameters, finds the cheapest remaining schedule with no selling, then executes the first trade | No |
| True model, latest signal only + planner | Same, but estimates the drift from the current signal alone, without history | No |
| True trend (not achievable) + planner | Sees the true drift, which no real policy can. It shows how much better drift estimates could help. | No |
| **Fitted classical model + planner** (Stage 2) | The true model's equations, with the fade rate, trend size and noise levels estimated from training episodes by maximum likelihood | Four parameters fitted |
| **Learned model + planner** (Stage 2) | A small neural network that reads price changes and signals minute by minute, builds its own state, and forecasts future price changes. It is given none of the simulator's equations. | Trained on training episodes |

For risk-neutral costs with linear impact, the planner's schedule is the Euclidean projection of the forecast drift to the deadline, scaled by 1/(2κ), onto the set of non-negative schedules that complete the order. It therefore needs no iterative optimization.

**The learned model** ([world_model.py](world_model.py), 1,666 parameters) has three parts:
- a recurrent encoder (GRU, 16 units) that builds a state from all observations so far;
- a learned latent transition (a residual two-layer network on an 8-dimensional latent state) that is rolled forward to the deadline;
- a linear decoder that turns each latent state into a predicted price change and signal.

It is trained self-supervised to predict future observations at all horizons by mean squared error, using Adam, with early stopping on 20% of the training episodes held out. One structural assumption is imposed, which the classical model shares: mirroring every price change and signal mirrors the forecast. The classical model is fitted with the Kalman filter's exact likelihood ([fit_classical.py](fit_classical.py)).

## How to run

```bash
cd experiments/execution-planning
pip install -r requirements.txt
python -m pytest -q        # 16 correctness tests
python run_stage1.py       # about 3 minutes on a laptop CPU; writes results/
python run_stage2.py       # about 23 minutes on a laptop CPU; writes results/stage2*
```

Tested with Python 3.14, NumPy 2.4, SciPy 1.17, PyTorch 2.12 (CPU) and pytest 9.1. Seeds: Stage 1 training 101 (10,000 episodes) and validation 202 (20,000 episodes). Stage 2 training seeds 101–105 (one per run) and test seed 303 (20,000 episodes), first used in Stage 2.

**Tests:**
- The cost formula matches a summation-by-parts identity.
- The planner equals the closed-form optimum exactly when the constraint does not bind.
- Without a signal, the planner and the closed-form optimum both reduce to TWAP.
- Changing future prices or signals never changes earlier actions (no look-ahead). This also holds for the learned model + planner.
- The Kalman filter estimates the drift more accurately than the latest signal alone.
- Stage 2's planner reproduces Stage 1's planner exactly when given the true model.
- The fitted model's likelihood matches a direct multivariate-normal calculation, and maximum likelihood recovers the true parameters from 1,000 episodes.
- The learned model's forecast is sign-symmetric, and multi-horizon forecasts are consistent.

## Stage 1 results

Full tables are in [results/stage1.md](results/stage1.md), [results/gate_a.csv](results/gate_a.csv) and [results/gate_b.csv](results/gate_b.csv). The urgency rule's β and the best simple rule were selected on training episodes. Costs below are measured on validation episodes, with paired 95% intervals.

**Gate A (simulator checks): passed.**
- **No signal:** every model-based policy reduces exactly to TWAP. The tuned urgency rule costs 1.29 bps more [1.18, 1.39], because it waits on noise and then has to catch up.
- **Base case:** no achievable policy beats the true-model planner.

| Base case | Mean cost (bps) | vs True model + planner (bps) [95% CI] |
|---|---:|---|
| TWAP (best simple rule on training) | 1.46 | +2.87 [+2.69, +3.05] |
| Urgency rule (β = 0.1) | 1.96 | +3.37 [+3.16, +3.58] |
| LQG reference, clipped | −1.16 | +0.25 [+0.23, +0.27] |
| **True model + planner** | **−1.41** | — |
| True model, latest signal only + planner | −0.79 | +0.62 [+0.56, +0.68] |
| True trend (not achievable) + planner | −2.05 | −0.64 [−0.71, −0.58] |

**Gate B (headroom): passed in all 21 settings tested.** The true-model planner beats the best simple rule by more than twice the standard error in every setting.
- **Signal strength and persistence:** headroom grows from +0.28 bps (R² 0.25%, half-life 3 steps) to +10.2 bps (R² 5%, half-life 30 steps).
- **Best simple rule:** TWAP in 10 of the 15 grid settings. The tuned urgency rule won only with stronger signals, and in the low-impact sweep.
- **Memory:** using the signal's history beats using only the latest signal by +0.62 bps in the base case. The gain is zero when the signal is observed without noise, and largest for strong, persistent signals (+4.2 bps at R² 5%, half-life 30).

**What to take from this:**
1. Planning can only help if the signal is informative. Here, the gain over TWAP depends mainly on how predictive the signal is and how long it lasts. Real values for both are unknown until we measure them.
2. The no-sell constraint matters: the closed-form action had to be clipped in 44% of base-case decisions. The constrained planner is therefore the best reference available, **not a proven optimum**. A scenario-based planner could do better.
3. Stage 1 says nothing about learned models. Stage 2 asks how much of this headroom survives when the planner uses a model fitted to limited data.

## Stage 2 results

Full tables are in [results/stage2.md](results/stage2.md), with per-run values in [results/stage2_runs.csv](results/stage2_runs.csv). Five runs per training size; each run draws its own training hours (seeds 101–105) and initializes the network differently. Everything that learns or tunes sees only the training hours. Costs are measured on 20,000 test hours from the reserved seed 303. The best simple rule, selected on training hours, was TWAP in all 15 runs.

| Training hours | Best simple rule (TWAP) | Fitted classical model + planner | Learned model + planner | True model + planner (ceiling) |
|---:|---:|---:|---:|---:|
| 50 | +1.38 | −1.56 | −0.96 | −1.53 |
| 200 | +1.38 | −1.54 | −1.47 | −1.53 |
| 1,000 | +1.38 | −1.53 | −1.47 | −1.53 |

*Mean cost in bps, lower is better, averaged over runs.* The true model's headroom over TWAP on the test hours is 2.91 bps.

| Training hours | Learned saving vs best rule [95% CI] | Share of headroom | Range over runs | Learned − fitted [95% CI] |
|---:|---|---:|---|---|
| 50 | +2.34 [+2.21, +2.47] | 80% | +1.44 to +2.80 | +0.60 [+0.54, +0.66] |
| 200 | +2.86 [+2.68, +3.03] | 98% | +2.76 to +2.97 | +0.06 [+0.05, +0.07] |
| 1,000 | +2.85 [+2.67, +3.04] | 98% | +2.82 to +2.90 | +0.06 [+0.05, +0.07] |

*Intervals are paired over test hours for the run-averaged costs; run-to-run variation is shown separately.*

**Hypotheses (prespecified in the [plan](PLAN.md)):**
- **H1, learned model + planner beats the best simple rule: supported** at all three training sizes. Every one of the 15 trained models beat TWAP.
- **H3, smaller forecast error means smaller decision loss: supported, with a plateau.** The learned model's cost above the ceiling fell from 0.57 to 0.06 bps as its forecast moved closer to the true model's (RMSE to the true forecast 5.0 → 1.8 bps). From 200 to 1,000 hours the forecast still improved (RMSE 1.8 → 1.2), but the cost gap stayed at 0.06 bps. Across the 15 trained models, forecast error and decision loss were rank-correlated (Spearman 0.66).
- **H4, the learned model matches the fitted classical model when the classical assumptions hold: not supported in the strict sense.** The fitted model was cheaper at every training size. From 200 hours the gap is small (0.06 bps, about 2% of the headroom) but clearly nonzero. No equivalence margin was prespecified. The second half of H4, whether learning wins when the assumptions are wrong, is Stage 3's question.

**Forecast quality (R² on test hours, relative to forecasting no change):**

| Forecast | 1 min | 5 min | 15 min | To deadline |
|---|---:|---:|---:|---:|
| True model (best achievable) | 0.0074 | 0.0273 | 0.0426 | 0.0306 |
| Fitted classical model, 50 hours | 0.0074 | 0.0272 | 0.0423 | 0.0300 |
| Learned model, 50 hours | 0.0037 | 0.0210 | 0.0307 | 0.0000 |
| Learned model, 1,000 hours | 0.0070 | 0.0266 | 0.0418 | 0.0289 |

**Tail cost (CVaR95, the mean cost of the worst 5% of test hours):** TWAP 51.6 bps, true model + planner 41.2, fitted classical model + planner 41.3–41.4. The learned model + planner averaged 42.9, 42.2 and 41.7 bps at 50, 200 and 1,000 hours. Its 50-hour runs ranged from 38.7 to 50.9 bps.

**Training budget:** 1,666 parameters; 375–1,775 gradient steps and 22–144 seconds per model on a laptop CPU, with early stopping.

**What to take from this:**
1. **A small learned model recovers most of the planning value without being told the market's equations:** 80% of the headroom from 50 simulated hours and 98% from 200.
2. **When the classical model's assumptions are exactly right, it is the better choice.** With four parameters to fit, it reaches the ceiling from 50 hours and is cheaper than the learned model at every size. Any case for learning has to come from markets where those assumptions fail, which Stage 3 tests.
3. **Forecast scores and decision value can disagree.** At 50 hours, the learned model's R² to the deadline averaged zero, yet its planner captured 80% of the headroom. Across the 15 models, the 5-minute R² tracked cost most closely (Spearman −0.83, against −0.66 for R² to the deadline). This is exploratory: the training sets are nested and the sample is small.
4. **Small data is fragile.** One of the five 50-hour models captured only 49% of the headroom, and its tail cost was close to TWAP's.
5. **The true-model planner is not exactly optimal.** At 50 hours, the fitted classical model + planner was 0.02 bps cheaper than the true model + planner [0.02, 0.03]. The fitted models' half-lives were slightly too short on average (9.2 steps against 10). This is consistent with Stage 1's caveat that the certainty-equivalent planner is not proven optimal under the no-sell constraint. The cause is untested.

## Limitations

- **Parameters:** illustrative and uncalibrated. Headroom depends strongly on signal strength and persistence.
- **Costs:** stylized. Constant half-spread, linear temporary impact, no permanent impact, order book, fees or latency; single asset, buying only.
- **Objective:** risk-neutral (mean cost only, no risk aversion).
- **Planner:** plans on the expected drift path (certainty equivalence). Under a binding constraint this is not proven optimal.
- **Statistics:** confidence intervals use a normal approximation over independent episodes. Stage 1 uses one validation seed. In Stage 2, intervals for run-averaged costs exclude run-to-run variation, which is shown separately over only five runs.
- **Learned model:** one small architecture with no hyperparameter search. Its settings were chosen in development runs on separate seeds (training 1000, evaluation 304). It gives point forecasts only, so the plan's likelihood-based forecast scores and interval calibration are not reported.
- **Comparison design:** in Stage 2 the classical model has exactly the simulator's form, which favors it by construction.
- **Test seed use:** before the reported run, a code check ran on 500 test hours with deliberately undertrained models, and an aborted first launch computed only the TWAP and true-model test costs. No setting was changed afterwards.
- **Scope:** synthetic results only. They are not evidence about real markets or live trading.

## Next step (Stage 3)

Stage 3 keeps the same strategies but breaks the classical model's assumptions in the simulator: regime switches in signal strength, fat-tailed price shocks and drifting signal quality. The question is whether the learned model, which assumes no equations, then beats the fitted classical model (the second half of H4). See the [research plan](PLAN.md) for the hypotheses and later stages.
