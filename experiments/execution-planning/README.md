# Execution planning with a market model

**Status:** Stage 1 (simulator checks and planning headroom) completed on 25 November 2026. Stage 2 (a learned world model and a fitted classical model) completed on 27 November 2026. Stage 3 (markets that break the classical model's assumptions) completed on 28 November 2026. A calibration on real crypto data (28 November 2026) found no measurable predictability in the public signals tested. All simulation results come from a synthetic market with illustrative parameters. They say nothing about real markets or live trading. The staged [research plan](PLAN.md) is updated as results arrive.

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
| Fitted regime-aware model + planner (Stage 3, scenarios A and B) | The correct two-regime structure of the scenario, with all parameters estimated by maximum likelihood | Six parameters fitted |
| Larger learned model + planner (Stage 3) | The learned model with a 32-unit memory and a 16-dimensional latent state (6,146 parameters), chosen on validation hours | Trained on training episodes |

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
python -m pytest -q        # 32 correctness tests
python run_stage1.py       # about 3 minutes on a laptop CPU; writes results/
python run_stage2.py       # about 23 minutes on a laptop CPU; writes results/stage2*
python probe_stage2.py     # about 30 minutes; Stage 2 supplement, writes results/stage2_probe*
python run_stage3.py gate      # Stage 3 Gate C1 on validation hours, about 3 minutes
python run_stage3.py select    # choose the larger learned model on validation hours, about 15 minutes
python run_stage3.py main A    # test run for one scenario (A, B, C or D), 1-2.5 hours each; run them in parallel
python run_stage3.py report    # writes results/stage3*
python calibrate_crypto.py     # real-data calibration: downloads public Binance data (about 50 MB), about 5 minutes
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
- The supplement's forecast comparison is exact for the true model and scales correctly.
- Stage 3: the scenario simulator reproduces Stage 1's simulator in the base case, and each scenario matches its design (average trend strength, equal signal variance across A's regimes, B's persistence, C's outlier variance and kurtosis).
- Stage 3: the best-possible forecaster equals the Kalman filter on the base market, ignores duplicated regimes, has no look-ahead, beats the misspecified Kalman filter in A, B and C, and agrees with a particle filter in C. The regime-aware fit recovers scenario A's parameters.
- Calibration: data parsing handles both timestamp units, gaps are filled, features use only the past, and on simulated data the pipeline recovers the theoretical next-minute R² (0.5%) and the trend's half-life.

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

### Stage 2 supplement: what did the learned model learn?

Exploratory. Full tables are in [results/stage2_probe.md](results/stage2_probe.md), from [probe_stage2.py](probe_stage2.py). The 15 learned models were retrained with the same data and seeds, and each reproduces its published test cost exactly. In simulation the hidden trend is known, so each model can be checked on 2,000 fresh hours.

**Storing the trend is easy: an untrained network does it too.** R² of a linear readout of the true trend:

| Source | Trend R² |
|---|---:|
| True model (best achievable) | 0.729 |
| Learned model's memory, any training size | 0.726–0.728 |
| **Untrained network's memory** (same architecture) | **0.727–0.728** |
| Latest 10 signals | 0.724 |
| Latest signal only | 0.504 |

The trend is close to a weighted average of recent signals, and any recurrent memory keeps those. A memory readout therefore cannot show what training added here.

**What training must get right is the forecast.** The learned model's 30-minute price forecast against the true model's (mean [min, max] over runs):

| Training hours | Correlation | Size (1 = right) | Relative error (0 = identical) |
|---:|---|---|---|
| 50 | 0.836 [0.730, 0.910] | 0.83 [0.23, 1.60] | 0.52 [0.21, 0.89] |
| 200 | 0.962 [0.937, 0.997] | 0.95 [0.62, 1.27] | 0.13 [0.01, 0.19] |
| 1,000 | 0.985 [0.951, 0.998] | 1.06 [0.99, 1.16] | 0.04 [0.01, 0.10] |

Across the 15 models, the forecast's relative error tracks the cost above the true model (Spearman +0.73, p = 0.002); the memory readout does not (−0.10, p = 0.71). The weakest 50-hour model forecast at a quarter of the right size, so its planner behaved almost like TWAP and captured 49% of the headroom. The best 50-hour model forecast at the right size and captured 96%.

**Lessons:**
1. Compare any readout of a learned model's internal state with an untrained network of the same architecture. A related caution appears in [Merchant et al. (2026)](https://arxiv.org/abs/2610.09048): no tested self-supervised encoder beat a frozen, randomly initialized one on their spread-change probe.
2. Judge a world model by the forecasts its planner uses, not by what its internal state contains.
3. With little data, the learned model's main failure was getting the size of its forecasts wrong, not missing the trend.

## Stage 3 results: when the classical model's assumptions are wrong

Prespecified in the [plan](PLAN.md#stage-3-prespecification) before any Stage 3 code was run. Full tables are in [results/stage3.md](results/stage3.md), with per-run values in [results/stage3_runs.csv](results/stage3_runs.csv).

**Scenarios.** Each keeps the base case's average signal strength and cost model, but breaks one assumption of the fitted classical model. Regimes are drawn once per hour.

| | Scenario | What breaks |
|---|---|---|
| A | Signal on/off | In half the hours the signal is pure noise; in the others it is twice as informative |
| B | Fast or slow trends | Trends fade with a half-life of 2 minutes in some hours and 60 in others (the severe setting; see Gate C1) |
| C | Signal outliers | 5% of signal readings carry 5 times larger noise (the severe setting) |
| D | Price outliers (negative control) | Outliers in prices instead; expected not to matter for decisions |

For each scenario, the *true model* is the best possible forecaster given the scenario's structure: posterior-weighted Kalman filters for A and B, and a Gaussian-sum filter for C and D that matched a particle filter's trend estimates (R² 0.797 against 0.797 in C).

**Gate C1 (misspecification headroom), on validation hours.** Learned models were trained only where the misspecified classical model fell at least 0.15 bps short of the best possible forecaster:
- A passed at its prespecified setting, with a gap of +0.24 bps [+0.18, +0.30].
- B and C fell short at their prespecified settings (+0.09 each) and passed at their prespecified severe settings (+0.15 and +0.16).
- D failed (+0.02), as predicted.

The violations that hurt the classical model most still cost it only 0.1–0.3 bps.

**Test results** (20,000 test hours, 5 runs; mean cost in bps, lower is better):

| Scenario | Training hours | Best simple rule | Fitted classical model + planner | Fitted regime-aware model + planner | Learned model + planner | Larger learned model + planner | True model + planner |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 200 | +1.56 | −0.37 | −0.65 | −0.22 | −0.11 | −0.67 |
| A | **1,000** | +1.37 | −0.40 | −0.67 | **−0.47** | −0.46 | −0.67 |
| A | 5,000 | +1.37 | −0.41 | −0.67 | −0.57 | −0.52 | −0.67 |
| B | 200 | +1.42 | −1.67 | −1.78 | −1.58 | −1.42 | −1.78 |
| B | **1,000** | +1.42 | −1.66 | −1.77 | **−1.64** | −1.66 | −1.78 |
| B | 5,000 | +1.42 | −1.67 | −1.78 | −1.71 | −1.69 | −1.78 |
| C | 200 | +1.40 | −1.42 | — | −1.15 | −0.97 | −1.54 |
| C | **1,000** | +1.40 | −1.43 | — | **−1.47** | −1.44 | −1.54 |
| C | 5,000 | +1.40 | −1.43 | — | −1.47 | −1.46 | −1.54 |
| D | 1,000 | +1.37 | −1.41 | — | −1.36 | — | −1.42 |

**Primary hypothesis H4b: the learned model + planner costs less than the fitted classical model + planner at 1,000 training hours.**

| Scenario | Learned − fitted classical (bps) [95% CI] | Bonferroni-corrected [98.3%] | Share of the classical model's shortfall recovered | Verdict |
|---|---|---|---:|---|
| A Signal on/off | −0.07 [−0.10, −0.05] | [−0.11, −0.04] | 27% | **Supported** |
| B Fast or slow trends | +0.02 [−0.01, +0.05] | [−0.02, +0.06] | none | **Not supported** |
| C Signal outliers | −0.04 [−0.07, −0.02] | [−0.07, −0.02] | 38% | **Supported** |

**Secondary results:**
- **More data helps the learned model; the misspecified classical model stays put.** With 5,000 hours, the learned model beats the fitted classical model in all three scenarios: A −0.17 [−0.20, −0.13], B −0.04 [−0.08, −0.00], C −0.03 [−0.06, −0.00]. With 200 hours it loses in all three (+0.09 to +0.27).
- **A correctly structured classical model beats learning.** Where it was tested (A and B), the fitted regime-aware model was cheaper than the learned model at every size: by 0.20 and 0.13 bps at 1,000 hours, and 0.10 and 0.06 at 5,000. It stays within about 0.01 bps of the true model.
- **The larger learned model did not help.** At 1,000 hours it was within 0.03 bps of the primary model, it was slightly worse at 5,000 hours, and it failed badly in two 200-hour runs.
- **Against the best simple rule,** every model-based strategy is cheaper, by 1.7–3.2 bps on average.
- **Negative control:** in D the learned model was slightly costlier than the classical model (+0.05 [+0.04, +0.06]), so there is no sign of spurious wins from planner artifacts.
- **Where the gain comes from in A:** the learned model gains mainly in quiet hours, where it trades less on a noise-only signal (+1.90 bps against the fitted classical model's +2.07; the true model gets +1.55). In active hours, the two are nearly level.
- **Tail cost:** the learned model's CVaR95 was similar to or below the true model's (for example 41.2 against 43.3 bps in A at 1,000 hours), because the true-model planner trades more aggressively on its forecasts. The objective is mean cost, so this is reported, not optimized.

**What to take from this:**
1. **Learning beats a hand-specified model that is wrong in ways that matter, but the margin is small.** With 1,000 training hours it recovers 27% (A) and 38% (C) of the classical model's shortfall, worth 0.04–0.07 bps, about $4–7 per $1M order. In B, 1,000 hours were not enough.
2. **Knowing the right structure beats learning it.** A classical model with the correct regime structure was better still. The learned model's value is that it does not need the structure to be known in advance; it still pays for that in data.
3. **Learning gains grow with data.** From 1,000 to 5,000 hours the learned model moved closer to the ceiling in A and B (in C it stayed about 0.08 bps short), while the misspecified classical model could not improve.
4. **These gaps are small relative to the gain from planning itself** (about 2–3 bps over the best simple rule). The bigger question for real markets remains whether real signals are predictive enough, which the planned calibration on real data measures.

## Calibration on real data: how predictive are public signals?

Stages 1–3 used an invented signal. The planner can only gain if real signals predict the next minutes' price moves. This calibration measures that on real data, with the same statistics computed on the simulator so that real markets can be placed in the Stage 1 headroom table. The analysis choices were fixed in [calibrate_crypto.py](calibrate_crypto.py) before any result was seen; one diagnostic, marked as such, was added afterwards. Full tables are in [results/calibration.md](results/calibration.md).

**Data:**
- Binance spot BTCUSDT and ETHUSDT one-minute bars from the public archive ([data.binance.vision](https://data.binance.vision)), November 2025 to October 2026: 525,600 minutes per market, with no gaps. Every file's checksum was verified.
- The raw data is downloaded by the script and not redistributed here.
- Fit period: November 2025 to June 2026. Evaluation period: July to October 2026.

**Signals known at the close of each minute:**
- *trade imbalance*: aggressive buy volume minus aggressive sell volume, over total volume;
- the last one-minute return;
- the last 30 minutes of both ("history").

Each is fitted by least squares on the fit period and scored out of sample. Intervals are 95% block-bootstrap intervals over days.

| | BTCUSDT | ETHUSDT | Simulator, base case | Simulator, weakest grid cell (R² 0.25%, half-life 3) |
|---|---|---|---|---|
| One-minute volatility (bps), fit / evaluation | 6.7 / 4.8 | 9.2 / 6.6 | 5.0 | 5.0 |
| Same-minute R²: the imbalance explaining its own minute's return | 16.9% [13.7, 19.6] | 10.3% [8.0, 12.6] | — | — |
| **Next-minute R² from the imbalance** | **+0.005%** [+0.001, +0.010] | **−0.004%** [−0.007, +0.000] | +0.49% | +0.11% |
| History R², 5 / 30 minutes ahead | −0.03% / −0.20% | −0.01% / −0.17% | +2.8% / +4.1% | +0.23% / +0.07% |
| Stage 1 planning headroom over the best simple rule | not estimable | not estimable | +2.87 bps | +0.28 bps |

**Diagnostic (added after the first results):** the negative values grow with the horizon because each fitted intercept carried the fit period's average drift (falling prices) into a rising evaluation period. Refitted without an intercept, every predictor scores about zero. The largest value is BTC's next-minute imbalance at +0.009% [+0.007, +0.013]; all other intervals lie within about ±0.26% and mostly cover zero.

**What to take from this:**
1. **The imbalance describes the present, not the future.** It explains 10–17% of the price move in its own minute, consistent with [Cont, Kukanov & Stoikov (2014)](https://arxiv.org/abs/1011.6402), but essentially nothing about the following minutes.
2. **Measured predictability is at least ten times weaker than the weakest Stage 1 setting** and about fifty times weaker than the base case. In the Stage 1 table, even the weakest setting left only +0.28 bps of planning headroom. With these signals, a planner has no measurable drift to exploit over a one-hour order, so Stages 1–3's gains would not appear.
3. **Volatility is in line with the simulator** (4.8–9.2 bps per minute against 5).
4. **This is a statement about these signals, not about markets in general.** It does not rule out predictability:
   - at higher frequency (seconds), where order-flow effects are known to decay fast; the predictive slope here halves within about one minute and then changes sign;
   - in full order-book data or other signals;
   - in the proprietary signals that firms such as HRT use.

## Limitations

- **Parameters:** illustrative and uncalibrated. Headroom depends strongly on signal strength and persistence.
- **Costs:** stylized. Constant half-spread, linear temporary impact, no permanent impact, order book, fees or latency; single asset, buying only.
- **Objective:** risk-neutral (mean cost only, no risk aversion).
- **Planner:** plans on the expected drift path (certainty equivalence). Under a binding constraint this is not proven optimal.
- **Statistics:** confidence intervals use a normal approximation over independent episodes. Stage 1 uses one validation seed. In Stage 2, intervals for run-averaged costs exclude run-to-run variation, which is shown separately over only five runs.
- **Learned model:** one small architecture with no hyperparameter search. Its settings were chosen in development runs on separate seeds (training 1000, evaluation 304). It gives point forecasts only, so the plan's likelihood-based forecast scores and interval calibration are not reported.
- **Comparison design:** in Stage 2 the classical model has exactly the simulator's form, which favors it by construction. In Stage 3 the scenarios are our own choice of violations; other violations could favor learning more or less.
- **Stage 3 scenarios:** regimes reset every hour, and strategies get no data from earlier hours, which makes regime detection harder than with persistent regimes. B and C ran at their prespecified severe settings because the prespecified settings did not leave enough headroom (Gate C1). Forecast metrics use the first 5,000 of the 20,000 test hours.
- **Stage 3 test-seed use:** an aborted launch ran for a few seconds before the reported run and produced no results.
- **Test seed use:** before the reported run, a code check ran on 500 test hours with deliberately undertrained models, and an aborted first launch computed only the TWAP and true-model test costs. No setting was changed afterwards.
- **Calibration:** two markets on one venue, one year, one-minute bars, two public signals and linear predictors. The spread is not measured: in this cost model every strategy buys the same total quantity, so the spread is the same for all and does not affect headroom. Market impact cannot be measured from public data.
- **Scope:** the execution results are synthetic, and the calibration is descriptive. Neither is evidence about live trading.

## Next step

Stages 2 and 3 complete the synthetic comparison of learned and hand-specified models, and the calibration shows that the public one-minute signals tested carry no exploitable drift. Gate C in the [research plan](PLAN.md) decides what comes next. A real-data execution stage with these signals would be expected to show no planning gain. Options include richer or higher-frequency signals, or applying world models where real targets are predictable, such as volatility and risk.
